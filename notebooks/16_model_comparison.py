"""
Phase 16 -- comparing the models. Everything so far uses LightGBM for both
learned components (03's tyre model, 05's pit classifier). This asks whether
that choice matters: the same data, folds and metrics for a few different
model families, then each alternative pushed through the backtest.

  tyre        leave-one-circuit-out MAE (03's evaluation) for each tyre model;
              with --cv 5, grouped 5-fold by circuit instead
  classifier  rolling-origin evaluation (05's, each season predicted from
              earlier races only) for each pit classifier
  downstream  each alternative swapped into 06's time-scored rolling backtest:
              tyre models with the deployed classifier's predictions, pit
              classifiers with the deployed tyre model

Nothing here touches the deployed or frozen models (data/models/*.joblib):
alternatives are fitted in memory and the results go to
data/model_comparison/. The "vs real" columns of the downstream stage are
paired within one tyre model (real strategies and candidates are priced by
the same model), so they are comparable across models; the "vs optimal"
columns are not, because 04's hindsight optimum is always priced by the
deployed LightGBM.

Run with: python notebooks/16_model_comparison.py [tyre [--cv 5]|classifier|downstream|rescore|all]
(rescore recomputes the classifier and downstream summaries from the saved files.)
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dm = _load('degradation_model_lib', '03_ml_degradation_model.py')
cl = _load('trigger_classifier_lib', '05_trigger_classifier.py')

OUT_DIR = Path('data/model_comparison')
SILVER_FILE = Path('data/silver/laps_ml_features.parquet')
N_BOOTSTRAP = 1000
KFOLD_REPEATS = 3
DEPLOYED = 'LightGBM (deployed)'
# Forward-test races (docs/forward_test.md) stay out of every comparison here:
# choosing a model from them would spend the only truly unseen races.
HOLDOUT_FROM = '2026_r17'


class Est:
    """One interface over LightGBM and scikit-learn estimators: fit(X, y, w),
    predict(X) for regressors, predict_proba(X) -> P(class 1) for classifiers."""

    def __init__(self, model, weight_arg='sample_weight', **fit_kw):
        self.model, self.weight_arg, self.fit_kw = model, weight_arg, fit_kw

    def fit(self, X, y, w):
        self.model.fit(X, y, **{self.weight_arg: np.asarray(w)}, **self.fit_kw)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def bootstrap_ci(values, seed=0):
    return tuple(np.percentile(values, [2.5, 97.5]))


# --- Tyre models -------------------------------------------------------------

COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
NUMERIC = [f for f in dm.FEATURE_COLS if f not in dm.CAT_FEATURES]


def _age_by_compound(X: pd.DataFrame) -> pd.DataFrame:
    """Adds a tyre-age slope per compound, so the linear model can wear each compound differently."""
    X = X.copy()
    for c in COMPOUNDS:
        X[f'age_{c}'] = X['TyreLife'] * (X['Compound'].astype(str) == c)
    return X


def _linear(compound_only: bool = False) -> Pipeline:
    cats = ['Compound'] if compound_only else dm.CAT_FEATURES
    nums = [] if compound_only else NUMERIC + [f'age_{c}' for c in COMPOUNDS]
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), nums),
    ])
    return Pipeline([('x', FunctionTransformer(_age_by_compound)), ('pre', pre), ('m', Ridge(alpha=10.0))])


def _ordinal_trees(model) -> Pipeline:
    pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                              dm.CAT_FEATURES)], remainder='passthrough')
    return Pipeline([('pre', pre), ('m', model)])


def tyre_candidates() -> dict:
    mono = {'TyreLife': 1}
    return {
        DEPLOYED: lambda: Est(dm.new_regressor(), categorical_feature=dm.CAT_FEATURES),
        'LightGBM, no monotone constraint': lambda: Est(
            lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31,
                              min_child_samples=20, verbosity=-1), categorical_feature=dm.CAT_FEATURES),
        'HistGradientBoosting, monotone': lambda: Est(HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=5, max_leaf_nodes=31, min_samples_leaf=20,
            categorical_features='from_dtype', monotonic_cst=mono, random_state=0)),
        'Random forest': lambda: Est(_ordinal_trees(RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=0)),
            weight_arg='m__sample_weight'),
        'Ridge (linear, age per compound)': lambda: Est(_linear(), weight_arg='m__sample_weight'),
        'Compound mean only (no tyre age)': lambda: Est(_linear(compound_only=True), weight_arg='m__sample_weight'),
    }


def load_silver() -> pd.DataFrame:
    silver = pd.read_parquet(SILVER_FILE)
    for c in dm.CAT_FEATURES:
        silver[c] = silver[c].astype('category')
    return silver[silver['race'].str[:4].astype(int) >= dm.TRAIN_FROM_SEASON].reset_index(drop=True)


def compare_tyre_models():
    silver = load_silver()
    cols = dm.FEATURE_COLS
    folds = []
    cands = tyre_candidates()
    for held_out in silver['Circuit'].unique():
        train, test = silver[silver['Circuit'] != held_out], silver[silver['Circuit'] == held_out]
        if len(test) < 30:
            continue
        w = dm.era_weights(train['race'])
        for name, make in cands.items():
            pred = make().fit(train[cols], train['RelativePace'], w).predict(test[cols])
            folds.append({'model': name, 'circuit': held_out, 'n': len(test),
                          'mae': mean_absolute_error(test['RelativePace'], pred)})
        print(f"  held out {held_out:28s} " + '  '.join(
            f"{f['mae']:.3f}" for f in folds[-len(cands):]), flush=True)
    folds = pd.DataFrame(folds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT_DIR / 'tyre_folds.csv', index=False)

    wide = folds.pivot(index='circuit', columns='model', values='mae')
    n = folds.drop_duplicates('circuit').set_index('circuit')['n'].reindex(wide.index).to_numpy()
    res = _circuit_summary(wide, n, cands)
    res.to_csv(OUT_DIR / 'tyre_summary.csv', index=False)
    print(f"\n=== Tyre models, leave-one-circuit-out over {len(wide)} circuits (MAE of relative pace, "
          f"95% CI resampling circuits) ===")
    print(res.round(3).to_string(index=False))
    return res


def _circuit_summary(wide: pd.DataFrame, n: np.ndarray, names) -> pd.DataFrame:
    """Lap-weighted MAE per model from a circuit x model table, with 95% CIs
    from resampling whole circuits and the paired difference vs. deployed."""
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(wide), (N_BOOTSTRAP, len(wide)))
    rows = []
    for name in names:
        m = wide[name].to_numpy()
        boot = [np.average(m[d], weights=n[d]) for d in draws]
        diff = [np.average((m - wide[DEPLOYED].to_numpy())[d], weights=n[d]) for d in draws]
        rows.append({'model': name, 'mae_s': np.average(m, weights=n),
                     'mae_ci_low': bootstrap_ci(boot)[0], 'mae_ci_high': bootstrap_ci(boot)[1],
                     'vs_deployed_s': np.average(m - wide[DEPLOYED].to_numpy(), weights=n),
                     'vs_deployed_ci_low': bootstrap_ci(diff)[0], 'vs_deployed_ci_high': bootstrap_ci(diff)[1],
                     'circuits_better_than_deployed': int((m < wide[DEPLOYED].to_numpy()).sum())})
    return pd.DataFrame(rows)


def compare_tyre_models_kfold(k: int = 5):
    """Grouped k-fold: whole circuits are held out together, never laps of a
    circuit split across train and test, so this asks the same question as
    leave-one-circuit-out at a fraction of the fits, with less training data
    per fold. Repeated with KFOLD_REPEATS different assignments of circuits to
    folds, because one assignment can be lucky. Per-circuit errors are averaged
    over the repeats, then summarised exactly like leave-one-circuit-out, so
    the two can be compared directly."""
    silver = load_silver()
    cols = dm.FEATURE_COLS
    cands = tyre_candidates()
    rows = []
    for repeat in range(KFOLD_REPEATS):
        splitter = GroupKFold(n_splits=k, shuffle=True, random_state=repeat)
        for fold, (tr, te) in enumerate(splitter.split(silver, groups=silver['Circuit'].astype(str))):
            train, test = silver.iloc[tr], silver.iloc[te]
            w = dm.era_weights(train['race'])
            maes = []
            for name, make in cands.items():
                pred = make().fit(train[cols], train['RelativePace'], w).predict(test[cols])
                err = pd.Series(np.abs(test['RelativePace'].to_numpy() - pred), index=test.index)
                for circuit, e in err.groupby(test['Circuit'].astype(str)):
                    rows.append({'model': name, 'repeat': repeat, 'fold': fold, 'circuit': circuit,
                                 'n': len(e), 'mae': e.mean()})
                maes.append(err.mean())
            print(f"  repeat {repeat} fold {fold} ({test['Circuit'].nunique()} circuits, {len(test)} laps) "
                  + '  '.join(f"{m:.3f}" for m in maes), flush=True)
    folds = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT_DIR / f'tyre_cv{k}_circuits.csv', index=False)

    wide = folds.groupby(['circuit', 'model'])['mae'].mean().unstack()
    n = folds.drop_duplicates(['circuit']).set_index('circuit')['n'].reindex(wide.index).to_numpy()
    res = _circuit_summary(wide, n, cands)
    by_fold = folds.groupby(['model', 'repeat', 'fold']).apply(
        lambda g: np.average(g['mae'], weights=g['n']), include_groups=False)
    spread = by_fold.groupby('model').agg(['std', 'min', 'max'])
    by_repeat = folds.groupby(['model', 'repeat']).apply(
        lambda g: np.average(g['mae'], weights=g['n']), include_groups=False).unstack()
    for col, s in {'fold_sd': spread['std'], 'fold_min': spread['min'], 'fold_max': spread['max'],
                   'repeat_range': by_repeat.max(axis=1) - by_repeat.min(axis=1)}.items():
        res[col] = res['model'].map(s)
    res.to_csv(OUT_DIR / f'tyre_cv{k}_summary.csv', index=False)
    print(f"\n=== Tyre models, grouped {k}-fold by circuit, {KFOLD_REPEATS} repeats over {len(wide)} circuits "
          f"(MAE of relative pace, 95% CI resampling circuits) ===")
    print(res.round(3).to_string(index=False))

    loco = OUT_DIR / 'tyre_summary.csv'
    if loco.exists():
        both = res.set_index('model')[['mae_s', 'vs_deployed_s']].join(
            pd.read_csv(loco).set_index('model')[['mae_s', 'vs_deployed_s']], rsuffix='_loco')
        print(f"\n{k}-fold vs. leave-one-circuit-out:")
        print(both.round(3).to_string())
        print(f"Spearman rank correlation of the models' MAE: "
              f"{both['mae_s'].corr(both['mae_s_loco'], method='spearman'):.2f}")
    return res


def fit_tyre_bundles() -> dict:
    """{name: bundle in 03's persisted format} fitted on all the data, for 06 to price strategies with."""
    silver = load_silver()
    w = dm.era_weights(silver['race'])
    bundles = {}
    for name, make in tyre_candidates().items():
        print(f"  fitting {name}", flush=True)
        bundles[name] = {'model': make().fit(silver[dm.FEATURE_COLS], silver['RelativePace'], w),
                         'feature_cols': dm.FEATURE_COLS,
                         'compound_categories': silver['Compound'].cat.categories.tolist(),
                         'circuit_categories': silver['Circuit'].cat.categories.tolist()}
    return bundles


# --- Pit classifiers ---------------------------------------------------------

FEATURES_FILE = OUT_DIR / 'classifier_features.parquet'
CLF_FEATURES = cl.FEATURE_COLS
CLF_CAT = ['own_compound']


def classifier_frame() -> pd.DataFrame:
    """05's feature table (cached: building it reads every race)."""
    if FEATURES_FILE.exists():
        return pd.read_parquet(FEATURES_FILE)
    rivals_df = pd.read_csv('data/championship_rivals.csv')
    rivals_by_race = {race: grp.set_index('driver')[['rival_driver', 'rival_team']].to_dict('index')
                      for race, grp in rivals_df.groupby('race')}
    rows = []
    for f in sorted(cl.BRONZE_DIR.glob('*_laps.parquet')):
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps = laps[laps['Compound'].isin(cl.DRY_COMPOUNDS + ['INTERMEDIATE', 'WET'])]
        if laps.empty:
            continue
        event = laps['EventName'].iloc[0] if 'EventName' in laps else race_id
        laps = cl.fuel_correct(laps)
        laps = laps.join(cl.compute_field_pace(laps), on='LapNumber')
        laps['RelativePace'] = laps['FuelCorrectedLapTime'] - laps['FieldPace']
        rows.append(cl.build_race_features(laps, race_id, event, rivals_by_race.get(race_id, {})))
    df = cl.add_circuit_sc_prob(pd.concat(rows, ignore_index=True))
    df['own_compound'] = df['own_compound'].astype('category')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_FILE, index=False)
    return df


def _clf_pipe(model, cols=None) -> Pipeline:
    cols = cols or CLF_FEATURES
    cat = [c for c in cols if c in CLF_CAT]
    num = [c for c in cols if c not in CLF_CAT]
    return Pipeline([('pre', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num),
    ])), ('m', model)])


def classifier_candidates() -> dict:
    return {
        DEPLOYED: lambda: Est(cl.new_classifier(), categorical_feature=CLF_CAT),
        'Logistic regression': lambda: Est(_clf_pipe(LogisticRegression(max_iter=500, C=1.0)),
                                           weight_arg='m__sample_weight'),
        'Logistic, tyre age + compound + progress only': lambda: Est(_clf_pipe(
            LogisticRegression(max_iter=500), ['own_tyre_age', 'own_compound', 'race_progress', 'laps_remaining']),
            weight_arg='m__sample_weight'),
        'HistGradientBoosting': lambda: Est(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, categorical_features='from_dtype',
            random_state=0)),
        'Random forest': lambda: Est(Pipeline([
            ('pre', ColumnTransformer(
                [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CLF_CAT)],
                remainder='passthrough')),
            ('m', RandomForestClassifier(n_estimators=200, min_samples_leaf=20, max_features='sqrt',
                                         n_jobs=-1, random_state=0))]), weight_arg='m__sample_weight'),
    }


def _columns(name: str) -> list:
    return ['own_tyre_age', 'own_compound', 'race_progress', 'laps_remaining'] \
        if name.startswith('Logistic, tyre age') else CLF_FEATURES


def rolling_predictions(df: pd.DataFrame, name: str, make) -> pd.DataFrame:
    """05's rolling_origin_predictions for any estimator: each season predicted
    by a model trained on earlier races only, threshold chosen on the last
    validation block of that training data."""
    cols = _columns(name)
    in_training = df['race'].str[:4].astype(int) >= cl.TRAIN_FROM_SEASON
    out = []
    for season in cl.ROLLING_SEASONS:
        test = df[(df['race'].str[:4].astype(int) == season) & df[cl.LABEL].notna()]
        train = df[(df['race'] < f"{season}_r00") & in_training & df[cl.LABEL].notna()]
        if test.empty or train.empty:
            continue
        fit_rows, val_rows = cl.split_validation(train)
        val_p = make().fit(fit_rows[cols], fit_rows[cl.LABEL].astype(int),
                           cl.era_weights(fit_rows['race'])).predict_proba(val_rows[cols])
        threshold = cl.best_f1_threshold(val_rows[cl.LABEL].values, val_p)
        proba = make().fit(train[cols], train[cl.LABEL].astype(int),
                           cl.era_weights(train['race'])).predict_proba(test[cols])
        o = cl.prediction_frame(test)
        o['pred_proba'], o['pred'], o['season'] = proba, (proba >= threshold).astype(int), season
        o['threshold'] = threshold
        out.append(o)
    return pd.concat(out, ignore_index=True)


def _slug(name: str) -> str:
    return ''.join(c if c.isalnum() else '_' for c in name.lower()).strip('_')


def _clf_pred_file(name: str) -> Path:
    return OUT_DIR / f'classifier_{_slug(name)}_rolling_predictions.csv'


def compare_classifiers():
    df = classifier_frame()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in classifier_candidates().items():
        print(f"  rolling-origin: {name}", flush=True)
        rolling_predictions(df, name, make).to_csv(_clf_pred_file(name), index=False)
    return summarise_classifiers()


def summarise_classifiers():
    """Scores the saved rolling-origin predictions, forward-test races left out."""
    preds = {name: pd.read_csv(_clf_pred_file(name)) for name in classifier_candidates()}
    preds = {name: p[p['race'] < HOLDOUT_FROM].reset_index(drop=True) for name, p in preds.items()}
    base = preds[DEPLOYED]
    races = base['race'].to_numpy()
    uniq = np.unique(races)
    idx = {r: np.flatnonzero(races == r) for r in uniq}
    y = base['label_pit'].to_numpy()
    rng = np.random.default_rng(0)
    draws = [np.concatenate([idx[r] for r in rng.choice(uniq, len(uniq))]) for _ in range(300)]
    ap = lambda p, ix: average_precision_score(y[ix], p[ix])
    rows = []
    for name, p in preds.items():
        pp = p.reset_index(drop=True)
        assert (pp['race'] == base['race']).all() and (pp['lap'] == base['lap']).all()
        proba = pp['pred_proba'].to_numpy()
        boot = [ap(proba, d) for d in draws]
        diff = [ap(proba, d) - ap(base['pred_proba'].to_numpy(), d) for d in draws]
        rows.append({'model': name, 'avg_precision': average_precision_score(y, proba),
                     'ap_ci_low': bootstrap_ci(boot)[0], 'ap_ci_high': bootstrap_ci(boot)[1],
                     'vs_deployed': average_precision_score(y, proba) - average_precision_score(
                         y, base['pred_proba'].to_numpy()),
                     'vs_deployed_ci_low': bootstrap_ci(diff)[0], 'vs_deployed_ci_high': bootstrap_ci(diff)[1],
                     'brier': brier_score_loss(y, proba),
                     'calibration_error': cl.expected_calibration_error(y, proba),
                     'f1_at_threshold': f1_score(y, pp['pred'], zero_division=0),
                     'mean_P': proba.mean(), 'pit_rate': y.mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'classifier_summary.csv', index=False)
    print(f"\n=== Pit classifiers, rolling-origin {uniq[0]} to {uniq[-1]} ({len(uniq)} races, {len(y)} laps; "
          f"95% CI resampling races) ===")
    print(res.round(4).to_string(index=False))
    per_season = pd.concat([p.assign(model=n) for n, p in preds.items()]).groupby(['model', 'season']).apply(
        lambda g: average_precision_score(g['label_pit'], g['pred_proba']), include_groups=False).unstack()
    per_season.to_csv(OUT_DIR / 'classifier_ap_by_season.csv')
    print("\nAverage precision by season:")
    print(per_season.round(3).to_string())
    return res


# --- Downstream: through the backtest ---------------------------------------

KEEP = ['Real teams', 'Classifier, threshold', 'Classifier, multi-stop ranking',
        'Default plan (pre-race)', 'Live policy (default + SC/VSC)']


def _backtest_file(kind: str, name: str) -> Path:
    return OUT_DIR / f'backtest_{kind}_{_slug(name)}.csv'


def downstream():
    bt = _load('backtest_lib', '06_backtest_evaluation.py')
    dp = pd.read_csv('data/dp_baseline_backtest.csv')
    deg_df = pd.read_csv('data/degradation_summary.csv')
    deployed_pred = pd.read_csv('data/trigger_classifier_rolling_predictions.csv')

    def run(kind, name, pred, bundle):
        # Forward-test races are dropped before replaying, not just before scoring.
        out = bt.run_backtest(pred[pred['race'] < HOLDOUT_FROM], dp, deg_df, bundle)
        out['season'] = out['race'].str[:4].astype(int)
        out.to_csv(_backtest_file(kind, name), index=False)
        print(f"  [{kind}] {name}: {out['race'].nunique()} races, {len(out)} driver-races", flush=True)

    print("=== Downstream: tyre model varied (deployed classifier's predictions) ===")
    for name, bundle in fit_tyre_bundles().items():
        run('tyre', name, deployed_pred, bundle)

    import joblib
    deployed_bundle = joblib.load('data/models/degradation_model.joblib')
    print("\n=== Downstream: pit classifier varied (deployed tyre model) ===")
    for name in classifier_candidates():
        if not _clf_pred_file(name).exists():
            print(f"  {name}: run the classifier stage first")
            continue
        run('classifier', name, pd.read_csv(_clf_pred_file(name)), deployed_bundle)
    return summarise_downstream(bt)


def summarise_downstream(bt=None):
    """Bootstraps the saved backtests (06's bootstrap_medians), forward-test races left out."""
    bt = bt or _load('backtest_lib', '06_backtest_evaluation.py')
    rows = []
    for kind, names in [('tyre', tyre_candidates()), ('classifier', classifier_candidates())]:
        for name in names:
            if not _backtest_file(kind, name).exists():
                continue
            out = pd.read_csv(_backtest_file(kind, name))
            out = out[out['race'] < HOLDOUT_FROM]
            table = bt.bootstrap_medians(out)
            table = table[table['strategy'].isin(KEEP)]
            table.insert(0, 'model', name)
            table.insert(0, 'varied', kind)
            table['races'] = out['race'].nunique()
            rows.append(table)
            print(f"\n[{kind}] {name}: {out['race'].nunique()} races, {len(out)} driver-races")
            print(table.drop(columns=['varied']).round(2).to_string(index=False), flush=True)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT_DIR / 'downstream_summary.csv', index=False)
    return res


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if '--cv' in sys.argv:
        compare_tyre_models_kfold(int(sys.argv[sys.argv.index('--cv') + 1]))
    elif stage in ('tyre', 'all'):
        compare_tyre_models()
    if stage in ('classifier', 'all'):
        compare_classifiers()
    if stage in ('downstream', 'all'):
        downstream()
    if stage == 'rescore':   # summaries again from the saved predictions and backtests, no refitting
        summarise_classifiers()
        summarise_downstream()


if __name__ == '__main__':
    main()
