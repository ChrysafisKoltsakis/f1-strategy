"""
ML-based tyre degradation model, replacing the hand-fit linear regression
in 02_fixed_degradation_pitloss.py.

Rationale: a per-compound linear slope fit separately per circuit needs
hand-designed pooling logic (credibility weighting between this circuit's
history and this season's other circuits) to be usable on low-sample
circuits. Instead, train one gradient-boosted model on the pooled Silver
dataset across all circuits/compounds/drivers at once, with circuit,
compound, and weather as features -- the model generalizes across circuits
via shared feature structure instead of a hand-written blending formula,
and low-sample circuit/compound combos borrow strength from the rest of
the training data automatically (that's what the trees splitting on
Circuit/Compound do, rather than something we need to design by hand).

Cleaning (pit laps, standing-start lap 1, SC/VSC-restart-affected laps,
non-green flag) is kept identical to 02_fixed_degradation_pitloss.py --
that's data-quality filtering, independent of which model reads the
result afterward.

Run: python notebooks/03_ml_degradation_model.py
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import importlib.util
from pathlib import Path
from sklearn.metrics import mean_absolute_error

_spec = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpl)

BRONZE_DIR = Path('data/bronze')
SILVER_DIR = Path('data/silver')
SILVER_DIR.mkdir(parents=True, exist_ok=True)
VALID_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}


def build_silver_dataset() -> pd.DataFrame:
    """
    Target is RelativePace = LapTimeSeconds - FieldPace[lap], not raw
    LapTimeSeconds. A first pass predicting raw lap time got a
    leave-one-circuit-out MAE of ~11.6s (worse for some circuits) because
    absolute pace is dominated by each circuit's baseline lap time --
    something the model has zero ability to infer for a circuit it's never
    seen, since the categorical circuit feature carries no signal on an
    unseen category. Detrending against the field's own median pace that
    lap removes the baseline entirely, leaving "how much worse than the
    field right now" as the target -- a quantity that should behave far
    more similarly across circuits.
    """
    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    rows = []

    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps['TrackStatus'] = laps['TrackStatus'].astype(str)
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        total_laps = laps['LapNumber'].max()
        laps['RaceProgress'] = laps['LapNumber'] / total_laps
        field_pace = laps.groupby('LapNumber')['LapTimeSeconds'].median().rename('FieldPace')
        laps = laps.join(field_pace, on='LapNumber')
        laps['RelativePace'] = laps['LapTimeSeconds'] - laps['FieldPace']

        non_green_laps = set(laps.loc[laps['TrackStatus'] != '1', 'LapNumber'].unique())
        restart_affected = set()
        for ln in non_green_laps:
            restart_affected.update([ln + 1, ln + 2])

        weather_path = BRONZE_DIR / f"{race_id}_weather.parquet"
        weather = pd.read_parquet(weather_path).sort_values('Time')

        clean = laps[
            laps['PitInTime'].isna() &
            laps['PitOutTime'].isna() &
            laps['LapTimeSeconds'].notna() &
            laps['TyreLife'].notna() &
            laps['Compound'].isin(VALID_COMPOUNDS) &
            (laps['LapNumber'] != 1) &
            (~laps['LapNumber'].isin(restart_affected)) &
            (laps['TrackStatus'] == '1')
        ].copy()

        if clean.empty:
            continue

        # IQR trim per compound within this race, same as the linear-model script.
        trimmed = []
        for compound, sub in clean.groupby('Compound'):
            q1, q3 = sub['LapTimeSeconds'].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            trimmed.append(sub[sub['LapTimeSeconds'].between(lo, hi)])
        clean = pd.concat(trimmed) if trimmed else clean.iloc[0:0]
        if clean.empty:
            continue

        clean = clean.sort_values('Time')
        clean = pd.merge_asof(clean, weather[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']],
                               on='Time', direction='nearest')

        clean['race'] = race_id  # lets 12_as_of_context.py train on earlier races only
        clean['Circuit'] = dpl.circuit_of(race_id)   # not EventName: see 02's circuit_of
        for flag, v in dpl.era_flags(race_id).items():
            clean[flag] = v
        rows.append(clean[[
            'race', 'Driver', 'LapNumber', 'RaceProgress', 'Compound', 'TyreLife', 'Stint',
            'LapTimeSeconds', 'RelativePace', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
            'EventName', 'Circuit', 'RoundNumber', *dpl.era_flags(race_id),
        ]])

    silver = pd.concat(rows, ignore_index=True)
    silver.to_parquet(SILVER_DIR / 'laps_ml_features.parquet', index=False)
    return silver


FEATURE_COLS = ['TyreLife', 'RaceProgress', 'Compound', 'AirTemp', 'TrackTemp',
                'Humidity', 'Rainfall', 'Stint', 'Circuit',
                'Regs2022']  # regulation-era flag, see 02's era_flags
# No Regs2026: predicting 2026 rounds 5-16 from earlier races, the model
# without it was slightly better (MAE 1.043s vs 1.063s with it) -- pace
# relative to the field carries across a rules change that every car
# makes at once. 2026 still counts as its own era for usable tyre life.
CAT_FEATURES = ['Compound', 'Circuit']
# Seasons the model trains on (the Silver dataset keeps every season).
TRAIN_FROM_SEASON = 2019


# Older regulation eras are down-weighted as a fading prior: 02's
# era_weights / PRIOR_ERA_RACES, shared with the trigger classifier.
era_weights = dpl.era_weights


def new_regressor() -> lgb.LGBMRegressor:
    """Shared with 12_as_of_context.py so its as-of models match this one.
    Pace can only get worse with tyre age (monotone constraint): the raw
    data say otherwise late in long stints, but only because of
    survivorship -- see shape_wear()."""
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=20, verbosity=-1,
        monotone_constraints=[1 if f == 'TyreLife' else 0 for f in FEATURE_COLS],
        monotone_constraints_method='advanced',
    )


# --- Censored tyre data -------------------------------------------------
# Teams change tyres before they fall away, so race data never show the
# drop-off: within stints, observed wear even SLOWS with age (long stints
# exist only where tyres were lasting, and drivers manage pace on them), and
# the model extrapolates gentle wear into stint lengths nobody has run -- it
# picked a one-stop for every driver in Spain 2024, where every team
# two-stopped. Two corrections were tried and rejected by the teammate
# validation (13), which prices real strategies against real outcomes:
#   - forcing non-decreasing wear within, and a drop-off beyond, the 90th
#     percentile of stint lengths: slope 0.93 -> 0.18;
#   - a drop-off beyond the longest stint previously run at that circuit:
#     slope 0.47 (real stints often set new records and were fine).
# So race data can confirm the model where teams have run a strategy but
# can't prove an untried one would fail; that belongs to the pre-race
# default strategy (Pirelli / history). What stays here is (1) the monotone
# constraint in new_regressor() -- pace can't improve with tyre age; slope
# 0.91 -- and (2) a safety net against absurd extrapolation: a drop-off
# beyond the longest stint ever run on that compound at any circuit (slope
# 0.87, within noise of no drop-off).
TYRE_LIFE_MIN_STINTS = 5     # fewer stints of a compound in this era -> use every era
CLIFF_S_PER_LAP2 = 0.05      # drop-off beyond usable life: +0.05 * (laps past it)^2 s/lap
_STINT_LENGTHS = None


def stint_lengths() -> pd.DataFrame:
    """Longest tyre age reached in every dry stint (race, circuit, compound)."""
    global _STINT_LENGTHS
    if _STINT_LENGTHS is None:
        s = pd.read_parquet(SILVER_DIR / 'laps_ml_features.parquet',
                            columns=['race', 'Driver', 'Stint', 'Compound', 'TyreLife', 'Circuit'])
        s = s[s['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
        _STINT_LENGTHS = (s.groupby(['race', 'Circuit', 'Driver', 'Stint', 'Compound'], observed=True)['TyreLife']
                          .max().reset_index())
    return _STINT_LENGTHS


def usable_tyre_life(circuit: str, compounds: list, before_race: str) -> dict:
    """{compound: usable life in laps}: the longest stint run on that
    compound at any circuit in races strictly before before_race (same
    rules era if it has enough stints). circuit is kept so a per-circuit
    limit -- e.g. Pirelli's published maximum stint length -- can override."""
    st = stint_lengths()
    st = st[st['race'] < before_race]
    era = st[st['race'].map(dpl.era_of) == dpl.era_of(before_race)]
    life = {}
    for c in compounds:
        pool = era[era['Compound'] == c]['TyreLife']
        if len(pool) < TYRE_LIFE_MIN_STINTS:
            pool = st[st['Compound'] == c]['TyreLife']   # thin history (early 2022): any era
        life[c] = int(pool.max()) if len(pool) else None
    return life


def shape_wear(pace_by_age: np.ndarray, life: int | None) -> np.ndarray:
    """pace_by_age[i] = predicted relative pace at tyre age i+1. Unchanged
    up to the usable life; beyond it, the wear rate over the last 5 laps of
    usable life continues, plus a quadratic drop-off."""
    p = np.asarray(pace_by_age, dtype=float)
    if life is None or life < 6 or life >= len(p):
        return p
    life = int(life)
    rate = max((p[life - 1] - p[life - 6]) / 5.0, 0.0)
    beyond = np.arange(1, len(p) - life + 1)
    out = p.copy()
    out[life:] = p[life - 1] + rate * beyond + CLIFF_S_PER_LAP2 * beyond ** 2
    return out


def train_model(silver: pd.DataFrame):
    """
    Leave-one-circuit-out evaluation: train on 11 races, test on the 12th,
    rotating through all 12. This tests whether the model actually
    generalizes to an unseen circuit rather than just memorizing it --
    the realistic test, since in production this model has to predict
    degradation at a circuit before/during that circuit's own race.
    """
    for c in CAT_FEATURES:
        silver[c] = silver[c].astype('category')  # in place: main() saves these categories with the model
    silver = silver[silver['race'].str[:4].astype(int) >= TRAIN_FROM_SEASON]

    feature_cols = FEATURE_COLS

    circuits = silver['Circuit'].unique()
    fold_errors = []

    for held_out in circuits:
        train = silver[silver['Circuit'] != held_out]
        test = silver[silver['Circuit'] == held_out]
        if len(test) < 30:
            continue

        model = new_regressor()
        model.fit(train[feature_cols], train['RelativePace'], sample_weight=era_weights(train['race']),
                  categorical_feature=CAT_FEATURES)
        preds = model.predict(test[feature_cols])
        mae = mean_absolute_error(test['RelativePace'], preds)
        fold_errors.append((held_out, mae, len(test)))
        print(f"  held out {held_out:28s}: MAE {mae:.3f}s  (n={len(test)})")

    overall_mae = np.average([e[1] for e in fold_errors], weights=[e[2] for e in fold_errors])
    print(f"\nOverall leave-one-circuit-out MAE: {overall_mae:.3f}s")

    # Final model trained on everything, for extracting degradation curves / downstream use.
    final_model = new_regressor()
    final_model.fit(silver[feature_cols], silver['RelativePace'], sample_weight=era_weights(silver['race']),
                    categorical_feature=CAT_FEATURES)
    return final_model, feature_cols, fold_errors


def extract_degradation_curves(model, feature_cols, silver: pd.DataFrame):
    """
    Partial-dependence-style extraction: for each circuit/compound combo
    actually seen in training, predict lap time across a range of tyre
    ages, holding other features at that combo's median, to get an
    interpretable "degradation curve" out of the trained model -- the
    same kind of output the linear-regression version produced, but
    learned rather than hand-fit per circuit.
    """
    rows = []
    combos = silver[['Circuit', 'Compound']].drop_duplicates()
    for _, combo in combos.iterrows():
        sub = silver[(silver['Circuit'] == combo['Circuit']) & (silver['Compound'] == combo['Compound'])]
        if len(sub) < 10:
            continue

        tyre_range = np.arange(1, min(int(sub['TyreLife'].max()) + 1, 40))
        template = sub[['RaceProgress', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall', 'Stint']].median()

        query = pd.DataFrame({'TyreLife': tyre_range})
        for col, val in template.items():
            query[col] = val
        query['Regs2022'], query['Regs2026'] = 1, 0  # curves describe the 2022-2025 rules (most data)
        query['Compound'] = pd.Categorical([combo['Compound']] * len(query), categories=silver['Compound'].cat.categories)
        query['Circuit'] = pd.Categorical([combo['Circuit']] * len(query), categories=silver['Circuit'].cat.categories)

        preds = model.predict(query[feature_cols])
        slope = np.polyfit(tyre_range, preds, 1)[0]
        rows.append({
            'circuit': combo['Circuit'], 'compound': combo['Compound'],
            'n_laps_observed': len(sub), 'implied_slope_s_per_lap': slope,
            'pred_at_tyre_5': preds[4] if len(preds) > 4 else None,
            'pred_at_tyre_20': preds[19] if len(preds) > 19 else None,
        })
    return pd.DataFrame(rows).sort_values(['circuit', 'compound'])


MAX_AGE_GRID = 60  # laps -- covers any plausible single-stint length


def load_degradation_model():
    """Loads the persisted model bundle -- shared by 04_dp_baseline.py,
    06_backtest_evaluation.py, and 10_live_risk_comparison.py, which
    import this function dynamically (notebook filenames start with a
    digit and can't be `import`ed as normal Python modules -- same
    pattern 09_streaming_consumer.py already uses to reuse
    05_trigger_classifier.py's feature-building code)."""
    return joblib.load('data/models/degradation_model.joblib')


def build_ml_pace_lookup(race_id: str, bundle: dict, compounds: list) -> dict:
    """Precomputes predicted RelativePace for every (compound, lap, tyre
    age) combination that could plausibly occur in this race, in ONE
    batched model.predict() call, returning a
    {compound: {(lap, age): relative_pace}} lookup. A dynamic-programming
    search (or a backtest replaying many candidate strategies) needs this
    at thousands of individual states -- querying the model call-by-call
    would be far slower than one batch prediction over the full grid.

    Uses the race's own real historical weather (matching what the model
    was trained on) rather than a fixed/assumed value -- legitimate for
    replaying a historical race, same principle already used for
    FieldPace elsewhere in this codebase. Stint number, which the model
    was also trained on, is not knowable in advance for a hypothetical
    strategy still being evaluated, so it's held at each race's own
    observed median stint (the same partial-dependence-style
    simplification extract_degradation_curves() already uses for
    circuit-level curves, just applied per-race here instead)."""
    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    n_laps = int(laps['LapNumber'].max())
    circuit = dpl.circuit_of(race_id)
    median_stint = laps['Stint'].median() if laps['Stint'].notna().any() else 1.0

    weather = pd.read_parquet(BRONZE_DIR / f"{race_id}_weather.parquet").sort_values('Time')
    lap_times = laps.sort_values('Time')[['Time', 'LapNumber']].drop_duplicates('LapNumber')
    lap_weather = pd.merge_asof(lap_times, weather[['Time', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']],
                                 on='Time', direction='nearest').set_index('LapNumber')

    if circuit not in bundle['circuit_categories']:
        return {}  # circuit never seen in training -- caller should fall back
    return predict_pace_grid(bundle, circuit, n_laps, lap_weather, median_stint, compounds, dpl.era_flags(race_id),
                             usable_tyre_life(circuit, compounds, before_race=race_id))


def predict_pace_grid(bundle: dict, circuit: str, n_laps: int, lap_weather: pd.DataFrame,
                      median_stint: float, compounds: list, regs: dict, life: dict | None = None) -> dict:
    """The batched (compound, lap, tyre age) prediction behind
    build_ml_pace_lookup(), given per-lap weather (indexed by lap number).
    A circuit the model never saw is predicted with a missing circuit
    category, i.e. from the other features alone -- 12_as_of_context.py
    needs that for a circuit's first visit."""
    model, feature_cols = bundle['model'], bundle['feature_cols']
    compound_categories, circuit_categories = bundle['compound_categories'], bundle['circuit_categories']
    valid_compounds = [c for c in compounds if c in compound_categories]

    grid_rows = []
    for compound in valid_compounds:
        for lap in range(1, n_laps + 1):
            if lap not in lap_weather.index:
                continue
            w = lap_weather.loc[lap]
            for age in range(1, MAX_AGE_GRID + 1):
                grid_rows.append({
                    'lap': lap, 'compound': compound, 'age': age,
                    'TyreLife': age, 'RaceProgress': lap / n_laps,
                    'AirTemp': w['AirTemp'], 'TrackTemp': w['TrackTemp'],
                    'Humidity': w['Humidity'], 'Rainfall': w['Rainfall'],
                    'Stint': median_stint, **regs,
                })
    if not grid_rows:
        return {}

    grid = pd.DataFrame(grid_rows)
    grid['Compound'] = pd.Categorical(grid['compound'], categories=compound_categories)
    grid['Circuit'] = pd.Categorical([circuit] * len(grid), categories=circuit_categories)
    preds = model.predict(grid[feature_cols])

    grid['pred'] = preds
    lookup = {c: {} for c in valid_compounds}
    for (compound, lap), g in grid.groupby(['compound', 'lap']):
        shaped = shape_wear(g.sort_values('age')['pred'].values, (life or {}).get(compound))
        for age, v in zip(sorted(g['age']), shaped):
            lookup[compound][(lap, age)] = float(v)
    return lookup


def main():
    print("Building Silver-layer ML feature dataset from all cached races...")
    silver = build_silver_dataset()
    print(f"  {len(silver)} clean laps across {silver['Circuit'].nunique()} circuits\n")

    print("Training LightGBM with leave-one-circuit-out cross-validation:")
    model, feature_cols, fold_errors = train_model(silver)

    print("\nExtracting learned degradation curves per circuit/compound...")
    curves = extract_degradation_curves(model, feature_cols, silver)
    curves.to_csv('data/ml_degradation_curves.csv', index=False)
    print(curves.to_string(index=False))
    print(f"\nSaved data/ml_degradation_curves.csv ({len(curves)} rows)")

    # Persist the trained model itself, not just the extracted summary
    # curves -- 04_dp_baseline.py, 06_backtest_evaluation.py, and
    # 10_live_risk_comparison.py need to query it directly (tyre_age x
    # compound x lap, for an arbitrary hypothetical strategy), which the
    # summary CSV's fixed per-circuit slope/two-point snapshot can't
    # support.
    Path('data/models').mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'model': model, 'feature_cols': feature_cols,
        'compound_categories': silver['Compound'].cat.categories.tolist(),
        'circuit_categories': silver['Circuit'].cat.categories.tolist(),
    }, 'data/models/degradation_model.joblib')
    print("Saved data/models/degradation_model.joblib")


if __name__ == '__main__':
    main()
