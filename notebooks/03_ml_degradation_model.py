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
from pathlib import Path
from sklearn.metrics import mean_absolute_error

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

        rows.append(clean[[
            'Driver', 'LapNumber', 'RaceProgress', 'Compound', 'TyreLife', 'Stint',
            'LapTimeSeconds', 'RelativePace', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
            'EventName', 'RoundNumber',
        ]])

    silver = pd.concat(rows, ignore_index=True)
    silver.to_parquet(SILVER_DIR / 'laps_ml_features.parquet', index=False)
    return silver


def train_model(silver: pd.DataFrame):
    """
    Leave-one-circuit-out evaluation: train on 11 races, test on the 12th,
    rotating through all 12. This tests whether the model actually
    generalizes to an unseen circuit rather than just memorizing it --
    the realistic test, since in production this model has to predict
    degradation at a circuit before/during that circuit's own race.
    """
    cat_features = ['Compound', 'EventName']
    for c in cat_features:
        silver[c] = silver[c].astype('category')

    feature_cols = ['TyreLife', 'RaceProgress', 'Compound', 'AirTemp', 'TrackTemp',
                     'Humidity', 'Rainfall', 'Stint', 'EventName']

    circuits = silver['EventName'].unique()
    fold_errors = []

    for held_out in circuits:
        train = silver[silver['EventName'] != held_out]
        test = silver[silver['EventName'] == held_out]
        if len(test) < 30:
            continue

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=20, verbosity=-1,
        )
        model.fit(train[feature_cols], train['RelativePace'], categorical_feature=['Compound', 'EventName'])
        preds = model.predict(test[feature_cols])
        mae = mean_absolute_error(test['RelativePace'], preds)
        fold_errors.append((held_out, mae, len(test)))
        print(f"  held out {held_out:28s}: MAE {mae:.3f}s  (n={len(test)})")

    overall_mae = np.average([e[1] for e in fold_errors], weights=[e[2] for e in fold_errors])
    print(f"\nOverall leave-one-circuit-out MAE: {overall_mae:.3f}s")

    # Final model trained on everything, for extracting degradation curves / downstream use.
    final_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        num_leaves=31, min_child_samples=20, verbosity=-1,
    )
    final_model.fit(silver[feature_cols], silver['RelativePace'], categorical_feature=['Compound', 'EventName'])
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
    combos = silver[['EventName', 'Compound']].drop_duplicates()
    for _, combo in combos.iterrows():
        sub = silver[(silver['EventName'] == combo['EventName']) & (silver['Compound'] == combo['Compound'])]
        if len(sub) < 10:
            continue

        tyre_range = np.arange(1, min(int(sub['TyreLife'].max()) + 1, 40))
        template = sub[['RaceProgress', 'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall', 'Stint']].median()

        query = pd.DataFrame({'TyreLife': tyre_range})
        for col, val in template.items():
            query[col] = val
        query['Compound'] = pd.Categorical([combo['Compound']] * len(query), categories=silver['Compound'].cat.categories)
        query['EventName'] = pd.Categorical([combo['EventName']] * len(query), categories=silver['EventName'].cat.categories)

        preds = model.predict(query[feature_cols])
        slope = np.polyfit(tyre_range, preds, 1)[0]
        rows.append({
            'circuit': combo['EventName'], 'compound': combo['Compound'],
            'n_laps_observed': len(sub), 'implied_slope_s_per_lap': slope,
            'pred_at_tyre_5': preds[4] if len(preds) > 4 else None,
            'pred_at_tyre_20': preds[19] if len(preds) > 19 else None,
        })
    return pd.DataFrame(rows).sort_values(['circuit', 'compound'])


def main():
    print("Building Silver-layer ML feature dataset from all cached races...")
    silver = build_silver_dataset()
    print(f"  {len(silver)} clean laps across {silver['EventName'].nunique()} circuits\n")

    print("Training LightGBM with leave-one-circuit-out cross-validation:")
    model, feature_cols, fold_errors = train_model(silver)

    print("\nExtracting learned degradation curves per circuit/compound...")
    curves = extract_degradation_curves(model, feature_cols, silver)
    curves.to_csv('data/ml_degradation_curves.csv', index=False)
    print(curves.to_string(index=False))
    print(f"\nSaved data/ml_degradation_curves.csv ({len(curves)} rows)")


if __name__ == '__main__':
    main()
