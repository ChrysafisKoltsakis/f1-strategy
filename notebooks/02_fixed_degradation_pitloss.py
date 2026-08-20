"""
Fixed degradation-curve and pit-loss estimation, addressing the issues
found in 00_explore_one_race.py on 2023 Spain and in the first pass at
this script across the half-season:

  1. SOFT compound showed negative degradation -- likely track evolution
     (whole field getting faster early in the race) confounded with tyre
     wear. Fix: detrend each lap against the field's median pace on that
     lap number FIRST (RelativePace = FuelCorrectedLapTime - FieldPace),
     then fit tyre-age effect on the residual. This avoids the collinearity
     a first attempt at LapTime ~ TyreAge + FieldPace ran into (a driver's
     own lap time is part of the field median that same lap, so the two
     regressors are nearly collinear -- the fit "worked" but produced an
     intercept/coefficient pair that were only valid together, and pit-loss
     prediction silently dropped the FieldPace term, producing nonsense
     80-200s "pit losses").

  2. Outlier laps (85-89s vs ~78s baseline) survived the green-flag
     filter -- these were lap 1 (standing start, not pit-related) and
     laps immediately after a SC/VSC restart (pack bunching under a
     nominally-green status). Fix: explicitly drop LapNumber == 1 and
     the 2 laps following any non-green TrackStatus lap, plus an IQR
     trim per compound as a statistical backstop. Also discovered
     TrackStatus is not single-character -- FastF1 concatenates flags
     active during a lap (e.g. "124") -- so exact-match '1' is required,
     not a substring check.

  3. Pit-loss was estimated by comparing the pit lap's raw time to
     green pace, which conflated several effects, and a first fix still
     produced 50-200s "pit losses" because of the collinearity bug above.
     Now: pair each driver's in-lap (PitInTime set) with their out-lap
     (PitOutTime set, following lap), restrict to stops where both laps
     were run under a pure green flag, and compare against
         FieldPace[lap] + (intercept + slope * TyreAge)
     i.e. the detrended degradation curve added back onto that lap's
     actual field pace -- not a global average.

Run across all cached Bronze races: python notebooks/02_fixed_degradation_pitloss.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path('data/bronze')
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
VALID_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['TrackStatus'] = laps['TrackStatus'].astype(str)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    """Median fuel-corrected lap time across the whole field, per lap number."""
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def clean_laps(laps: pd.DataFrame, field_pace: pd.Series) -> pd.DataFrame:
    """Green-flag, non-pit, non-restart-affected, IQR-trimmed laps, detrended vs field pace."""
    laps = laps.join(field_pace, on='LapNumber')
    laps['RelativePace'] = laps['FuelCorrectedLapTime'] - laps['FieldPace']

    non_green_laps = set(laps.loc[laps['TrackStatus'] != '1', 'LapNumber'].unique())
    restart_affected = set()
    for ln in non_green_laps:
        restart_affected.update([ln + 1, ln + 2])

    clean = laps[
        laps['PitInTime'].isna() &
        laps['PitOutTime'].isna() &
        laps['RelativePace'].notna() &
        laps['TyreLife'].notna() &
        laps['Compound'].isin(VALID_COMPOUNDS) &
        (laps['LapNumber'] != 1) &
        (~laps['LapNumber'].isin(restart_affected)) &
        (laps['TrackStatus'] == '1')
    ].copy()

    trimmed = []
    for compound, sub in clean.groupby('Compound'):
        q1, q3 = sub['RelativePace'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed.append(sub[sub['RelativePace'].between(lo, hi)])
    return pd.concat(trimmed) if trimmed else clean.iloc[0:0]


def fit_degradation(clean: pd.DataFrame) -> dict:
    """
    RelativePace ~ TyreAge, i.e. "seconds off the field's median pace on that
    lap, as a function of tyre age" -- with track evolution already removed by
    the field-pace detrend, so this is a single-variable fit with no
    collinearity risk. Also reports the median of independent per-stint
    slopes as a check the pooled number isn't an artifact of a few stints.
    """
    results = {}
    for compound, sub in clean.groupby('Compound'):
        if len(sub) < 15 or sub['TyreLife'].nunique() < 2:
            continue

        slope, intercept = np.polyfit(sub['TyreLife'], sub['RelativePace'], 1)

        stint_slopes = []
        for _, stint_sub in sub.groupby(['Driver', 'Stint']):
            if len(stint_sub) < 5 or stint_sub['TyreLife'].nunique() < 2:
                continue
            s_slope, _ = np.polyfit(stint_sub['TyreLife'], stint_sub['RelativePace'], 1)
            stint_slopes.append(s_slope)

        results[compound] = {
            'n_laps': len(sub),
            'tyre_age_slope': slope,
            'relative_intercept': intercept,
            'n_stints': len(stint_slopes),
            'median_stint_slope': np.median(stint_slopes) if stint_slopes else None,
        }
    return results


def estimate_pit_loss(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> dict:
    """
    Pair in-laps with out-laps per driver, restricted to stops run entirely
    under green flag (SC/VSC/red-flag stops have in/out times inflated by
    the incident itself, unrelated to pit-lane cost, and badly distort the
    estimate if included). Compare each lap's actual time against
    FieldPace[lap] + degradation curve at that tyre age -- the field's
    actual pace that lap, adjusted for tyre wear, not a flat average.
    """
    laps = laps.join(field_pace, on='LapNumber')
    green = laps['TrackStatus'] == '1'

    in_laps = laps[laps['PitInTime'].notna() & green][
        ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']]
    out_laps = laps[laps['PitOutTime'].notna() & green][
        ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']]

    def expected_pace(row):
        deg = degradation.get(row['Compound'])
        if deg is None or pd.isna(row['FieldPace']) or pd.isna(row['TyreLife']):
            return None
        return row['FieldPace'] + deg['relative_intercept'] + deg['tyre_age_slope'] * row['TyreLife']

    losses = []
    for _, in_lap in in_laps.iterrows():
        match = out_laps[
            (out_laps['Driver'] == in_lap['Driver']) &
            (out_laps['LapNumber'] == in_lap['LapNumber'] + 1)
        ]
        if match.empty or pd.isna(in_lap['FuelCorrectedLapTime']):
            continue
        out_lap = match.iloc[0]
        if pd.isna(out_lap['FuelCorrectedLapTime']):
            continue

        expected_in = expected_pace(in_lap)
        expected_out = expected_pace(out_lap)
        if expected_in is None or expected_out is None:
            continue

        loss = (in_lap['FuelCorrectedLapTime'] - expected_in) + (out_lap['FuelCorrectedLapTime'] - expected_out)
        losses.append(loss)

    if not losses:
        return {'n_stops': 0, 'median_pit_loss': None}
    return {'n_stops': len(losses), 'median_pit_loss': float(np.median(losses)), 'std': float(np.std(losses))}


def main():
    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    print(f"Processing {len(lap_files)} races\n")

    all_degradation_rows = []
    all_pit_loss_rows = []

    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps = fuel_correct(laps)
        event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id

        field_pace = compute_field_pace(laps)
        clean = clean_laps(laps, field_pace)
        degradation = fit_degradation(clean)
        pit_loss = estimate_pit_loss(laps, field_pace, degradation)

        print(f"=== {race_id}: {event_name} ===")
        print(f"  {len(laps)} total laps -> {len(clean)} clean laps after filtering "
              f"({100*len(clean)/len(laps):.0f}%)")
        for compound, d in degradation.items():
            print(f"  {compound:8s}: pooled slope {d['tyre_age_slope']:+.3f} s/lap, "
                  f"median per-stint slope {d['median_stint_slope']:+.3f} s/lap "
                  f"(n={d['n_laps']} laps, {d['n_stints']} stints)")
            all_degradation_rows.append({
                'race': race_id, 'event': event_name, 'compound': compound, **d
            })
        if pit_loss['median_pit_loss'] is not None:
            print(f"  Pit loss: {pit_loss['median_pit_loss']:.2f}s "
                  f"(std {pit_loss['std']:.2f}, n={pit_loss['n_stops']} stops)")
            all_pit_loss_rows.append({'race': race_id, 'event': event_name, **pit_loss})
        else:
            print("  Pit loss: no valid green-flag stops")
        print()

    deg_df = pd.DataFrame(all_degradation_rows)
    pit_df = pd.DataFrame(all_pit_loss_rows)
    deg_df.to_csv('data/degradation_summary.csv', index=False)
    pit_df.to_csv('data/pit_loss_summary.csv', index=False)
    print(f"Saved data/degradation_summary.csv ({len(deg_df)} rows)")
    print(f"Saved data/pit_loss_summary.csv ({len(pit_df)} rows)")


if __name__ == '__main__':
    main()
