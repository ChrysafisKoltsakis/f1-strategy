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


def collect_pit_stop_losses(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> list:
    """
    Pair in-laps with out-laps per driver, restricted to stops run entirely
    under green flag (SC/VSC/red-flag stops have in/out times inflated by
    the incident itself, unrelated to pit-lane cost, and badly distort the
    estimate if included). Compare each lap's actual time against
    FieldPace[lap] + degradation curve at that tyre age -- the field's
    actual pace that lap, adjusted for tyre wear, not a flat average.
    Returns one record per stop (Driver, Team, loss) rather than an
    aggregate, so callers can pool per-race (circuit constant) or across
    races (per-team crew-speed adjustment).
    """
    laps = laps.join(field_pace, on='LapNumber')
    green = laps['TrackStatus'] == '1'
    team_col = 'Team' if 'Team' in laps.columns else None
    cols = ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']
    if team_col:
        cols.append(team_col)

    in_laps = laps[laps['PitInTime'].notna() & green][cols]
    out_laps = laps[laps['PitOutTime'].notna() & green][cols]

    def expected_pace(row):
        deg = degradation.get(row['Compound'])
        if deg is None or pd.isna(row['FieldPace']) or pd.isna(row['TyreLife']):
            return None
        return row['FieldPace'] + deg['relative_intercept'] + deg['tyre_age_slope'] * row['TyreLife']

    records = []
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
        records.append({
            'driver': in_lap['Driver'], 'team': in_lap[team_col] if team_col else None, 'loss': loss,
        })
    return records


MIN_PLAUSIBLE_GREEN_PIT_LOSS = 12.0  # seconds -- pit-lane transit time alone rarely drops below this


def estimate_pit_loss(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> dict:
    """Green-flag pit loss. A physically implausible median (found via
    the DP baseline exploiting it into a 47-stop "optimal" strategy once
    a more expressive degradation model made the DP aggressive enough to
    actually chase the exploit) means the in-lap/out-lap decomposition
    got corrupted for this race -- e.g. a wet/disrupted race where the
    "expected pace" baseline itself is unreliable (Emilia Romagna 2022,
    caught this way, is a race already flagged elsewhere in this project
    for exactly that kind of disruption) -- not a genuinely fast pit
    lane. Rejected outright rather than silently trusted, same principle
    as the >=30-lap gate on degradation curves elsewhere in this file."""
    losses = [r['loss'] for r in collect_pit_stop_losses(laps, field_pace, degradation)]
    if not losses:
        return {'n_stops': 0, 'median_pit_loss': None}
    median_loss = float(np.median(losses))
    if median_loss < MIN_PLAUSIBLE_GREEN_PIT_LOSS:
        return {'n_stops': len(losses), 'median_pit_loss': None,
                'rejected_implausible_median': median_loss}
    return {'n_stops': len(losses), 'median_pit_loss': median_loss, 'std': float(np.std(losses))}


def _is_sc_or_vsc(track_status) -> bool:
    """True if full Safety Car ('4') or Virtual Safety Car ('6'/'7') is
    active per FastF1's concatenated per-lap TrackStatus codes. Red flag
    ('5') is deliberately excluded even if it co-occurs -- a red-flag stop
    is a different regime (near-zero cost, field fully stationary) that
    would corrupt an SC/VSC pit-loss estimate if pooled in."""
    return isinstance(track_status, str) and any(c in track_status for c in '467') and '5' not in track_status


def estimate_sc_pit_loss(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> dict:
    """
    Same in-lap/out-lap decomposition as estimate_pit_loss, but for stops
    taken while SC/VSC is active on the in-lap (the lap the driver commits
    to pitting) -- this is the separate, much cheaper pit-loss constant
    real teams rely on when deciding to "pit under the SC". FieldPace
    already reflects the SC/VSC-slowed pace for that lap number (it's
    computed from the whole unfiltered field, not just green laps), so the
    same actual-vs-expected decomposition isolates the pit-lane-specific
    cost on top of the SC slowdown, same as the green-flag estimate does.
    """
    laps = laps.join(field_pace, on='LapNumber')
    sc_mask = laps['TrackStatus'].apply(_is_sc_or_vsc)

    in_laps = laps[laps['PitInTime'].notna() & sc_mask][
        ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']]
    out_laps = laps[laps['PitOutTime'].notna()][
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
    all_sc_pit_loss_rows = []
    all_stop_records = []  # per-stop (team, loss, circuit baseline) for the team-adjustment pass

    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps = fuel_correct(laps)
        event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id

        field_pace = compute_field_pace(laps)
        clean = clean_laps(laps, field_pace)
        degradation = fit_degradation(clean)
        stop_records = collect_pit_stop_losses(laps, field_pace, degradation)
        pit_loss = estimate_pit_loss(laps, field_pace, degradation)
        sc_pit_loss = estimate_sc_pit_loss(laps, field_pace, degradation)

        if pit_loss['median_pit_loss'] is not None:
            for r in stop_records:
                all_stop_records.append({**r, 'race': race_id, 'circuit_pit_loss': pit_loss['median_pit_loss']})

        print(f"=== {race_id}: {event_name} ===")
        print(f"  {len(laps)} total laps -> {len(clean)} clean laps after filtering "
              f"({100*len(clean)/len(laps):.0f}%)")
        for compound, d in degradation.items():
            stint_slope_str = (f"{d['median_stint_slope']:+.3f}" if d['median_stint_slope'] is not None
                                else "n/a")
            print(f"  {compound:8s}: pooled slope {d['tyre_age_slope']:+.3f} s/lap, "
                  f"median per-stint slope {stint_slope_str} s/lap "
                  f"(n={d['n_laps']} laps, {d['n_stints']} stints)")
            all_degradation_rows.append({
                'race': race_id, 'event': event_name, 'compound': compound, **d
            })
        if pit_loss['median_pit_loss'] is not None:
            print(f"  Pit loss (green): {pit_loss['median_pit_loss']:.2f}s "
                  f"(std {pit_loss['std']:.2f}, n={pit_loss['n_stops']} stops)")
            all_pit_loss_rows.append({'race': race_id, 'event': event_name, **pit_loss})
        elif 'rejected_implausible_median' in pit_loss:
            print(f"  Pit loss (green): REJECTED, implausible median "
                  f"{pit_loss['rejected_implausible_median']:.2f}s from n={pit_loss['n_stops']} stops "
                  f"(likely a disrupted race corrupting the expected-pace baseline)")
        else:
            print("  Pit loss (green): no valid stops")
        if sc_pit_loss['median_pit_loss'] is not None:
            print(f"  Pit loss (SC/VSC): {sc_pit_loss['median_pit_loss']:.2f}s "
                  f"(std {sc_pit_loss['std']:.2f}, n={sc_pit_loss['n_stops']} stops)")
            all_sc_pit_loss_rows.append({'race': race_id, 'event': event_name, **sc_pit_loss})
        else:
            print("  Pit loss (SC/VSC): no valid stops")
        print()

    deg_df = pd.DataFrame(all_degradation_rows)
    pit_df = pd.DataFrame(all_pit_loss_rows)
    sc_pit_df = pd.DataFrame(all_sc_pit_loss_rows)
    deg_df.to_csv('data/degradation_summary.csv', index=False)
    pit_df.to_csv('data/pit_loss_summary.csv', index=False)
    sc_pit_df.to_csv('data/sc_pit_loss_summary.csv', index=False)
    print(f"Saved data/degradation_summary.csv ({len(deg_df)} rows)")
    print(f"Saved data/pit_loss_summary.csv ({len(pit_df)} rows)")
    print(f"Saved data/sc_pit_loss_summary.csv ({len(sc_pit_df)} rows)")

    # Team-specific pit-loss adjustment: a team's per-race stop count is far
    # too small (2 cars, 1-2 stops each) to fit a stable per-(circuit, team)
    # constant directly. Pit-crew execution speed is also largely
    # circuit-independent (only the pit-lane transit time is circuit-
    # specific), so instead pool each stop's deviation from that race's own
    # circuit-wide pit-loss constant across ALL of a team's stops, at every
    # circuit -- giving a much bigger, more stable sample per team. Applied
    # downstream as: team_pit_loss(circuit, team) = circuit_pit_loss(circuit)
    # + team_adjustment(team).
    MIN_TEAM_STOPS = 15
    stops_df = pd.DataFrame(all_stop_records)
    stops_df['deviation'] = stops_df['loss'] - stops_df['circuit_pit_loss']
    team_adj = stops_df.groupby('team')['deviation'].agg(['median', 'count']).reset_index()
    team_adj.columns = ['team', 'adjustment_s', 'n_stops']
    team_adj = team_adj[team_adj['n_stops'] >= MIN_TEAM_STOPS].sort_values('adjustment_s')
    team_adj.to_csv('data/team_pit_loss_adjustment.csv', index=False)
    print(f"\nTeam pit-loss adjustment (negative = faster than field average):")
    print(team_adj.to_string(index=False))
    print(f"Saved data/team_pit_loss_adjustment.csv ({len(team_adj)} teams, "
          f"min {MIN_TEAM_STOPS} stops to be included)")


if __name__ == '__main__':
    main()
