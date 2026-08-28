"""
Phase 3 -- DP hindsight-optimal baseline. Deliberately kept small: its only
job is producing the "hindsight-optimal" column for the eventual actual
vs. classifier vs. hindsight-optimal backtest. Deterministic only -- no
safety-car/rain stochastic extension, that scope is explicitly out per the
project's scope guardrail.

Formulation follows Carrasco Heine et al. 2022, "On the optimization of pit
stop strategies via dynamic programming": state = (current compound, tyre
age, set of compounds used so far), transition each lap = stay out vs. pit
into another compound, objective = minimize total race time subject to
using at least two distinct compounds (the mandatory two-compound rule).

Reuses the Gold-layer degradation curves and green pit-loss constants
already computed by 02_fixed_degradation_pitloss.py (data/degradation_summary.csv,
data/pit_loss_summary.csv), plus per-lap FieldPace recomputed from Bronze
laps -- the same lap-time model the pit-loss/degradation pipeline uses.
Each driver's *actual* strategy is replayed through this same model (not
their real recorded lap times), so the comparison isolates "was the
strategy choice good" from "was the driving good" -- actual-strategy cost
and DP-optimal cost are both estimates under the identical model.

Team-aware pit loss: the DP is solved separately per team present in each
race, using data/team_pit_loss_adjustment.csv (circuit pit-loss + that
team's pooled deviation from field average -- see
02_fixed_degradation_pitloss.py for how it's derived). A faster pit crew
genuinely changes the optimal strategy, not just the cost of executing a
fixed one, so "hindsight-optimal" is now "optimal for that team's actual
pit-crew speed", not a single team-agnostic number per race.

Run with: python notebooks/04_dp_baseline.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

BRONZE_DIR = Path('data/bronze')
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
INF = float('inf')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def solve_dp(n_laps: int, field_pace: pd.Series, degradation: dict, pit_loss: float) -> dict:
    """Returns {'cost': float, 'stops': [(lap, compound), ...]} for the
    hindsight-optimal strategy, or None if fewer than 2 usable compounds."""
    compounds = [c for c in DRY_COMPOUNDS if c in degradation]
    if len(compounds) < 2:
        return None

    def expected_pace(lap, compound, age):
        d = degradation[compound]
        fp = field_pace.get(lap)
        if fp is None or pd.isna(fp):
            return None
        return fp + d['relative_intercept'] + d['tyre_age_slope'] * age

    def bit(c):
        return 1 << compounds.index(c)

    # state key: (compound, age) -> {mask: (cost, path)}
    states = {}
    for c in compounds:
        ep = expected_pace(1, c, 1)
        if ep is None:
            continue
        states[(c, 1, bit(c))] = (ep, [])

    for lap in range(2, n_laps + 1):
        new_states = {}
        for (c, age, mask), (cost, path) in states.items():
            ep = expected_pace(lap, c, age + 1)
            if ep is not None:
                key = (c, age + 1, mask)
                new_cost = cost + ep
                if new_cost < new_states.get(key, (INF, None))[0]:
                    new_states[key] = (new_cost, path)
            for c2 in compounds:
                if c2 == c:
                    continue
                ep2 = expected_pace(lap, c2, 1)
                if ep2 is None:
                    continue
                key2 = (c2, 1, mask | bit(c2))
                new_cost2 = cost + pit_loss + ep2
                if new_cost2 < new_states.get(key2, (INF, None))[0]:
                    new_states[key2] = (new_cost2, path + [(lap, c2)])
        states = new_states

    valid = [(cost, path) for (c, age, mask), (cost, path) in states.items()
             if bin(mask).count('1') >= 2]
    if not valid:
        return None
    best_cost, best_path = min(valid, key=lambda x: x[0])
    return {'cost': best_cost, 'stops': best_path}


def replay_actual(driver_laps: pd.DataFrame, field_pace: pd.Series, degradation: dict, pit_loss: float):
    """Cost of a driver's real stint sequence, evaluated through the same
    expected-pace model the DP uses -- not their real lap times."""
    total = 0.0
    n_pits = 0
    compounds_used = set()
    for _, row in driver_laps.sort_values('LapNumber').iterrows():
        compound, age, lap = row['Compound'], row['TyreLife'], row['LapNumber']
        if compound not in degradation or pd.isna(age):
            return None
        fp = field_pace.get(lap)
        if fp is None or pd.isna(fp):
            return None
        total += fp + degradation[compound]['relative_intercept'] + degradation[compound]['tyre_age_slope'] * age
        compounds_used.add(compound)
        if row['LapNumber'] > 1 and age == 1:
            n_pits += 1
    if len(compounds_used) < 2:
        return None
    total += n_pits * pit_loss
    return {'cost': total, 'n_pits': n_pits, 'compounds': compounds_used}


def main():
    deg_df = pd.read_csv('data/degradation_summary.csv')
    pit_df = pd.read_csv('data/pit_loss_summary.csv').set_index('race')['median_pit_loss']
    team_adj = pd.read_csv('data/team_pit_loss_adjustment.csv').set_index('team')['adjustment_s']

    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    print(f"Processing {len(lap_files)} races\n")

    rows = []
    skipped = 0
    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        if race_id not in pit_df.index or pd.isna(pit_df[race_id]):
            skipped += 1
            continue
        circuit_pit_loss = float(pit_df[race_id])

        laps = pd.read_parquet(f)
        laps = fuel_correct(laps)
        event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id
        n_laps = int(laps['LapNumber'].max())
        field_pace = compute_field_pace(laps)

        race_deg = deg_df[deg_df['race'] == race_id].set_index('compound').to_dict('index')
        # Require >=30 laps behind a compound's per-race curve before letting the DP
        # consider it -- thin samples (e.g. a handful of dry laps in a wet-disrupted
        # race) produce noisy slope/intercept pairs that make alternating into that
        # compound look artificially cheap, which is what drove the 6-stop
        # Emilia Romagna 2022 result caught in the initial sanity check.
        degradation = {c: race_deg[c] for c in DRY_COMPOUNDS
                        if c in race_deg and race_deg[c]['n_laps'] >= 30}
        if len(degradation) < 2:
            skipped += 1
            continue

        print(f"=== {race_id}: {event_name} ({n_laps} laps) ===")
        race_had_dp = False

        for team, team_laps in laps.groupby('Team'):
            # A team with too few pooled stops to trust gets adjustment_s=0,
            # i.e. falls back to the plain circuit-wide constant.
            pit_loss = circuit_pit_loss + float(team_adj.get(team, 0.0))

            dp = solve_dp(n_laps, field_pace, degradation, pit_loss)
            if dp is None:
                continue
            race_had_dp = True

            stop_str = ", ".join(f"L{lap} -> {c}" for lap, c in dp['stops']) or "no stops found"
            print(f"  [{team}] pit loss {pit_loss:.1f}s -- DP hindsight-optimal: {dp['cost']:.1f}s, "
                  f"{len(dp['stops'])} stop(s): {stop_str}")

            deltas = []
            for driver, driver_laps in team_laps.groupby('Driver'):
                if driver_laps['LapNumber'].max() < n_laps - 2:
                    continue  # retired before the finish -- not comparable to a full-distance DP total
                actual = replay_actual(driver_laps, field_pace, degradation, pit_loss)
                if actual is None:
                    continue
                delta = actual['cost'] - dp['cost']
                deltas.append(delta)
                rows.append({
                    'race': race_id, 'event': event_name, 'driver': driver, 'team': team,
                    'n_laps': n_laps, 'team_pit_loss_s': pit_loss,
                    'dp_optimal_s': dp['cost'], 'dp_n_stops': len(dp['stops']),
                    'actual_s': actual['cost'], 'actual_n_pits': actual['n_pits'],
                    'delta_s': delta,
                })
            if deltas:
                print(f"    {len(deltas)} drivers replayed -- delta vs. optimal: "
                      f"median {np.median(deltas):+.1f}s")

        if not race_had_dp:
            skipped += 1
        print()

    print(f"Skipped {skipped} races (missing pit-loss constant, or <2 usable compounds)")
    out = pd.DataFrame(rows)
    out.to_csv('data/dp_baseline_backtest.csv', index=False)
    print(f"Saved data/dp_baseline_backtest.csv ({len(out)} rows)")


if __name__ == '__main__':
    main()
