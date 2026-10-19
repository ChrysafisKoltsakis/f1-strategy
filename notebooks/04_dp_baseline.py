"""
Phase 3 -- DP hindsight-optimal baseline. Deliberately kept small: its only
job is producing the "hindsight-optimal" column for the eventual actual
vs. classifier vs. hindsight-optimal backtest. Deterministic only -- no
safety-car/rain stochastic extension, that scope is explicitly out per the
project's scope guardrail.

Formulation follows Carrasco Heine et al. 2022, "On the optimization of pit
stop strategies via dynamic programming": state = (current compound, tyre
age, set of compounds used so far), transition each lap = stay out vs. pit
onto a fresh set of any compound (including the one already fitted, so
e.g. MEDIUM -> HARD -> HARD is reachable), objective = minimize total race
time subject to using at least two distinct compounds (the mandatory
two-compound rule, enforced on the final mask rather than per stop).

Degradation now comes from the trained ML model (03_ml_degradation_model.py's
LightGBM, loaded dynamically since notebook filenames can't be `import`ed
normally), not the per-race linear regression this file originally used --
that per-race-only fit couldn't capture non-linear/cliff wear and had to
be refit independently for every race with no pooling. The ML model pools
across every race/circuit pulled (leave-one-circuit-out validated,
~0.75s MAE) and is queried via a precomputed per-race (compound, lap,
tyre age) grid -- build_ml_pace_lookup() in 03 -- rather than per-state
model.predict() calls, since the DP explores thousands of states. Uses
the race's own real historical weather (legitimate for replaying a
historical race, same principle FieldPace itself already relies on).
Green and SC/VSC pit-loss constants come from 02_fixed_degradation_pitloss.py
(data/pit_loss_summary.csv, data/sc_pit_loss_summary.csv), plus per-lap
FieldPace recomputed from Bronze laps. A stop at the end of a lap most of
the field ran under SC/VSC is priced at the SC/VSC pit loss -- in the
replayed real strategies and in the hindsight optimum alike (which may use
the race's real safety cars, being hindsight). Full SC and VSC are priced
separately: each stop costs the circuit's green pit loss x a pooled ratio
measured over individual stops of that kind (~0.21 SC, ~0.68 VSC; per-race
estimates are too noisy, and near-zero ones let the DP pit on every SC
lap). A full safety car also bunches the field, erasing most accumulated
gaps -- the effect Heilmeier et al.'s race simulation models with a
"ghost" safety car. A per-car model can't represent positions, so it uses
the statistical equivalent: when a full safety car comes out, a plan's
deficit to the field median accumulated so far is multiplied by beta
(~0.21, fitted from real gaps before/after safety cars, 02's
sc_gap_compression); stops during the safety car keep their measured cost.
Only the first stop in each SC/VSC period is cheap (02's stop_price): a
second stop under the same safety car costs track position a per-car model
can't see, and without the rule the DP stopped up to 4 times under one. Each driver's *actual* strategy is replayed through this
same model (not their real recorded lap times), so the comparison
isolates "was the strategy choice good" from "was the driving good" --
actual-strategy cost and DP-optimal cost are both estimates under the
identical model.

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
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'degradation_model_lib', Path(__file__).parent / '03_ml_degradation_model.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

_spec2 = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(dpl)

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


def solve_dp(n_laps: int, field_pace: pd.Series, degradation: dict, pit_loss: float,
             neutral: dict | None = None) -> dict:
    """Returns {'cost': float, 'stops': [(lap, compound), ...]} for the
    hindsight-optimal strategy, or None if fewer than 2 usable compounds.
    With `neutral` (02's neutralisation()), the race's real safety cars are
    priced like the real strategies it's compared with: a stop at the end
    of an SC or VSC lap costs that kind's price, and when a full safety car
    comes out every plan's deficit to the field accumulated so far shrinks
    by beta. The same increasing transform applied to every state on the same
    lap preserves their order, so the DP stays exact."""
    compounds = [c for c in DRY_COMPOUNDS if c in degradation]
    if len(compounds) < 2:
        return None

    def expected_pace(lap, compound, age):
        rel_pace = degradation[compound].get((lap, age))
        fp = field_pace.get(lap)
        if rel_pace is None or fp is None or pd.isna(fp):
            return None
        return fp + rel_pace

    def bit(c):
        return 1 << compounds.index(c)

    def compress(states_now, lap, fp_cum):
        if neutral is None or lap not in neutral['compress_after']:
            return states_now
        beta = neutral['beta']
        return {k: (fp_cum + beta * (cost - fp_cum), path) for k, (cost, path) in states_now.items()}

    # state key: (compound, age, compounds-used mask, last neutralisation
    # period a cheap stop was made in) -> (cost, path)
    states = {}
    for c in compounds:
        ep = expected_pace(1, c, 1)
        if ep is None:
            continue
        states[(c, 1, bit(c), -1)] = (ep, [])
    fp_cum = float(field_pace.get(1, 0.0))
    states = compress(states, 1, fp_cum)

    for lap in range(2, n_laps + 1):
        new_states = {}
        for (c, age, mask, last_p), (cost, path) in states.items():
            ep = expected_pace(lap, c, age + 1)
            if ep is not None:
                key = (c, age + 1, mask, last_p)
                new_cost = cost + ep
                if new_cost < new_states.get(key, (INF, None))[0]:
                    new_states[key] = (new_cost, path)
            for c2 in compounds:
                ep2 = expected_pace(lap, c2, 1)
                if ep2 is None:
                    continue
                price, new_p = dpl.stop_price(neutral, lap - 1, pit_loss, last_p)
                key2 = (c2, 1, mask | bit(c2), new_p)
                new_cost2 = cost + price + ep2
                if new_cost2 < new_states.get(key2, (INF, None))[0]:
                    new_states[key2] = (new_cost2, path + [(lap, c2)])
        fp_cum += float(field_pace.get(lap, 0.0))
        states = compress(new_states, lap, fp_cum)

    valid = [(cost, path) for (c, age, mask, _), (cost, path) in states.items()
             if bin(mask).count('1') >= 2]
    if not valid:
        return None
    best_cost, best_path = min(valid, key=lambda x: x[0])
    return {'cost': best_cost, 'stops': best_path}


def replay_actual_laps(driver_laps: pd.DataFrame, field_pace: pd.Series, degradation: dict, pit_loss: float,
                       neutral: dict | None = None):
    """Per-lap cost of a driver's real stint sequence through the same
    expected-pace model the DP uses -- not their real lap times. Returns a
    {'laps': Series of predicted lap time by lap number, with each stop's
    pit loss on its out-lap, 'n_pits', 'compounds'} dict, or None if any lap
    can't be priced or the two-compound rule isn't met. Per-lap so
    13_strategy_cost_validation.py can compare teammates over the laps both
    completed."""
    driver_laps = driver_laps.sort_values('LapNumber')
    lap_costs = {}
    compounds_used = set()
    for _, row in driver_laps.iterrows():
        compound, age, lap = row['Compound'], row['TyreLife'], row['LapNumber']
        if compound not in degradation or pd.isna(age):
            return None
        age_key = int(age)
        if age_key > dm.MAX_AGE_GRID:
            return None  # stint longer than the precomputed grid -- implausible, treat as unsupported
        rel_pace = degradation[compound].get((int(lap), age_key))
        fp = field_pace.get(lap)
        if rel_pace is None or fp is None or pd.isna(fp):
            return None
        lap_costs[int(lap)] = fp + rel_pace
        compounds_used.add(compound)
    if len(compounds_used) < 2:
        return None
    # A stop = the tyre set actually changed: a different compound, or tyre
    # age that didn't just carry on from the previous lap. Not TyreLife
    # resetting to 1 -- TyreLife counts laps from earlier sessions, so a stop
    # onto a used set (e.g. scrubbed quali softs) starts above 1. And not
    # Stint changes -- FastF1 opens a new stint on every pit-lane pass,
    # including drive-throughs and red-flag stoppages where the set stays on.
    # A change made under a red flag ('5') is free, the race being suspended,
    # so it isn't charged pit loss. With `neutral`, a stop at the end of an
    # SC or VSC lap costs that kind's price, and the deficit to the field
    # accumulated before each full safety car shrinks by beta (see solve_dp).
    prev = driver_laps.shift()
    new_set = ((driver_laps['Compound'] != prev['Compound'])
               | (driver_laps['TyreLife'] - prev['TyreLife'] != driver_laps['LapNumber'] - prev['LapNumber']))
    under_red_flag = prev['TrackStatus'].astype(str).str.contains('5')
    charged = (new_set & ~under_red_flag).iloc[1:]
    last_p = -1
    for lap in sorted(driver_laps.loc[charged[charged].index, 'LapNumber']):
        price, last_p = dpl.stop_price(neutral, int(lap) - 1, pit_loss, last_p)
        lap_costs[int(lap)] += price
    if neutral is not None and neutral['compress_after']:
        deficit = 0.0
        for lap in sorted(lap_costs):
            deficit += lap_costs[lap] - float(field_pace.get(lap, 0.0))
            if lap in neutral['compress_after']:
                lap_costs[lap] += (neutral['beta'] - 1.0) * deficit
                deficit *= neutral['beta']
    return {'laps': pd.Series(lap_costs).sort_index(), 'n_pits': int(charged.sum()), 'compounds': compounds_used}


def replay_actual(driver_laps: pd.DataFrame, field_pace: pd.Series, degradation: dict, pit_loss: float,
                  neutral: dict | None = None):
    """Total cost of a driver's real stint sequence (see replay_actual_laps)."""
    r = replay_actual_laps(driver_laps, field_pace, degradation, pit_loss, neutral)
    if r is None:
        return None
    return {'cost': float(r['laps'].sum()), 'n_pits': r['n_pits'], 'compounds': r['compounds']}


def main():
    deg_df = pd.read_csv('data/degradation_summary.csv')
    pit_df = pd.read_csv('data/pit_loss_summary.csv').set_index('race')['median_pit_loss']
    team_adj = pd.read_csv('data/team_pit_loss_adjustment.csv').set_index('team')['adjustment_s']
    ratios = dpl.neutralised_pit_ratios()   # {'SC': .., 'VSC': ..} share of green pit loss
    beta = dpl.sc_gap_compression()         # deficit kept through a full safety car
    print(f"SC/VSC pricing: SC stop {ratios['SC']:.2f} x green, VSC stop {ratios['VSC']:.2f} x green, "
          f"gap compression through a full SC beta = {beta:.2f}\n")
    ml_bundle = dm.load_degradation_model()

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
        # Eligibility: was this compound genuinely used in this race at all
        # (not just a formation-lap fluke)? Unlike the old per-race linear
        # fit, the ML model doesn't need many laps *from this race* to
        # predict reliably -- it pools across every race -- so this is a
        # much lighter bar (>=5 laps) than the >=30 the linear version
        # needed to avoid noisy per-race slopes. The Emilia Romagna 2022
        # 6-stop sanity-check failure that originally motivated that gate
        # was a property of the old model, not of compound usage itself.
        eligible = [c for c in DRY_COMPOUNDS if c in race_deg and race_deg[c]['n_laps'] >= 5]
        if len(eligible) < 2:
            skipped += 1
            continue
        ml_lookup = dm.build_ml_pace_lookup(race_id, ml_bundle, eligible)
        degradation = {c: ml_lookup[c] for c in eligible if c in ml_lookup}
        if len(degradation) < 2:
            skipped += 1
            continue

        print(f"=== {race_id}: {event_name} ({n_laps} laps) ===")
        race_had_dp = False

        for team, team_laps in laps.groupby('Team'):
            # A team with too few pooled stops to trust gets adjustment_s=0,
            # i.e. falls back to the plain circuit-wide constant.
            pit_loss = circuit_pit_loss + float(team_adj.get(team, 0.0))
            # SC and VSC stops: the circuit's green pit loss x each kind's
            # pooled ratio, plus the team's pit-crew adjustment, which
            # applies under SC/VSC too.
            sc_price = circuit_pit_loss * ratios['SC'] + float(team_adj.get(team, 0.0))
            vsc_price = circuit_pit_loss * ratios['VSC'] + float(team_adj.get(team, 0.0))
            neutral = dpl.neutralisation(laps, sc_price, vsc_price, beta)

            dp = solve_dp(n_laps, field_pace, degradation, pit_loss, neutral)
            if dp is None:
                continue
            race_had_dp = True

            stop_str = ", ".join(f"L{lap} -> {c}" for lap, c in dp['stops']) or "no stops found"
            print(f"  [{team}] pit loss {pit_loss:.1f}s -- DP hindsight-optimal: {dp['cost']:.1f}s, "
                  f"{len(dp['stops'])} stop(s): {stop_str}")

            # A lapped driver's actual cost only covers the laps they drove,
            # so they're compared against the DP optimum over that same
            # distance (each lap ~90s, so a full-distance optimum would make
            # any lapped driver's real strategy look better than optimal).
            dp_by_distance = {n_laps: dp}
            deltas = []
            for driver, driver_laps in team_laps.groupby('Driver'):
                driver_n_laps = int(driver_laps['LapNumber'].max())
                if driver_n_laps < n_laps - 2:
                    continue  # retired, not lapped -- a partial race isn't a strategy to evaluate
                if driver_n_laps not in dp_by_distance:
                    dp_by_distance[driver_n_laps] = solve_dp(driver_n_laps, field_pace, degradation, pit_loss,
                                                             neutral)
                driver_dp = dp_by_distance[driver_n_laps]
                if driver_dp is None:
                    continue
                actual = replay_actual(driver_laps, field_pace, degradation, pit_loss, neutral)
                if actual is None:
                    continue
                delta = actual['cost'] - driver_dp['cost']
                deltas.append(delta)
                rows.append({
                    'race': race_id, 'event': event_name, 'driver': driver, 'team': team,
                    'n_laps': n_laps, 'driver_n_laps': driver_n_laps, 'team_pit_loss_s': pit_loss,
                    'team_sc_pit_loss_s': sc_price, 'team_vsc_pit_loss_s': vsc_price,
                    'dp_optimal_s': driver_dp['cost'], 'dp_n_stops': len(driver_dp['stops']),
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
