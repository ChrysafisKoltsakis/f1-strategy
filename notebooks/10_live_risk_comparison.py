"""
Phase 6 extension -- live risk/reward comparison for the "act now or wait"
decision under a Safety Car, prompted directly by the MAG/PER investigation
in the streaming consumer: a driver on a long first stint, tempted by a
cheap SC/VSC pit-loss window to switch from a planned 1-stop to a 2-stop.
Real teams face exactly this dilemma, and it changes track position, not
just lap time -- this tool makes the time tradeoff and the *uncertainty*
in it explicit, rather than a single point prediction.

Two pieces of real, historical evidence feed the comparison, both derived
from the 103-race dataset, not a hand-built probability model:
  1. How long SC/VSC periods actually last (their duration distribution).
  2. How often a race that's already had one SC/VSC gets ANOTHER one later
     -- this is the actual "risk" in waiting: if you stay out hoping for a
     future cheap window, how often does history say that bet pays off.

This stays inside the DP/deterministic-optimizer machinery already built
(04_dp_baseline.py) rather than a new stochastic layer -- solve_dp() is
generalized to start from an arbitrary mid-race state (not just lap 1),
so "1-stop-remaining vs. 2-stop-remaining, from right now" is a real,
re-solvable question, evaluated under a small number of concrete,
data-derived scenarios (not a full Monte Carlo simulation).

Run with: python notebooks/10_live_risk_comparison.py <race_id> <lap>
  e.g.    python notebooks/10_live_risk_comparison.py 2023_r08 8

09_streaming_consumer.py also imports this module directly (dynamic
import, same pattern as everywhere else in this project) to run this same
pit-now-vs-wait comparison live, off only the laps observed so far in the
stream, whenever its peak detector flags a candidate lap -- see
causal_field_pace_forecast() below for the one piece that had to change
for that (main()'s field-pace lookup uses real historical future laps,
which a live decision can't do).
"""
import sys
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

SC_START_STATUSES = {'DEPLOYED'}
SC_END_STATUSES = {'ENDING', 'IN THIS LAP'}


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def extract_sc_events(race_id: str) -> list:
    """(type, start_lap, end_lap) for every completed SC/VSC period in a
    race, matching a DEPLOYED message to the next ENDING/IN THIS LAP of
    the same kind."""
    rc_path = BRONZE_DIR / f"{race_id}_race_control.parquet"
    if not rc_path.exists():
        return []
    rc = pd.read_parquet(rc_path).sort_values('Time')
    sc = rc[rc['Category'] == 'SafetyCar']

    events = []
    open_event = None
    for _, row in sc.iterrows():
        kind = 'VSC' if 'VIRTUAL' in str(row['Message']) else 'SC'
        if row['Status'] in SC_START_STATUSES:
            open_event = (kind, row['Lap'])
        elif row['Status'] in SC_END_STATUSES and open_event and open_event[0] == kind:
            events.append((kind, open_event[1], row['Lap']))
            open_event = None
    return events


def build_historical_sc_stats():
    """Across all 103 races: (a) SC/VSC duration distribution, (b) how
    often a race with one SC/VSC event gets at least one more later --
    the real, historical answer to "if I wait, how often does that pay
    off." Both computed from the full population, not race-specific, so
    there's no leakage concern querying them for any one race's decision."""
    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    durations = []
    recurrence_rows = []

    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        events = extract_sc_events(race_id)
        if not events:
            continue
        n_laps = int(pd.read_parquet(f, columns=['LapNumber'])['LapNumber'].max())
        events_sorted = sorted(events, key=lambda e: e[1])
        for kind, start, end in events_sorted:
            durations.append({'race': race_id, 'type': kind, 'duration_laps': end - start,
                               'start_progress': start / n_laps})
        for i, (kind, start, end) in enumerate(events_sorted):
            had_another = any(e[1] > end for e in events_sorted[i + 1:])
            recurrence_rows.append({'race': race_id, 'start_progress': start / n_laps,
                                     'had_another_later': had_another})

    dur_df = pd.DataFrame(durations)
    rec_df = pd.DataFrame(recurrence_rows)
    return dur_df, rec_df


def p_another_event(rec_df: pd.DataFrame, current_progress: float) -> float:
    """P(at least one more SC/VSC later in the race | one is happening
    now at this race-progress point), pooled from all other historical
    SC/VSC events at a similar stage of the race (+/-15% race distance)."""
    window = rec_df[(rec_df['start_progress'] >= current_progress - 0.15) &
                     (rec_df['start_progress'] <= current_progress + 0.15)]
    if len(window) < 10:
        window = rec_df  # not enough nearby events -- fall back to the full population
    return window['had_another_later'].mean(), len(window)


def causal_field_pace_forecast(laps_so_far: pd.DataFrame, n_laps: int) -> pd.Series:
    """Live-safe field-pace baseline, for use by the streaming consumer
    (09_streaming_consumer.py) instead of main()'s hindsight version above.
    main() reads the real historical field pace for the WHOLE race,
    including laps that haven't happened yet from a live decision's point
    of view -- fine for validating the tool against known history (its
    original purpose), not fine for an actual live call, since it would
    silently bake in whether a real future SC/rain event happened.

    Laps already seen use their real observed field pace. Laps not yet
    seen are held flat at the recent green-flag median -- i.e. "assume
    conditions continue as they've been," which is the honest, causal
    definition of "no more incident," and matches what solve_dp_from_state's
    non-discounted branch is supposed to represent. The *chance* of a
    future incident is priced separately, via p_another_event()'s
    historical recurrence rate and the discounted stop price -- not by
    projecting a specific future pace-spike shape, which a live system
    can't know in advance."""
    laps_so_far = laps_so_far.copy()
    laps_so_far['TrackStatus'] = laps_so_far['TrackStatus'].astype(str)
    observed = compute_field_pace(laps_so_far)
    green = laps_so_far[~laps_so_far['TrackStatus'].apply(dpl._is_sc_or_vsc)]
    green_pace = compute_field_pace(green)
    flat_level = green_pace.tail(5).median() if len(green_pace) else observed.tail(5).median()

    last_lap = int(laps_so_far['LapNumber'].max())
    forecast = observed.reindex(range(1, n_laps + 1))
    forecast.loc[forecast.index > last_lap] = flat_level
    return forecast.rename('FieldPace')


def solve_dp_from_state(n_laps: int, field_pace: pd.Series, degradation: dict,
                         start_lap: int, start_compound: str, start_age: int, start_mask: int,
                         pit_loss: float, compounds: list, min_total_compounds: int = 2,
                         discount_price: float = None, force_stop_at_start: bool = False) -> dict:
    """Same DP as 04_dp_baseline.py's solve_dp, generalized to (a) start
    from an arbitrary mid-race state instead of always lap 1, and (b)
    optionally make ONE stop -- and only one, whichever the DP actually
    uses -- available at a discounted price. discount_price models "a
    cheap SC/VSC window is available for whichever stop uses it"; only
    the first stop taken gets it, every stop after reverts to the normal
    pit_loss, since a real SC/VSC period doesn't stay open for the whole
    remainder of the race. force_stop_at_start=True additionally requires
    that first (discounted) stop to happen immediately, for "pit under
    the SC right now" rather than "a discount becomes available at some
    point, whenever the optimizer prefers."""
    def expected_pace(lap, compound, age):
        pace_lookup = degradation.get(compound)
        fp = field_pace.get(lap)
        if pace_lookup is None or fp is None or pd.isna(fp):
            return None
        rel_pace = pace_lookup.get((int(lap), int(age)))
        if rel_pace is None:
            return None
        return fp + rel_pace

    def bit(c):
        return 1 << compounds.index(c)

    has_discount = discount_price is not None
    # state: (compound, age, mask, discount_still_available)
    states = {(start_compound, start_age, start_mask, has_discount): (0.0, [])}

    for lap in range(start_lap, n_laps + 1):
        new_states = {}
        for (c, age, mask, disc), (cost, path) in states.items():
            must_stop_this_lap = force_stop_at_start and lap == start_lap
            if not must_stop_this_lap:
                ep = expected_pace(lap, c, age + 1)
                if ep is not None:
                    key = (c, age + 1, mask, disc)
                    new_cost = cost + ep
                    if new_cost < new_states.get(key, (INF, None))[0]:
                        new_states[key] = (new_cost, path)
            for c2 in compounds:
                if c2 == c:
                    continue
                ep2 = expected_pace(lap, c2, 1)
                if ep2 is None:
                    continue
                stop_cost = discount_price if disc else pit_loss
                key2 = (c2, 1, mask | bit(c2), False)  # discount (if any) is spent now
                new_cost2 = cost + stop_cost + ep2
                if new_cost2 < new_states.get(key2, (INF, None))[0]:
                    new_states[key2] = (new_cost2, path + [(lap, c2)])
        states = new_states

    valid = [(cost, path) for (c, age, mask, disc), (cost, path) in states.items()
             if bin(mask).count('1') >= min_total_compounds]
    if not valid:
        return None
    best_cost, best_path = min(valid, key=lambda x: x[0])
    return {'cost': best_cost, 'stops': best_path}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    race_id, decision_lap = sys.argv[1], int(sys.argv[2])

    print("Building historical SC/VSC statistics across all races...")
    dur_df, rec_df = build_historical_sc_stats()
    print(f"  {len(dur_df)} completed SC/VSC events, {dur_df['race'].nunique()} races with at least one\n")
    print("Duration distribution (laps):")
    print(dur_df.groupby('type')['duration_laps'].describe()[['count', 'mean', '50%', 'max']])

    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    laps = fuel_correct(laps)
    n_laps = int(laps['LapNumber'].max())
    field_pace = compute_field_pace(laps)
    event_name = laps['EventName'].iloc[0]

    deg_df = pd.read_csv('data/degradation_summary.csv')
    race_deg = deg_df[deg_df['race'] == race_id].set_index('compound').to_dict('index')
    eligible = [c for c in DRY_COMPOUNDS if c in race_deg and race_deg[c]['n_laps'] >= 5]
    ml_bundle = dm.load_degradation_model()
    ml_lookup = dm.build_ml_pace_lookup(race_id, ml_bundle, eligible)
    degradation = {c: ml_lookup[c] for c in eligible if c in ml_lookup}
    compounds = list(degradation.keys())

    green_pit_loss = float(pd.read_csv('data/pit_loss_summary.csv').set_index('race')['median_pit_loss'][race_id])
    sc_pit_loss = float(pd.read_csv('data/sc_pit_loss_summary.csv').set_index('race')['median_pit_loss'].get(race_id, green_pit_loss * 0.5))

    print(f"\n{'='*70}\n{race_id} ({event_name}), decision point: lap {decision_lap}")
    print(f"Green pit loss: {green_pit_loss:.1f}s | SC/VSC pit loss: {sc_pit_loss:.1f}s\n")

    for driver in ['MAG', 'PER']:
        row = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == decision_lap)]
        if row.empty:
            continue
        row = row.iloc[0]
        compound, age = row['Compound'], int(row['TyreLife'])
        if compound not in compounds:
            continue
        mask_bit = 1 << compounds.index(compound)
        progress = decision_lap / n_laps

        print(f"--- {driver}: currently on {compound}, tyre age {age}, lap {decision_lap}/{n_laps} ---")

        # Option A: pit NOW under the cheap SC/VSC window (only this one
        # stop is discounted; anything after reverts to normal green
        # pricing, since a real SC/VSC period doesn't stay open all race).
        opt_a = solve_dp_from_state(n_laps, field_pace, degradation, decision_lap, compound, age,
                                     mask_bit, green_pit_loss, compounds, min_total_compounds=2,
                                     discount_price=sc_pit_loss, force_stop_at_start=True)

        # Option B, worst case: stay out, no more SC/VSC luck -- the
        # eventual stop (whenever the DP prefers) costs full green price.
        opt_b_worst = solve_dp_from_state(n_laps, field_pace, degradation, decision_lap, compound, age,
                                           mask_bit, green_pit_loss, compounds, min_total_compounds=2)
        # Option B, best case: the bet pays off -- a discount becomes
        # available for whichever stop the DP ends up choosing (not
        # necessarily right now), same one-stop-only discount mechanic.
        opt_b_best = solve_dp_from_state(n_laps, field_pace, degradation, decision_lap, compound, age,
                                          mask_bit, green_pit_loss, compounds, min_total_compounds=2,
                                          discount_price=sc_pit_loss, force_stop_at_start=False)

        if not (opt_a and opt_b_worst and opt_b_best):
            print("  (not enough data to solve from this state)\n")
            continue

        p_again, n_events = p_another_event(rec_df, progress)
        expected_b = p_again * opt_b_best['cost'] + (1 - p_again) * opt_b_worst['cost']

        print(f"  Option A -- pit now under SC/VSC:            {opt_a['cost']:.1f}s "
              f"({len(opt_a['stops'])} more stop(s): {opt_a['stops']})")
        print(f"  Option B -- wait, worst case (no more SC):   {opt_b_worst['cost']:.1f}s "
              f"({len(opt_b_worst['stops'])} more stop(s))")
        print(f"  Option B -- wait, best case (SC comes again): {opt_b_best['cost']:.1f}s")
        print(f"  P(another SC/VSC later in a race at this stage) = {p_again*100:.0f}% "
              f"(from {n_events} historical events at similar race progress)")
        print(f"  Option B expected cost: {expected_b:.1f}s")
        diff = expected_b - opt_a['cost']
        verdict = "pit now (A) looks better on expectation" if diff > 0 else "waiting (B) looks better on expectation"
        print(f"  --> {verdict}: {abs(diff):.1f}s {'saved by pitting now' if diff>0 else 'saved by waiting'} "
              f"on average, but B's worst case is {opt_b_worst['cost']-opt_a['cost']:+.1f}s vs A\n")


if __name__ == '__main__':
    main()
