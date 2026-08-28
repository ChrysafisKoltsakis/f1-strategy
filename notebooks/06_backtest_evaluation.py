"""
Phase 5 -- the 3-way backtest: actual strategy vs. classifier-driven
strategy vs. DP hindsight-optimal, all replayed through the same
expected-pace model so the comparison is about strategy quality, not
driving quality. This is the thesis's central evaluation chapter.

Scoped to the 20 races the trigger classifier was evaluated on (held out
of its own training, per notebooks/05_trigger_classifier.py) intersected
with the races the DP baseline could solve (notebooks/04_dp_baseline.py
already excludes wet/disrupted races and races missing a pit-loss
constant) -- so all three legs of the comparison are computed on exactly
the same race/driver set.

Classifier -> strategy conversion: the classifier only predicts *when* to
pit, not which compound to fit, so a driver's predicted-positive laps are
first collapsed into pit *events* (consecutive/near predictions within 2
laps count as one stop, not several), then at each event the compound is
chosen greedily by whichever eligible compound minimizes expected pace
from that point -- while still short of the two-compound rule, an unused
compound is forced; a driver with zero predicted stops is flagged as an
explicit rule-violation failure mode rather than silently reported as if
it were a legal, comparable strategy.

Run with: python notebooks/06_backtest_evaluation.py
"""
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'degradation_model_lib', Path(__file__).parent / '03_ml_degradation_model.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

BRONZE_DIR = Path('data/bronze')
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
PIT_CLUSTER_GAP = 2  # predicted-pit laps within this many laps of each other count as one stop


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def argmax_single_stop(dgrp: pd.DataFrame, n_laps: int) -> list:
    """Ranking-based decision rule: ignore the classification threshold
    entirely and just take the single lap with the highest predicted
    probability, as long as it's a realistic lap to pit on (not the first
    or last couple of laps). Tests whether the model's probability
    *ranking* is more useful than its thresholded yes/no output -- most
    real races are one-stop anyway (76/83 in the DP baseline), so a
    single-stop rule is a fair like-for-like comparison."""
    eligible = dgrp[(dgrp['lap'] > 2) & (dgrp['lap'] < n_laps - 1)]
    if eligible.empty:
        return []
    best_lap = eligible.loc[eligible['pred_proba_balanced'].idxmax(), 'lap']
    return [int(best_lap)]


MIN_STOP_GAP = 8   # laps -- real stints are essentially never shorter than this
MAX_STOPS = 3       # ceiling matching the number of dry compounds available


def select_candidate_stops(dgrp: pd.DataFrame, n_laps: int, k: int, min_gap: int = MIN_STOP_GAP) -> list:
    """Greedily pick up to k laps with the highest predicted probability,
    each at least min_gap laps from any lap already picked. These are
    *candidate* stop locations only -- the classifier proposes where a
    stop might make sense; how many of them are actually worth taking is
    decided separately by simulating each stop-count through the same
    expected-pace/pit-loss cost model the DP baseline uses, and keeping
    whichever total time is lowest. This keeps "how many stops" grounded
    in the physics-based cost model rather than an arbitrary probability
    cutoff -- which is exactly the failure mode the threshold rule had."""
    eligible = dgrp[(dgrp['lap'] > 2) & (dgrp['lap'] < n_laps - 1)].sort_values(
        'pred_proba_balanced', ascending=False)
    picked = []
    for _, row in eligible.iterrows():
        lap = int(row['lap'])
        if all(abs(lap - p) >= min_gap for p in picked):
            picked.append(lap)
        if len(picked) >= k:
            break
    return sorted(picked)


def rank_multi_stop(dgrp: pd.DataFrame, start_compound: str, n_laps: int,
                     field_pace: pd.Series, degradation: dict, pit_loss: float):
    """Try 1, 2, and 3 candidate stops (from select_candidate_stops) and
    keep whichever gives the lowest predicted total race time. Returns
    (total_seconds, stop_laps) for the winning stop-count, or (None, [])
    if nothing produced a legal (>=2-compound) strategy."""
    best_s, best_laps = None, []
    for k in range(1, MAX_STOPS + 1):
        candidate_laps = select_candidate_stops(dgrp, n_laps, k)
        if len(candidate_laps) < k:
            break  # ran out of well-separated candidates -- more stops won't help
        s = replay_strategy(candidate_laps, start_compound, n_laps, field_pace, degradation, pit_loss)
        if s is not None and (best_s is None or s < best_s):
            best_s, best_laps = s, candidate_laps
    return best_s, best_laps


def cluster_pit_events(pit_laps: list) -> list:
    """Collapse consecutive/near predicted-positive laps into single pit
    events -- a burst of predictions at laps 30, 31, 32 is one stop, not
    three, since the model firing for a few laps around the real trigger
    point is a near-hit, not three separate pit calls."""
    if not pit_laps:
        return []
    pit_laps = sorted(pit_laps)
    events = [[pit_laps[0]]]
    for lap in pit_laps[1:]:
        if lap - events[-1][-1] <= PIT_CLUSTER_GAP:
            events[-1].append(lap)
        else:
            events.append([lap])
    return [event[0] for event in events]  # pit on the first lap of each cluster


def replay_strategy(stop_laps: list, start_compound: str, n_laps: int,
                     field_pace: pd.Series, degradation: dict, pit_loss: float):
    """Cost of a given (pit-lap list, compound-choice policy) strategy,
    evaluated through the same expected-pace model as the DP baseline and
    the actual-strategy replay. At each stop, picks whichever eligible
    compound minimizes total expected pace over the *whole upcoming
    stint* (to the next stop, or race end), not just its first lap --
    stop_laps is fully known upfront, so the stint length at each
    decision point is knowable in advance. A compound that's fastest at
    tyre age 1 can still be the wrong choice if it degrades faster over
    a long stint; comparing only age-1 pace systematically favored
    whichever compound looks best on lap one regardless of how long it
    then has to survive, which was quietly costing 2-stop plans more
    than 1-stop ones (more, and less predictable, stint lengths) --
    found by comparing the classifier's multi-stop ranking against
    DP-optimal and seeing it under-call 2-stops even when the ranking's
    own candidate-lap confidence was high, so the gap had to be in how a
    chosen plan's cost was computed, not in which laps got proposed.
    Forces an unused compound first if the two-compound rule isn't
    satisfied yet."""
    def expected_pace(lap, compound, age):
        pace_lookup = degradation.get(compound)
        fp = field_pace.get(lap)
        if pace_lookup is None or fp is None or pd.isna(fp):
            return None
        rel_pace = pace_lookup.get((int(lap), int(age)))
        if rel_pace is None:
            return None
        return fp + rel_pace

    stops_sorted = sorted(stop_laps)

    def stint_end(lap):
        later = [s for s in stops_sorted if s > lap]
        return (later[0] - 1) if later else n_laps

    def best_compound(used: set, lap: int):
        candidates = [c for c in degradation if c not in used] or list(degradation)
        end = stint_end(lap)
        scored = []
        for c in candidates:
            paces = [expected_pace(l, c, i + 1) for i, l in enumerate(range(int(lap), int(end) + 1))]
            if any(p is None for p in paces):
                continue
            scored.append((c, sum(paces)))
        if not scored:
            return None
        return min(scored, key=lambda x: x[1])[0]

    if start_compound not in degradation:
        start_compound = best_compound(set(), 1)
        if start_compound is None:
            return None

    compound = start_compound
    used = {compound}
    age = 0
    total = 0.0
    stop_set = set(stop_laps)
    for lap in range(1, n_laps + 1):
        if lap in stop_set:
            compound = best_compound(used, lap)
            if compound is None:
                return None
            used.add(compound)
            age = 0
            total += pit_loss
        age += 1
        ep = expected_pace(lap, compound, age)
        if ep is None:
            return None
        total += ep
    if len(used) < 2:
        return None  # illegal strategy -- never satisfied the two-compound rule
    return total


def main():
    pred = pd.read_csv('data/trigger_classifier_test_predictions.csv')
    dp = pd.read_csv('data/dp_baseline_backtest.csv')
    deg_df = pd.read_csv('data/degradation_summary.csv')
    ml_bundle = dm.load_degradation_model()

    races = sorted(set(pred['race'].unique()) & set(dp['race'].unique()))
    print(f"3-way backtest over {len(races)} races (test-set races the DP baseline could also solve)\n")

    rows = []
    zero_stop_drivers = 0
    for race_id in races:
        laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
        laps = fuel_correct(laps)
        n_laps = int(laps['LapNumber'].max())
        field_pace = compute_field_pace(laps)

        race_deg = deg_df[deg_df['race'] == race_id].set_index('compound').to_dict('index')
        eligible = [c for c in DRY_COMPOUNDS if c in race_deg and race_deg[c]['n_laps'] >= 5]
        if len(eligible) < 2:
            continue
        ml_lookup = dm.build_ml_pace_lookup(race_id, ml_bundle, eligible)
        degradation = {c: ml_lookup[c] for c in eligible if c in ml_lookup}
        if len(degradation) < 2:
            continue

        dp_race = dp[dp['race'] == race_id].set_index('driver')
        race_pred = pred[pred['race'] == race_id]

        for driver, dgrp in race_pred.groupby('driver'):
            if driver not in dp_race.index:
                continue
            dp_row = dp_race.loc[driver]
            # Team-specific pit loss, matching what the DP baseline used for
            # this driver's team (see 04_dp_baseline.py) -- a faster pit
            # crew changes what's actually optimal, not just the cost of a
            # fixed plan, so classifier/ranking replays need the same
            # team-adjusted constant to be a fair comparison against it.
            pit_loss = float(dp_row['team_pit_loss_s'])

            start_compound = laps.loc[
                (laps['Driver'] == driver) & (laps['LapNumber'] == 1), 'Compound']
            if start_compound.empty:
                continue
            start_compound = start_compound.iloc[0]

            pit_laps = cluster_pit_events(dgrp.loc[dgrp['pred_balanced'] == 1, 'lap'].tolist())
            used_fallback = False
            if not pit_laps:
                # Nothing crossed threshold -- rather than reporting this
                # driver-race as an outright failure, fall back to the
                # single highest-probability lap so it still gets a legal,
                # comparable strategy. Tracked via used_fallback so the
                # 3-way summary can show the effect explicitly instead of
                # quietly folding low-confidence picks into the headline
                # numbers.
                zero_stop_drivers += 1
                used_fallback = True
                best_lap = dgrp.loc[dgrp['pred_proba_balanced'].idxmax(), 'lap']
                pit_laps = [int(best_lap)]

            classifier_s = replay_strategy(pit_laps, start_compound, n_laps,
                                            field_pace, degradation, pit_loss)

            ranked_laps = argmax_single_stop(dgrp, n_laps)
            ranked_s = (replay_strategy(ranked_laps, start_compound, n_laps,
                                         field_pace, degradation, pit_loss)
                        if ranked_laps else None)

            ranked_multi_s, ranked_multi_laps = rank_multi_stop(
                dgrp, start_compound, n_laps, field_pace, degradation, pit_loss)

            rows.append({
                'race': race_id, 'event': dp_row['event'], 'driver': driver, 'n_laps': n_laps,
                'dp_optimal_s': dp_row['dp_optimal_s'],
                'actual_s': dp_row['actual_s'],
                'classifier_s': classifier_s,
                'classifier_n_stops': len(pit_laps),
                'classifier_used_fallback': used_fallback,
                'classifier_delta_vs_optimal': (classifier_s - dp_row['dp_optimal_s']) if classifier_s else None,
                'actual_delta_vs_optimal': dp_row['delta_s'],
                'ranked_s': ranked_s,
                'ranked_delta_vs_optimal': (ranked_s - dp_row['dp_optimal_s']) if ranked_s else None,
                'ranked_multi_s': ranked_multi_s,
                'ranked_multi_n_stops': len(ranked_multi_laps),
                'ranked_multi_delta_vs_optimal': (ranked_multi_s - dp_row['dp_optimal_s']) if ranked_multi_s else None,
            })

    out = pd.DataFrame(rows)
    valid = out.dropna(subset=['classifier_s'])
    fallback = valid[valid['classifier_used_fallback']]
    fired = valid[~valid['classifier_used_fallback']]
    print(f"{len(out)} driver-race rows; {zero_stop_drivers} needed the highest-probability-lap "
          f"fallback (nothing crossed threshold)")
    print(f"{len(valid)} rows with a legal (>=2-compound) classifier-driven strategy "
          f"({len(fired)} threshold-fired, {len(fallback)} fallback)\n")

    def report(label, d, delta_col='classifier_delta_vs_optimal'):
        if d.empty:
            print(f"  {label:34s} n=0")
            return
        beat = (d[delta_col] < d['actual_delta_vs_optimal']).sum()
        print(f"  {label:34s} n={len(d):<4d} median {d[delta_col].median():+7.1f}s vs. optimal, "
              f"actual median {d['actual_delta_vs_optimal'].median():+7.1f}s, beats actual {beat}/{len(d)}")

    print("Median seconds lost vs. DP hindsight-optimal, broken out by how the strategy was picked:")
    report("All (threshold-fired + fallback)", valid)
    report("Threshold-fired only", fired)
    report("Fallback only", fallback)

    ranked_valid = out.dropna(subset=['ranked_s'])
    ranked_multi_valid = out.dropna(subset=['ranked_multi_s'])
    print(f"\n=== Threshold+cluster vs. single-stop ranking vs. multi-stop-capable ranking ===")
    report("Threshold+cluster+fallback (all)", valid)
    report("Ranking, single-stop only", ranked_valid, delta_col='ranked_delta_vs_optimal')
    report("Ranking, multi-stop-capable", ranked_multi_valid, delta_col='ranked_multi_delta_vs_optimal')
    print(f"\nStop-count chosen by multi-stop ranking vs. DP-optimal's own stop count:")
    stop_compare = out.dropna(subset=['ranked_multi_n_stops'])
    dp_n_stops = dp.set_index(['race', 'driver'])['dp_n_stops']
    stop_compare = stop_compare.set_index(['race', 'driver'])
    stop_compare['dp_n_stops'] = dp_n_stops
    matches = (stop_compare['ranked_multi_n_stops'] == stop_compare['dp_n_stops']).sum()
    print(f"  Matches DP-optimal's stop count on {matches}/{len(stop_compare)} driver-races")
    print(stop_compare['ranked_multi_n_stops'].value_counts().sort_index().to_string())

    out.to_csv('data/backtest_3way.csv', index=False)
    print(f"\nSaved data/backtest_3way.csv ({len(out)} rows)")


if __name__ == '__main__':
    main()
