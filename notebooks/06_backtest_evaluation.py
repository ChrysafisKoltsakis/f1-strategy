"""
Phase 5 -- the 3-way backtest: actual strategy vs. classifier-driven
strategy vs. DP hindsight-optimal, all replayed through the same
expected-pace model so the comparison is about strategy quality, not
driving quality. This is the thesis's central evaluation chapter.

Scoped to the races the trigger classifier was evaluated on (the most
recent ~20%, held out of its own training, per
notebooks/05_trigger_classifier.py) intersected with the races the DP
baseline could solve (notebooks/04_dp_baseline.py already excludes
wet/disrupted races and races missing a pit-loss constant) -- so all
three legs of the comparison are computed on exactly the same race/driver
set, and over the same distance per driver (a lapped driver's own lap
count, as in 04).

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
import sys
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'degradation_model_lib', Path(__file__).parent / '03_ml_degradation_model.py')
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

_spec3 = importlib.util.spec_from_file_location(
    'default_strategy_lib', Path(__file__).parent / '14_default_strategy.py')
ds = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(ds)

_spec2 = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(dpl)

_spec4 = importlib.util.spec_from_file_location(
    'live_risk_lib', Path(__file__).parent / '10_live_risk_comparison.py')
lr = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(lr)

_spec5 = importlib.util.spec_from_file_location(
    'position_model_lib', Path(__file__).parent / '15_position_model.py')
pos = importlib.util.module_from_spec(_spec5)
_spec5.loader.exec_module(pos)

# --position: also replay the position-aware policy (15's PositionDecider;
# ~1 hour more over the rolling races). Scored in 15's position backtest.
POSITION_POLICY = '--position' in sys.argv
# --strategy: also replay the strategy-level policy (15's StrategyDecider).
STRATEGY_POLICY = '--strategy' in sys.argv

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


def candidate_window(dgrp: pd.DataFrame, n_laps: int) -> pd.DataFrame:
    """Laps a strategic stop can be recommended on: not the first two
    (a lap-1 stop is a reaction to first-lap damage, which no lap-1 race
    state can predict -- the classifier still scores ~40% of drivers above
    0.5 there, against a real pit rate of ~3%) and not the last lap or two.
    Every decision rule here, and the live peak detector in 09, uses this
    same window so they're compared like for like."""
    return dgrp[(dgrp['lap'] > 2) & (dgrp['lap'] < n_laps - 1)]


def argmax_single_stop(dgrp: pd.DataFrame, n_laps: int) -> list:
    """Ranking-based decision rule: ignore the classification threshold
    entirely and just take the single lap with the highest predicted
    probability, as long as it's a realistic lap to pit on (not the first
    or last couple of laps). Tests whether the model's probability
    *ranking* is more useful than its thresholded yes/no output -- most
    real races are one-stop anyway (101/118 in the DP baseline), so a
    single-stop rule is a fair like-for-like comparison."""
    eligible = candidate_window(dgrp, n_laps)
    if eligible.empty:
        return []
    best_lap = eligible.loc[eligible['pred_proba'].idxmax(), 'lap']
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
    eligible = candidate_window(dgrp, n_laps).sort_values(
        'pred_proba', ascending=False)
    picked = []
    for _, row in eligible.iterrows():
        lap = int(row['lap'])
        if all(abs(lap - p) >= min_gap for p in picked):
            picked.append(lap)
        if len(picked) >= k:
            break
    return sorted(picked)


def rank_multi_stop(dgrp: pd.DataFrame, start_compound: str, n_laps: int,
                     field_pace: pd.Series, degradation: dict, pit_loss: float,
                     neutral: dict | None = None):
    """Try 1, 2, and 3 candidate stops (from select_candidate_stops) and
    keep whichever gives the lowest predicted total race time. Returns
    (total_seconds, stop_laps) for the winning stop-count, or (None, [])
    if nothing produced a legal (>=2-compound) strategy."""
    best_s, best_laps = None, []
    for k in range(1, MAX_STOPS + 1):
        candidate_laps = select_candidate_stops(dgrp, n_laps, k)
        if len(candidate_laps) < k:
            break  # ran out of well-separated candidates -- more stops won't help
        s = replay_strategy(candidate_laps, start_compound, n_laps, field_pace, degradation, pit_loss, neutral)
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


def replay_strategy(pit_in_laps: list, start_compound: str, n_laps: int,
                     field_pace: pd.Series, degradation: dict, pit_loss: float,
                     neutral: dict | None = None, trace: list | None = None):
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
    satisfied yet.

    pit_in_laps are the classifier's laps: "pits at the end of lap L", so
    L is the in-lap and new tyres start on L + 1. (They were once used as
    the first lap on new tyres, replaying every classifier strategy a lap
    early.) With `neutral` (02's neutralisation()), safety cars are priced
    as in 04's DP and actual-strategy replay: a stop at the end of an SC or
    VSC lap costs that kind's price, and the deficit to the field
    accumulated before each full safety car shrinks by beta. `trace`, if
    given, collects the compound fitted for each stint."""
    def expected_pace(lap, compound, age):
        pace_lookup = degradation.get(compound)
        fp = field_pace.get(lap)
        if pace_lookup is None or fp is None or pd.isna(fp):
            return None
        rel_pace = pace_lookup.get((int(lap), int(age)))
        if rel_pace is None:
            return None
        return fp + rel_pace

    stops_sorted = sorted(int(l) + 1 for l in pit_in_laps)  # out-laps

    def stint_end(lap):
        later = [s for s in stops_sorted if s > lap]
        return (later[0] - 1) if later else n_laps

    def best_compound(used: set, lap: int):
        # Only force an unused compound while the two-compound rule is still
        # unmet; after that, any compound (same one included) is fair game,
        # matching what the DP baseline is allowed to do.
        candidates = list(degradation)
        if len(used) < 2:
            candidates = [c for c in candidates if c not in used] or candidates
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
    if trace is not None:
        trace.append(compound)
    age = 0
    total = 0.0
    stop_set = set(stops_sorted)
    deficit, last_p = 0.0, -1
    for lap in range(1, n_laps + 1):
        lap_start_total = total
        if lap in stop_set:
            compound = best_compound(used, lap)
            if compound is None:
                return None
            used.add(compound)
            if trace is not None:
                trace.append(compound)
            age = 0
            price, last_p = dpl.stop_price(neutral, lap - 1, pit_loss, last_p)
            total += price
        age += 1
        ep = expected_pace(lap, compound, age)
        if ep is None:
            return None
        total += ep
        if neutral is not None:
            deficit += total - lap_start_total - float(field_pace.get(lap, 0.0))
            if lap in neutral['compress_after']:
                total += (neutral['beta'] - 1.0) * deficit
                deficit *= neutral['beta']
    if len(used) < 2:
        return None  # illegal strategy -- never satisfied the two-compound rule
    return total


def follow_default_policy(plan: dict, start_compound: str, n_laps: int, field_pace: pd.Series,
                          degradation: dict, pit_loss: float, neutral: dict, decide,
                          react_to_neutral: bool = True, shift_in_window: bool = True,
                          green_value=None) -> list:
    """The live policy (14's PlanFollower: follow the default plan, leave it
    only on a clear cost-model margin or for an SC/VSC stop), replayed lap
    by lap with only what's known at each lap. `decide` is 10's
    pit_decision_model for this race and team; one cheap stop per
    neutralisation. Returns the policy's stop in-laps.

    green_value, if given, replaces the cost model for green-flag window
    calls: green_value(lap, done, pending) -> how much better pitting now
    is than the best later lap in the window, in 15's expected-points
    objective (the position-aware policy), with 15's margin."""
    compounds = list(degradation)
    policy = ds.PlanFollower(plan, n_laps)
    last_cheap_period = -1

    def state(lap):
        """Compound, tyre age and compounds-used mask on `lap`, with the
        stops so far plus the rest of the plan (which sets compound choice)."""
        tentative = sorted(policy.done + [st for st, _ in policy.pending])
        trace = []
        if replay_strategy(tentative, start_compound, n_laps, field_pace, degradation, pit_loss,
                           None, trace) is None:
            return None
        k = sum(1 for st in tentative if st < lap)
        last = max([st for st in tentative if st < lap], default=0)
        mask = 0
        for c in trace[:k + 1]:
            mask |= 1 << compounds.index(c)
        return trace[k], lap - last, mask

    def diff(lap, kind):
        st = state(lap)
        d = decide(lap, *st, kind, with_no_sc=False) if st else None
        return d['diff_s'] if d else None

    for lap in range(3, n_laps - 1):
        kind = neutral['kinds'].get(lap) if neutral else None
        if kind and react_to_neutral and neutral['period'][lap] != last_cheap_period:
            if policy.neutral_call(diff(lap, kind)):
                policy.record_stop(lap)
                last_cheap_period = neutral['period'][lap]
                continue
        if green_value is not None:
            d = green_value(lap, list(policy.done), list(policy.pending)) if policy.wants_evaluation(lap) else None
            call = policy.green_call(lap, d, margin=pos.POSITION_MARGIN, unit=' pts')
        else:
            d = diff(lap, None) if shift_in_window and policy.wants_evaluation(lap) else None
            call = policy.green_call(lap, d)
        if call and call[0]:
            policy.record_stop(lap)
    return sorted(policy.done)


def follow_strategy_policy(plan: dict, start_compound: str, n_laps: int, field_pace: pd.Series,
                           degradation: dict, pit_loss: float, neutral: dict, decide, strategy_decider) -> list:
    """The strategy-level policy, replayed lap by lap: start from the
    default plan, re-plan the rest of the race with 15's StrategyDecider
    (expected points vs the rival against a forecast field; stop count and
    timing free) every REPLAN_EVERY laps and near each stop, and stop when
    the current plan says so. SC/VSC calls as in follow_default_policy (the
    time-based expected-cost model, one cheap stop per neutralisation),
    after which the remaining stops are re-spaced."""
    compounds = list(degradation)
    future = [int(st) for st in plan['stops'] if 2 < st < n_laps - 1]
    done, comps_so_far, last_cheap_period = [], [start_compound], -1

    def pit(lap):
        nonlocal future
        stops = sorted(done + [lap] + [st for st in future if st > lap])
        comps = pos.choose_compounds(degradation, comps_so_far, stops, n_laps)
        done.append(lap)
        comps_so_far.append(comps[len(comps_so_far)] if comps else comps_so_far[-1])

    def sc_diff(lap, kind):
        tentative = sorted(done + future)
        trace = []
        if replay_strategy(tentative, start_compound, n_laps, field_pace, degradation, pit_loss, None, trace) is None:
            return None
        k = sum(1 for st in tentative if st < lap)
        last = max([st for st in tentative if st < lap], default=0)
        mask = 0
        for c in trace[:k + 1]:
            mask |= 1 << compounds.index(c)
        d = decide(lap, trace[k], lap - last, mask, kind, with_no_sc=False)
        return d['diff_s'] if d else None

    for lap in range(3, n_laps - 1):
        kind = neutral['kinds'].get(lap) if neutral else None
        if kind and neutral['period'][lap] != last_cheap_period:
            if ds.PlanFollower.neutral_call(sc_diff(lap, kind)):
                # As 14's record_stop: replaces the next stop past the
                # stint's halfway point, then the rest are re-spaced.
                prev = done[-1] if done else 0
                rest = future[1:] if future and lap >= (prev + future[0]) / 2 else future
                pit(lap)
                last_cheap_period = neutral['period'][lap]
                k = len(rest)
                future = [int(round(lap + (i + 1) * (n_laps - lap) / (k + 1))) for i in range(k)]
                continue
        if kind is None and ((lap - 3) % pos.REPLAN_EVERY == 0 or (future and future[0] - lap <= pos.NEAR_STOP)):
            future = strategy_decider(lap, list(done), list(comps_so_far), list(future))
        future = [st for st in future if st >= lap]
        if future and future[0] == lap:
            pit(lap)
            future = future[1:]
    return sorted(done)


DELTA_COLS = {
    'Real teams': 'actual_delta_vs_optimal',
    'Classifier, threshold': 'classifier_delta_vs_optimal',
    'Classifier, single-stop ranking': 'ranked_delta_vs_optimal',
    'Classifier, multi-stop ranking': 'ranked_multi_delta_vs_optimal',
    'Default plan (pre-race)': 'default_delta_vs_optimal',
    'Default plan, history only': 'history_default_delta_vs_optimal',
    'Live policy (default + SC/VSC)': 'policy_sc_delta_vs_optimal',
    'Live policy (+ window shifts)': 'policy_delta_vs_optimal',
}
N_BOOTSTRAP = 2000


def run_backtest(pred: pd.DataFrame, dp: pd.DataFrame, deg_df: pd.DataFrame, ml_bundle: dict) -> pd.DataFrame:
    """One row per (race, driver): actual, classifier-driven and ranked
    strategies replayed through the pace model, each against the DP
    hindsight optimum over that driver's distance."""
    races = sorted(set(pred['race'].unique()) & set(dp['race'].unique()))
    beta = dpl.sc_gap_compression()
    rows = []
    zero_stop_drivers = 0
    if POSITION_POLICY or STRATEGY_POLICY:
        pass_events = pd.read_csv(pos.EVENTS_FILE)
        pos.FOLLOW_GAP_S = pos.follow_gap(pass_events)
        pass_models = {}
        rivals = {race: g.set_index('driver')['rival_driver'].to_dict()
                  for race, g in pd.read_csv('data/championship_rivals.csv').groupby('race')}
    for race_id in races:
        laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
        laps = fuel_correct(laps)
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
        # 14's default plan: Pirelli's prediction or what finishers did here
        # before, known before the race starts.
        plan = ds.default_plan(race_id, int(laps['LapNumber'].max()))
        # History alone too, to measure what Pirelli's prediction adds where we have one.
        history = ds.history_plan(race_id, int(laps['LapNumber'].max()))
        has_pirelli = ds.pirelli_plan(race_id, int(laps['LapNumber'].max())) is not None
        # The live policy decides with what was known before the race: SC/VSC
        # rates and gap compression from earlier races only.
        hazard_by_lap = lr.neutral_hazard_by_lap(lr.build_neutral_hazard(before_race=race_id),
                                                 int(laps['LapNumber'].max()))
        beta_as_of = dpl.sc_gap_compression(before_race=race_id)
        decision_models = {}
        if POSITION_POLICY or STRATEGY_POLICY:
            season = race_id[:4]
            if season not in pass_models:
                pass_models[season] = pos.fit_pass_model(pass_events, before_race=f"{season}_r00")
            pos_replay = pos.RaceReplay(race_id, degradation)
            pit_loss_by_driver = dp_race['team_pit_loss_s'].astype(float).to_dict()
            rivals_here = rivals.get(race_id, {})
            forecast_cache = {}

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
            neutral = dpl.neutralisation(laps, float(dp_row['team_sc_pit_loss_s']),
                                         float(dp_row['team_vsc_pit_loss_s']), beta)
            # Same distance the DP baseline and actual replay were scored
            # over: a lapped driver's own lap count, not the race's.
            n_laps = int(dp_row['driver_n_laps'])

            start_compound = laps.loc[
                (laps['Driver'] == driver) & (laps['LapNumber'] == 1), 'Compound']
            if start_compound.empty:
                continue
            start_compound = start_compound.iloc[0]

            window = candidate_window(dgrp, n_laps)
            pit_laps = cluster_pit_events(window.loc[window['pred'] == 1, 'lap'].tolist())
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
                if window.empty:
                    continue
                best_lap = window.loc[window['pred_proba'].idxmax(), 'lap']
                pit_laps = [int(best_lap)]

            classifier_s = replay_strategy(pit_laps, start_compound, n_laps,
                                            field_pace, degradation, pit_loss, neutral)

            ranked_laps = argmax_single_stop(dgrp, n_laps)
            ranked_s = (replay_strategy(ranked_laps, start_compound, n_laps,
                                         field_pace, degradation, pit_loss, neutral)
                        if ranked_laps else None)

            ranked_multi_s, ranked_multi_laps = rank_multi_stop(
                dgrp, start_compound, n_laps, field_pace, degradation, pit_loss, neutral)

            def replay_plan(p):
                stop_laps = [l for l in p['stops'] if 2 < l < n_laps - 1] if p is not None else []
                s = replay_strategy(stop_laps, start_compound, n_laps, field_pace, degradation,
                                    pit_loss, neutral) if stop_laps else None
                return s, (len(stop_laps) if p is not None else None)
            default_s, default_n = replay_plan(plan)
            history_s, _ = replay_plan(history)

            def strategy_str(stop_laps):
                """'in-laps|compounds', e.g. '18;41|MEDIUM;HARD;HARD' -- for 15's
                position-scored backtest."""
                trace = []
                if not stop_laps or replay_strategy(stop_laps, start_compound, n_laps, field_pace, degradation,
                                                    pit_loss, neutral, trace) is None:
                    return None
                return ';'.join(map(str, sorted(stop_laps))) + '|' + ';'.join(trace)

            policy_s = policy_sc_s = policy_n = policy_strategy = default_strategy = policy_pos_strategy = None
            policy_strat_strategy = None
            if plan is not None and start_compound in degradation:
                prices = {'SC': float(dp_row['team_sc_pit_loss_s']), 'VSC': float(dp_row['team_vsc_pit_loss_s'])}
                key = (n_laps, pit_loss, prices['SC'], prices['VSC'])
                if key not in decision_models:
                    decision_models[key] = lr.pit_decision_model(
                        n_laps, field_pace, degradation, list(degradation), 2, pit_loss, prices,
                        hazard_by_lap, beta_as_of)
                decide = decision_models[key]
                policy_laps = follow_default_policy(plan, start_compound, n_laps, field_pace, degradation,
                                                    pit_loss, neutral, decide)
                policy_sc_laps = follow_default_policy(plan, start_compound, n_laps, field_pace, degradation,
                                                       pit_loss, neutral, decide, shift_in_window=False)
                policy_s = replay_strategy(policy_laps, start_compound, n_laps, field_pace, degradation,
                                           pit_loss, neutral) if policy_laps else None
                policy_sc_s = replay_strategy(policy_sc_laps, start_compound, n_laps, field_pace, degradation,
                                              pit_loss, neutral) if policy_sc_laps else None
                policy_n = len(policy_laps)
                if POSITION_POLICY and pos_replay.ok:
                    decider = pos.PositionDecider(pos_replay, driver, plan, start_compound, pit_loss, neutral,
                                                  pass_models[race_id[:4]], rivals_here.get(driver),
                                                  pit_loss_by_driver, forecast_cache)
                    policy_pos_laps = follow_default_policy(plan, start_compound, n_laps, field_pace, degradation,
                                                            pit_loss, neutral, decide, green_value=decider)
                    policy_pos_strategy = strategy_str(policy_pos_laps)
                if STRATEGY_POLICY and pos_replay.ok:
                    sdecider = pos.StrategyDecider(pos_replay, driver, plan, pit_loss, neutral, pass_models[race_id[:4]],
                                                   rivals_here.get(driver), pit_loss_by_driver, forecast_cache)
                    policy_strat_strategy = strategy_str(follow_strategy_policy(
                        plan, start_compound, n_laps, field_pace, degradation, pit_loss, neutral, decide, sdecider))
                policy_strategy = strategy_str(policy_laps)
                default_strategy = strategy_str([l for l in plan['stops'] if 2 < l < n_laps - 1])

            rows.append({
                'race': race_id, 'event': dp_row['event'], 'driver': driver, 'n_laps': n_laps,
                'default_s': default_s, 'default_n_stops': default_n,
                'default_delta_vs_optimal': (default_s - dp_row['dp_optimal_s']) if default_s else None,
                'policy_s': policy_s, 'policy_n_stops': policy_n,
                'policy_strategy': policy_strategy, 'default_strategy': default_strategy,
                'policy_pos_strategy': policy_pos_strategy, 'policy_strat_strategy': policy_strat_strategy,
                'policy_delta_vs_optimal': (policy_s - dp_row['dp_optimal_s']) if policy_s else None,
                'policy_sc_s': policy_sc_s,
                'policy_sc_delta_vs_optimal': (policy_sc_s - dp_row['dp_optimal_s']) if policy_sc_s else None,
                'has_pirelli': has_pirelli, 'history_default_s': history_s,
                'history_default_delta_vs_optimal': (history_s - dp_row['dp_optimal_s']) if history_s else None,
                'race_had_sc': bool(neutral['kinds']),
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
    out.attrs['zero_stop_drivers'] = zero_stop_drivers
    return out


def report(label, d, delta_col='classifier_delta_vs_optimal'):
    if d.empty:
        print(f"  {label:34s} n=0")
        return
    beat = (d[delta_col] < d['actual_delta_vs_optimal']).sum()
    print(f"  {label:34s} n={len(d):<4d} median {d[delta_col].median():+7.1f}s vs. optimal, "
          f"actual median {d['actual_delta_vs_optimal'].median():+7.1f}s, beats actual {beat}/{len(d)}")


def bootstrap_medians(out: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Median seconds lost vs. optimal per strategy, with 95% intervals from
    resampling whole races (driver-races within a race aren't independent),
    plus the paired median difference vs. the real teams. Quantifies how
    much of each gap is noise, in the spirit of Brill et al. (2023)."""
    rng = np.random.default_rng(seed)
    by_race = {r: g for r, g in out.groupby('race')}
    race_ids = np.array(list(by_race))
    samples = {k: [] for k in DELTA_COLS}
    diffs = {k: [] for k in DELTA_COLS if k != 'Real teams'}
    for _ in range(N_BOOTSTRAP):
        b = pd.concat([by_race[r] for r in rng.choice(race_ids, len(race_ids))], ignore_index=True)
        for k, col in DELTA_COLS.items():
            samples[k].append(b[col].median())
            if k in diffs:
                both = b.dropna(subset=[col, 'actual_delta_vs_optimal'])
                diffs[k].append((both[col] - both['actual_delta_vs_optimal']).median())
    rows = []
    for k, col in DELTA_COLS.items():
        lo, hi = np.nanpercentile(samples[k], [2.5, 97.5])
        row = {'strategy': k, 'n': int(out[col].notna().sum()), 'median_s': out[col].median(),
               'ci_low': lo, 'ci_high': hi}
        if k in diffs:
            both = out.dropna(subset=[col, 'actual_delta_vs_optimal'])
            row['vs_real_median_diff_s'] = (both[col] - both['actual_delta_vs_optimal']).median()
            row['diff_ci_low'], row['diff_ci_high'] = np.nanpercentile(diffs[k], [2.5, 97.5])
            row['beats_real_share'] = (both[col] < both['actual_delta_vs_optimal']).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def pirelli_vs_history(out: pd.DataFrame):
    """Races with a Pirelli prediction: does it make a better default plan
    than history? Per race, median over drivers of (plan - real team), then
    per-race paired Pirelli - history."""
    d = out[out['has_pirelli']].dropna(subset=['default_s', 'history_default_s', 'actual_s'])
    if d.empty:
        return
    per_race = d.groupby('race')[d.columns].apply(lambda g: pd.Series({
        'event': g['event'].iloc[0], 'drivers': len(g),
        'pirelli_vs_real': (g['default_s'] - g['actual_s']).median(),
        'history_vs_real': (g['history_default_s'] - g['actual_s']).median(),
        'pirelli_minus_history': (g['default_s'] - g['history_default_s']).median()}))
    print(f"\n=== Default plan: Pirelli's prediction vs history ({len(per_race)} races with a Pirelli entry) ===")
    print("Median seconds per driver; negative pirelli_minus_history = Pirelli's plan is cheaper")
    print(per_race.round(1).to_string())
    rng = np.random.default_rng(0)
    diffs = per_race['pirelli_minus_history'].to_numpy(dtype=float)
    boot = [np.median(rng.choice(diffs, len(diffs))) for _ in range(N_BOOTSTRAP)]
    print(f"Median over races: Pirelli - history = {np.median(diffs):+.1f} s "
          f"(95% CI {np.percentile(boot, 2.5):+.1f} to {np.percentile(boot, 97.5):+.1f}); "
          f"Pirelli cheaper in {(diffs < 0).sum()}/{len(diffs)} races")


def main():
    pred = pd.read_csv('data/trigger_classifier_test_predictions.csv')
    dp = pd.read_csv('data/dp_baseline_backtest.csv')
    deg_df = pd.read_csv('data/degradation_summary.csv')
    ml_bundle = dm.load_degradation_model()

    out = run_backtest(pred, dp, deg_df, ml_bundle)
    zero_stop_drivers = out.attrs['zero_stop_drivers']
    print(f"3-way backtest over {out['race'].nunique()} races (pinned test-set races the DP baseline could "
          f"also solve)\n")
    valid = out.dropna(subset=['classifier_s'])
    fallback = valid[valid['classifier_used_fallback']]
    fired = valid[~valid['classifier_used_fallback']]
    print(f"{len(out)} driver-race rows; {zero_stop_drivers} needed the highest-probability-lap "
          f"fallback (nothing crossed threshold)")
    print(f"{len(valid)} rows with a legal (>=2-compound) classifier-driven strategy "
          f"({len(fired)} threshold-fired, {len(fallback)} fallback)\n")

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

    print("\nBootstrap over races (95% intervals; 'vs real' = paired median difference, negative = better than the teams):")
    print(bootstrap_medians(out).round(3).to_string(index=False))
    pirelli_vs_history(out)
    out.to_csv('data/backtest_3way.csv', index=False)
    print(f"\nSaved data/backtest_3way.csv ({len(out)} rows)")

    # Rolling-origin backtest: every season 2022-2026 replayed with
    # classifier predictions from a model trained only on earlier races
    # (05's rolling_origin_predictions), ~100 races instead of ~15.
    rolling_path = Path('data/trigger_classifier_rolling_predictions.csv')
    if rolling_path.exists():
        rolling_pred = pd.read_csv(rolling_path)
        roll = run_backtest(rolling_pred, dp, deg_df, ml_bundle)
        roll['season'] = roll['race'].str[:4].astype(int)
        print(f"\n=== Rolling-origin backtest: {roll['race'].nunique()} races, {len(roll)} driver-races ===")
        per_season = roll.groupby('season').agg(
            races=('race', 'nunique'), driver_races=('driver', 'size'),
            **{k: (col, 'median') for k, col in DELTA_COLS.items()})
        print("Median seconds lost vs. DP hindsight-optimal, per season:")
        print(per_season.round(1).to_string())
        print("\nPooled over all seasons, bootstrap over races (95% intervals):")
        print(bootstrap_medians(roll).round(3).to_string(index=False))
        # Robustness: the pace model prices green-flag strategy differences
        # at the right size (13_strategy_cost_validation.py, slope ~1.0) but
        # not strategies around safety cars, where field bunching -- which a
        # per-car lap-time model can't represent -- dominates (slope ~0.4-0.6
        # under either SC pricing). So also report races with no SC/VSC.
        green_only = roll[~roll['race_had_sc']]
        print(f"\nRobustness -- races with no SC/VSC lap ({green_only['race'].nunique()} races, "
              f"{len(green_only)} driver-races), bootstrap over races:")
        print(bootstrap_medians(green_only).round(3).to_string(index=False))
        roll.to_csv('data/backtest_3way_rolling.csv', index=False)
        print(f"\nSaved data/backtest_3way_rolling.csv ({len(roll)} rows)")


if __name__ == '__main__':
    main()
