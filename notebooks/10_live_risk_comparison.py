"""
Phase 6 extension -- live risk/reward comparison for the "pit now or wait"
decision, prompted directly by the MAG/PER investigation in the streaming
consumer: a driver on a long first stint, tempted by a cheap SC/VSC
pit-loss window to switch from a planned 1-stop to a 2-stop. Real teams
face exactly this dilemma, and it changes track position, not just lap
time -- this tool makes the time tradeoff and the *uncertainty* in it
explicit, rather than a single point prediction.

The uncertainty is future SC/VSC deployments, priced from history rather
than a hand-built probability model: build_neutral_hazard() measures, across
every earlier race, the chance a full SC and a VSC are deployed on any given lap at
each stage of a race. compare_pit_now_vs_wait() then solves the same
compound/tyre-age DP as 04_dp_baseline.py from the current state, but by
expected cost: every later lap carries its historical chance of a full SC
and of a VSC, under which a stop that lap costs that kind's (cheaper) pit
loss, and the plan re-optimizes from whatever actually happens. A full SC
also bunches the field, so time lost before one partly washes out -- the
same gap compression the offline tools apply to real safety cars (02's
neutralisation), here in expectation. So
"wait for a cheap window" is only worth what history says it's worth --
including the risk that a late SC/VSC comes too late to help, or that
the tyres can't last that long. This replaced an earlier scenario
comparison whose "SC comes again" case assumed the next SC/VSC would land
on exactly the lap the driver wanted, which made waiting look ~5-7s better
almost every time, even under an actual SC. (04's hindsight-optimal
baseline stays deterministic -- it replays known history, so there's no
uncertainty to price there.)

Run with: python notebooks/10_live_risk_comparison.py <race_id> <lap> [driver ...]
  e.g.    python notebooks/10_live_risk_comparison.py 2023_r08 12 MAG PER

Both use 12_as_of_context.py's models and constants, built only from
races before the one being decided. 09_streaming_consumer.py also imports this module directly (dynamic
import, same pattern as everywhere else in this project) to run this same
pit-now-vs-wait comparison live, off only the laps observed so far in the
stream, whenever its peak detector flags a candidate lap. main() runs the
identical live-safe comparison at a chosen lap (field pace from laps up to
that lap only, via causal_field_pace_forecast()), so the two always agree.
"""
import sys
import functools
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path

_spec2 = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(dpl)

BRONZE_DIR = Path('data/bronze')
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
INF = float('inf')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
WET_COMPOUNDS = {'INTERMEDIATE', 'WET'}
# Wet-weather strategy (when to cross over between wet and dry tyres) is
# out of scope: it's driven by rainfall and track drying, not tyre wear
# and pit loss. A lap counts as wet conditions if anyone ran wet tyres
# within this many laps of it, so a drying track with cars still on
# intermediates isn't treated as dry yet.
WET_LOOKBACK_LAPS = 3

SC_HAZARD_BINS = 10  # race-progress bands the historical SC/VSC rate is measured over
# Expected-time margins smaller than this are reported as too close to call:
# the pace model's own error (~0.7s/lap MAE) swamps them.
CLOSE_CALL_S = 1.0


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


@functools.lru_cache(maxsize=None)
def neutralisation_deployments(race_id: str) -> list:
    """(lap, 'SC' or 'VSC') for every safety car / virtual safety car
    deployed in a race (race-control DEPLOYED messages)."""
    rc_path = BRONZE_DIR / f"{race_id}_race_control.parquet"
    if not rc_path.exists():
        return []
    rc = pd.read_parquet(rc_path)
    deployed = rc[(rc['Category'] == 'SafetyCar') & (rc['Status'] == 'DEPLOYED')].dropna(subset=['Lap'])
    return [(int(r['Lap']), 'VSC' if dpl.is_vsc_message(r['Message']) else 'SC') for _, r in deployed.iterrows()]


@functools.lru_cache(maxsize=None)
def _race_n_laps(race_id: str) -> int:
    return int(pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet", columns=['LapNumber'])['LapNumber'].max())


def _progress_bin(laps: np.ndarray, n_laps: int) -> np.ndarray:
    return np.minimum(((laps - 1) * SC_HAZARD_BINS) // n_laps, SC_HAZARD_BINS - 1)


def build_neutral_hazard(before_race: str | None = None) -> dict:
    """{'SC': rates, 'VSC': rates}: per-lap chance of a full / virtual
    safety car being deployed, by race-progress band -- deployments in each
    band / race-laps run in that band, pooled over every race strictly
    before `before_race` (race ids sort chronologically), so neither it nor
    any later race informs its decision. SC and VSC are kept apart because
    they differ in what they do to strategy (stop price, gap compression),
    as in Pitwall (Santillana 2026)."""
    deployments = {'SC': np.zeros(SC_HAZARD_BINS), 'VSC': np.zeros(SC_HAZARD_BINS)}
    exposure = np.zeros(SC_HAZARD_BINS)
    for f in sorted(BRONZE_DIR.glob('*_laps.parquet')):
        race_id = f.stem.replace('_laps', '')
        if before_race is not None and race_id >= before_race:
            continue
        n_laps = _race_n_laps(race_id)
        exposure += np.bincount(_progress_bin(np.arange(1, n_laps + 1), n_laps), minlength=SC_HAZARD_BINS)
        for kind in deployments:
            laps = np.array([l for l, k in neutralisation_deployments(race_id) if k == kind and 1 <= l <= n_laps],
                            dtype=int)
            if len(laps):
                deployments[kind] += np.bincount(_progress_bin(laps, n_laps), minlength=SC_HAZARD_BINS)
    return {k: v / np.maximum(exposure, 1) for k, v in deployments.items()}


def neutral_hazard_by_lap(hazard: dict, n_laps: int) -> dict:
    """{lap: {'SC': p, 'VSC': p}}."""
    laps = np.arange(1, n_laps + 1)
    bins = _progress_bin(laps, n_laps)
    return {int(l): {k: float(hazard[k][b]) for k in hazard} for l, b in zip(laps, bins)}


def causal_field_pace_forecast(laps_so_far: pd.DataFrame, n_laps: int) -> pd.Series:
    """Live-safe field-pace baseline, for use by the streaming consumer
    (09_streaming_consumer.py) and by main().
    Reading the real historical field pace for the whole race would
    include laps that haven't happened yet from a live decision's point
    of view, silently baking in whether a real future SC/rain event
    happened.

    Laps already seen use their real observed field pace. Laps not yet
    seen are held flat at the recent green-flag median -- i.e. "assume
    conditions continue as they've been," which is the honest, causal
    definition of "no more incident." The *chance* of a future incident is
    priced separately, via build_neutral_hazard()'s historical per-lap SC/VSC
    rate and the cheaper stop it allows -- not by projecting a specific
    future pace-spike shape, which a live system can't know in advance."""
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


def wet_conditions(laps_so_far: pd.DataFrame, lap: int) -> bool:
    """True if anyone ran intermediate/wet tyres within WET_LOOKBACK_LAPS
    laps up to and including `lap` -- the cost model only covers dry running."""
    recent = laps_so_far[(laps_so_far['LapNumber'] > lap - WET_LOOKBACK_LAPS) & (laps_so_far['LapNumber'] <= lap)]
    return bool(recent['Compound'].isin(WET_COMPOUNDS).any())


def compound_rule_state(laps_so_far: pd.DataFrame, driver: str, lap: int, compounds: list) -> tuple:
    """(start_mask, min_total_compounds) for the two-compound rule, from
    what this driver has actually run so far -- not just the current set,
    or a driver already on their second compound would be forced into
    another stop. The rule doesn't apply at all to a driver who has used
    intermediate/wet tyres (sporting regulations), or once two dry
    compounds have been used (including one this race's model doesn't
    cover, which the mask can't represent)."""
    used = set(laps_so_far.loc[(laps_so_far['Driver'] == driver) & (laps_so_far['LapNumber'] <= lap),
                               'Compound'].dropna())
    mask = 0
    for c in used & set(compounds):
        mask |= 1 << compounds.index(c)
    if used & WET_COMPOUNDS or len(used & set(DRY_COMPOUNDS)) >= 2:
        return mask, 1
    return mask, 2


def pit_decision_model(n_laps: int, field_pace: pd.Series, degradation: dict, compounds: list,
                       min_total_compounds: int, pit_loss: float, prices: dict,
                       hazard_by_lap: dict, beta: float):
    """decide(lap, compound, age, start_mask, kind_now) -> the pit-now-vs-
    wait comparison at the end of `lap` (see compare_pit_now_vs_wait).

    Built once per (race, pit prices) and reused for every decision: the
    SC weights are a product over laps from the flag backwards, so they
    don't depend on the decision lap, and the value function's memo is
    keyed by (lap, compound, age, compounds used) -- a whole race of
    decisions for every car in a team costs about one DP solve.

    Safety cars, as in the offline tools (02's neutralisation), but in
    expectation since the future is unknown:
      - each later lap carries its historical chance of a full SC and of a
        VSC being deployed; if one is, a stop at the end of that lap costs
        that kind's price (prices['SC'] / prices['VSC']) instead of pit_loss.
        Pitting now costs the price of whatever is actually out now
        (kind_now), or pit_loss under green.
      - a full SC bunches the field: the deficit to the field accumulated
        before it shrinks by beta. Deployments don't depend on our
        decisions, so the time lost on lap t is worth, in expectation,
        weight(t) = prod over later laps k of (1 - P(SC deployed on k) x
        (1 - beta)) -- weighting each lap's cost keeps the DP exact with no
        extra state. So time lost now partly washes out if an SC comes
        later, and a plan that waits for a cheap SC stop is only worth what
        history says it is. (Published expected-value and simulation
        strategy models found in the literature price cheaper SC/VSC stops
        but not this bunching.)"""
    def pace(l, c, a):
        rel = degradation[c].get((int(l), int(a)))
        fp = field_pace.get(l)
        if rel is None or fp is None or pd.isna(fp):
            return None
        return float(rel)  # relative to the field; field pace is added back once at the end

    def bit(c):
        return 1 << compounds.index(c)

    def weights(hazard: dict) -> dict:
        w, running = {}, 1.0
        for l in range(n_laps, 0, -1):
            w[l] = running                       # SC deployments after lap l
            running *= 1.0 - hazard.get(l, {}).get('SC', 0.0) * (1.0 - beta)
        return w

    def value_function(hazard: dict, w: dict):
        memo = {}

        def V(l, c, a, mask):
            """Expected weighted time relative to the field for laps l..n_laps."""
            if l > n_laps:
                return 0.0 if bin(mask).count('1') >= min_total_compounds else INF
            key = (l, c, a, mask)
            if key in memo:
                return memo[key]
            p = pace(l, c, a + 1)
            stay = w[l] * p + V(l + 1, c, a + 1, mask) if p is not None else INF
            switch = INF   # best fresh-set stint from here, excluding the pit price
            for c2 in compounds:
                p2 = pace(l, c2, 1)
                if p2 is not None:
                    switch = min(switch, w[l] * p2 + V(l + 1, c2, 1, mask | bit(c2)))
            h = hazard.get(l - 1, {})   # a neutralisation on lap l-1 makes a stop at its end cheap
            h_sc, h_vsc = h.get('SC', 0.0), h.get('VSC', 0.0)
            green = min(stay, w[l] * pit_loss + switch)
            v = ((1 - h_sc - h_vsc) * green
                 + (h_sc * min(stay, w[l] * prices['SC'] + switch) if h_sc else 0.0)
                 + (h_vsc * min(stay, w[l] * prices['VSC'] + switch) if h_vsc else 0.0))
            memo[key] = v
            return v
        return V, w

    with_sc = value_function(hazard_by_lap, weights(hazard_by_lap))
    no_sc = value_function({}, weights({}))

    def options(model, out_lap, compound, age, start_mask, now_price):
        V, w = model
        p = pace(out_lap, compound, age + 1)
        wait = w[out_lap] * p + V(out_lap + 1, compound, age + 1, start_mask) if p is not None else INF
        pit = INF
        for c2 in compounds:
            p2 = pace(out_lap, c2, 1)
            if p2 is not None:
                pit = min(pit, w[out_lap] * (now_price + p2) + V(out_lap + 1, c2, 1, start_mask | bit(c2)))
        return pit, wait

    def decide(lap: int, compound: str, age: int, start_mask: int, kind_now: str | None,
               with_no_sc: bool = True) -> dict | None:
        out_lap = lap + 1
        if out_lap > n_laps or compound not in compounds:
            return None
        now_price = prices[kind_now] if kind_now else pit_loss
        field_total = float(sum(field_pace.get(l, 0.0) for l in range(out_lap, n_laps + 1)))
        pit_now, wait = options(with_sc, out_lap, compound, age, start_mask, now_price)
        pit_now_no_sc, wait_no_sc = (options(no_sc, out_lap, compound, age, start_mask, now_price)
                                     if with_no_sc else (pit_now, wait))
        if INF in (pit_now, wait, pit_now_no_sc, wait_no_sc):
            return None
        pit_now, wait = field_total + pit_now, field_total + wait
        p_none = float(np.prod([1 - sum(hazard_by_lap.get(l, {}).values()) for l in range(out_lap, n_laps)]))
        # Plain Python types: 09 emits this dict as JSON, which rejects numpy bools.
        return {'pit_now_s': float(pit_now), 'wait_s': float(wait),
                'pit_now_no_sc_s': float(field_total + pit_now_no_sc), 'wait_no_sc_s': float(field_total + wait_no_sc),
                'p_sc_before_end': 1 - p_none, 'diff_s': float(wait - pit_now),
                'verdict': 'pit NOW' if pit_now < wait else 'WAIT',
                'too_close': bool(abs(wait - pit_now) < CLOSE_CALL_S),
                'sc_active_now': kind_now is not None, 'kind_now': kind_now}
    return decide


def compare_pit_now_vs_wait(n_laps: int, field_pace: pd.Series, degradation: dict, compounds: list,
                            lap: int, compound: str, age: int, start_mask: int, min_total_compounds: int,
                            pit_loss: float, prices: dict, kind_now: str | None,
                            hazard_by_lap: dict, beta: float) -> dict | None:
    """Decision at the end of `lap` (`age` = tyre age on it): pit now, so
    lap+1 is the out-lap, or stay out. Both are scored as the expected
    total time from lap+1 to the flag by a stochastic DP over (lap,
    compound, tyre age, compounds used), re-optimising every later lap,
    with safety cars priced in expectation (see pit_decision_model). Also
    returns both costs with no SC/VSC ever (the pure tyre/pit-loss
    tradeoff). None if either option has no legal plan from here."""
    decide = pit_decision_model(n_laps, field_pace, degradation, compounds, min_total_compounds,
                                pit_loss, prices, hazard_by_lap, beta)
    return decide(lap, compound, age, start_mask, kind_now)


def neutral_kind(track_status) -> str | None:
    """'SC', 'VSC' or None from a lap's TrackStatus (red flags excluded)."""
    status = str(track_status)
    if '5' in status or not dpl._is_sc_or_vsc(status):
        return None
    return 'SC' if '4' in status else 'VSC'


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    race_id, decision_lap = sys.argv[1], int(sys.argv[2])
    drivers = sys.argv[3:] or ['MAG', 'PER']

    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    n_laps = int(laps['LapNumber'].max())
    laps = fuel_correct(laps)
    event_name = laps['EventName'].iloc[0]
    laps_so_far = laps[laps['LapNumber'] <= decision_lap]
    field_pace = causal_field_pace_forecast(laps_so_far, n_laps)

    # Same as-of-race models and constants the live consumer uses (imported
    # here, not at the top, because 12_as_of_context.py imports this file).
    spec = importlib.util.spec_from_file_location('as_of_context_lib', Path(__file__).parent / '12_as_of_context.py')
    ac = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    risk = ac.live_context(race_id, event_name, n_laps)['risk']
    if risk is None:
        print(f"{race_id}: no tyre/pit-loss history before this race -- nothing to compare with.")
        sys.exit(1)
    compounds = risk['compounds']
    degradation = ac.live_pace_lookup(risk['deg_bundle'], race_id, n_laps, laps_so_far['Time'].max(), compounds)
    green_pit_loss = risk['green_pit_loss']
    hazard_by_lap = neutral_hazard_by_lap(risk['neutral_hazard'], n_laps)

    print(f"{race_id} ({event_name}), decision at the end of lap {decision_lap}/{n_laps} "
          f"-- models and constants from the {len(ac.prior_races(race_id))} races before this one")
    print(f"Pit loss: green {green_pit_loss:.1f}s, SC {risk['sc_pit_loss']:.1f}s, VSC {risk['vsc_pit_loss']:.1f}s, "
          f"full-SC gap compression {risk['sc_beta']:.2f} "
          f"({risk['pit_loss_source']}) | historical chance per lap from here: "
          f"SC {hazard_by_lap[min(decision_lap + 1, n_laps)]['SC']*100:.1f}%, "
          f"VSC {hazard_by_lap[min(decision_lap + 1, n_laps)]['VSC']*100:.1f}%\n")

    for driver in drivers:
        row = laps_so_far[(laps_so_far['Driver'] == driver) & (laps_so_far['LapNumber'] == decision_lap)]
        if row.empty:
            print(f"--- {driver}: no lap {decision_lap} in this race ---\n")
            continue
        row = row.iloc[0]
        compound = row['Compound']
        if wet_conditions(laps_so_far, decision_lap):
            print(f"--- {driver}: skipped -- wet conditions (intermediate/wet tyres in use within "
                  f"{WET_LOOKBACK_LAPS} laps); wet-weather strategy is out of scope ---\n")
            continue
        if compound not in compounds:
            print(f"--- {driver}: skipped -- {compound} not modeled for this race ---\n")
            continue
        if pd.isna(row['TyreLife']):
            print(f"--- {driver}: skipped -- tyre age unknown on lap {decision_lap} ---\n")
            continue
        age = int(row['TyreLife'])
        kind_now = neutral_kind(row['TrackStatus'])
        mask, min_compounds = compound_rule_state(laps_so_far, driver, decision_lap, compounds)
        pit_loss = green_pit_loss + float(risk['team_adj'].get(row['Team'], 0.0))
        adj = float(risk['team_adj'].get(row['Team'], 0.0))
        prices = {'SC': risk['sc_pit_loss'] + adj, 'VSC': risk['vsc_pit_loss'] + adj}
        r = compare_pit_now_vs_wait(n_laps, field_pace, degradation, compounds, decision_lap, compound, age,
                                    mask, min_compounds, pit_loss, prices, kind_now, hazard_by_lap, risk['sc_beta'])
        print(f"--- {driver}: on {compound}, tyre age {age}, {kind_now + ' out' if kind_now else 'green flag'} ---")
        if r is None:
            print("  (no legal strategy can be solved from this state)\n")
            continue
        print(f"  Pit now:  {r['pit_now_s']:.1f}s expected  ({r['pit_now_no_sc_s']:.1f}s if no SC/VSC comes)")
        print(f"  Stay out: {r['wait_s']:.1f}s expected  ({r['wait_no_sc_s']:.1f}s if no SC/VSC comes)")
        print(f"  P(an SC/VSC before the end) = {r['p_sc_before_end']*100:.0f}%")
        print(f"  --> {r['verdict']}, by {abs(r['diff_s']):.1f}s on expectation"
              f"{' -- too close to call' if r['too_close'] else ''}\n")


if __name__ == '__main__':
    main()
