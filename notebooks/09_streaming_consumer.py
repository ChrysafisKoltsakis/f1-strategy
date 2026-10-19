"""
Phase 6 -- streaming consumer. Reads the replayed race-lap Kafka topic
(published by 08_streaming_producer.py) via Spark Structured Streaming,
maintains live per-driver state, and re-invokes the trained trigger
classifier after each new lap to emit a live pit recommendation. This is
what actually answers Psaltiras's declared future-work chapter -- not a
batch backtest, but a model queried mid-race exactly as a live system
would use it.

Reuses build_race_features()/fuel_correct()/compute_field_pace() from
05_trigger_classifier.py, loaded dynamically (notebook filenames start
with a digit and can't be `import`ed as normal Python modules), so live
and offline feature computation are guaranteed identical rather than a
separately maintained copy that could silently drift from the trained
model's expectations.

Live decision rule: pit windows, a lap ahead. Each lap record arrives
only when the lap ends, so the classifier (05) predicts whether a driver
pits at the end of the NEXT lap from everything known now. An earlier
version detected the peak of "pits at the end of this lap" and flagged it
a lap after the peak -- by which time the car had usually pitted already
(29 of 53 flags in one replay). Now a pit window opens as soon as the
probability crosses the model's threshold (chosen on the last training
season, never on test races), a lap before the stop, and stays open --
re-evaluated every lap -- until the car stops or the probability falls
below half the threshold.

Scope simplification, stated plainly: each micro-batch recomputes
features over ALL laps seen so far in the race (correct live semantics --
only ever uses information available up to that point), rather than
maintaining true incremental state. Fine for a single-race proof-of-
concept (at most ~1500 lap events), not how a production system would do
it at scale.

The classifier only ever answers "does this look like where similar
drivers have historically pitted" -- an imitation-learned behavioral
signal, not a cost or a risk. So every lap a pit window is open, this also
calls 10_live_risk_comparison.py's expected-cost DP (dynamic import, same
reuse pattern as everywhere else): pit at the end of the next lap or stay
out, given only what's been observed so far and the real historical odds
of SC/VSC windows later. Probability, cost, and risk together, computed
causally, lap by lap.

SC/VSC deployments arrive on a second topic (08 publishes them when race
control announced them, mid-lap) and trigger an immediate "pit at the end
of this lap, or stay out?" evaluation for every running driver -- lap
records alone came a lap too late, since a car pitting under the safety
car is already in the pit lane when its lap record arrives. Each
deployment waits until the lap stream has passed its timestamp, then uses
only laps completed before it.

Every model and constant used live comes from 12_as_of_context.py:
trained or measured on races strictly before the one being replayed, so
the replay doesn't flatter itself with knowledge of how that race (or
later ones) went.

Requires: a running Kafka broker (see README), JAVA_HOME set, and 02, 03
and 05 run (12_as_of_context.py trains its as-of models from their
outputs, once per race, cached under data/models/as_of/).

Run with: JAVA_HOME=~/.local/jdk python notebooks/09_streaming_consumer.py <race_id>
"""
import sys
import json
import threading
import importlib.util
from pathlib import Path

import pandas as pd
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC = 'f1-race-laps'
TOPIC_RACE_CONTROL = 'f1-race-control'
BRONZE_DIR = Path('data/bronze')
LIVE_EVENTS_DIR = Path('data/live_events')
# How green-flag strategy calls are made (SC/VSC calls are always the
# time-based expected-cost model):
#   'time'     -- 14's PlanFollower with the time-based cost model: follow the
#                 default plan, move a stop inside its window on a clear margin
#   'window'   -- the same, with 15's position model deciding inside the window
#                 (expected points vs the rival against a forecast field)
#   'strategy' -- 15's StrategyDecider: re-plan the whole rest of the race
#                 (stop count and timing) with the position model
# See the README's position-model section for how each scores offline.
STRATEGY_MODE = 'window'   # 'strategy' is built but its offline evaluation is pending

# Load 05_trigger_classifier.py's feature-building functions directly, so
# live and offline feature computation can never drift apart.
_spec = importlib.util.spec_from_file_location(
    'trigger_classifier_lib', Path(__file__).parent / '05_trigger_classifier.py')
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

# Load 10_live_risk_comparison.py's pit-now-vs-wait DP machinery directly,
# so the live recommendation uses the exact same decision logic already
# validated standalone (e.g. the 2023_r08 MAG/PER case).
_spec3 = importlib.util.spec_from_file_location(
    'live_risk_lib', Path(__file__).parent / '10_live_risk_comparison.py')
lr = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(lr)

# 14's default pre-race plan and the policy that follows it (also replayed
# offline in 06).
_spec5 = importlib.util.spec_from_file_location(
    'default_strategy_lib', Path(__file__).parent / '14_default_strategy.py')
ds = importlib.util.module_from_spec(_spec5)
_spec5.loader.exec_module(ds)

# 15's position model: the green-flag plan calls are decided by expected
# points vs the championship rival, racing a forecast of the field.
_spec6 = importlib.util.spec_from_file_location(
    'position_model_lib', Path(__file__).parent / '15_position_model.py')
pos = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(pos)

_spec4 = importlib.util.spec_from_file_location(
    'as_of_context_lib', Path(__file__).parent / '12_as_of_context.py')
ac = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(ac)

RACE_CONTROL_SCHEMA = StructType([
    StructField('race', StringType()), StructField('kind', StringType()),
    StructField('lap', LongType()), StructField('session_time_s', DoubleType()),
    StructField('message', StringType()),
])

LAP_SCHEMA = StructType([
    StructField('race', StringType()), StructField('driver', StringType()),
    StructField('driver_number', StringType()), StructField('team', StringType()),
    StructField('lap_number', LongType()), StructField('session_time_s', DoubleType()),
    StructField('lap_time_s', DoubleType()), StructField('compound', StringType()),
    StructField('tyre_life', DoubleType()), StructField('position', DoubleType()),
    StructField('track_status', StringType()), StructField('pit_in', BooleanType()),
    StructField('pit_out', BooleanType()),
])


def make_emitter(race_id: str):
    """Every event this consumer produces (lap scores, pit windows, risk
    comparisons) is also written as one JSON line per event to
    data/live_events/{race_id}.jsonl, alongside the existing print()
    output -- so dashboard/server.py can tail the same information a
    terminal user sees, without re-parsing printed strings. Truncates any
    previous run's file for this race_id. Line-buffered so a tailing
    reader sees each event promptly, not once Python's buffer fills."""
    LIVE_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    f = open(LIVE_EVENTS_DIR / f"{race_id}.jsonl", 'w', buffering=1)

    def emit(event: dict):
        f.write(json.dumps(event) + "\n")

    return emit


def load_static_context(race_id: str):
    """Everything knowable before the race starts: rival identities (from
    standings before this race), the race's lap count (known here because
    we're replaying a finished race; a genuinely live system would use the
    scheduled distance), and 12_as_of_context.py's as-of models and
    constants."""
    rivals_df = pd.read_csv('data/championship_rivals.csv')
    rivals = rivals_df[rivals_df['race'] == race_id].set_index('driver')[
        ['rival_driver', 'rival_team']].to_dict('index')
    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    n_laps = int(laps['LapNumber'].max())
    event_name = laps['EventName'].iloc[0]
    print(f"Loading models as of {race_id} (trained on earlier races only; "
          f"first replay of a race trains them, then they're cached)...")
    return rivals, n_laps, event_name, ac.live_context(race_id, event_name, n_laps)


def evaluate_pit_now_vs_wait(driver: str, team: str, lap: int, compound: str, age: int,
                              laps_so_far: pd.DataFrame, n_laps: int, risk_ctx: dict,
                              kind_now: str | None = None) -> dict | None:
    """The actual live recommendation: given only laps observed so far,
    what does pitting at the end of this lap cost versus staying out, in
    expected time, with every later lap carrying its real historical
    chance of an SC/VSC making a stop cheap. Same comparison as
    10_live_risk_comparison.py's standalone tool
    (compare_pit_now_vs_wait()), called from the accumulated live state.
    Returns None if a state can't be solved from here (e.g. too few laps
    remaining).

    Returns raw numbers (a dict) rather than a formatted string, so both
    the terminal print and dashboard/server.py's JSON event can be built
    from the same values without one having to re-parse the other's text.

    "pit now" only gets priced at the cheap SC/VSC rate when SC/VSC is
    ACTUALLY active on this lap in the real data -- checked directly,
    not assumed just because the trigger classifier's probability
    happened to peak. Found by tracing a specific dashboard case (VER,
    2025_r20, lap 36): the classifier peaked on an ordinary green-flag
    pit window (TrackStatus stayed '1' the whole time), but this
    function unconditionally offered the SC discount anyway, making
    "pit now" look artificially cheap and flipping the verdict from the
    correct WAIT to an incorrect pit-NOW. The classifier peak is only
    ever a behavioral trigger for *when to ask the question* -- it says
    nothing about whether a cheap window is actually open right now."""
    compounds = risk_ctx['compounds']
    pit_loss = risk_ctx['green_pit_loss'] + float(risk_ctx['team_adj'].get(team, 0.0))
    field_pace = lr.causal_field_pace_forecast(laps_so_far, n_laps)
    # Tyre-model predictions use the weather observed so far, so they're
    # rebuilt as the race goes on (once per lap, shared across drivers).
    cache = risk_ctx.setdefault('pace_lookup_by_lap', {})
    if lap not in cache:
        cache[lap] = ac.live_pace_lookup(risk_ctx['deg_bundle'], risk_ctx['race_id'], n_laps,
                                         laps_so_far['Time'].max(), compounds)
    degradation = cache[lap]
    current_status = laps_so_far.loc[(laps_so_far['Driver'] == driver) & (laps_so_far['LapNumber'] == lap),
                                      'TrackStatus']
    if kind_now is None:   # given explicitly for a lap still in progress (an SC/VSC just deployed)
        kind_now = lr.neutral_kind(current_status.iloc[-1]) if len(current_status) else None
    adj = float(risk_ctx['team_adj'].get(team, 0.0))
    prices = {'SC': risk_ctx['sc_pit_loss'] + adj, 'VSC': risk_ctx['vsc_pit_loss'] + adj}
    mask, min_compounds = lr.compound_rule_state(laps_so_far, driver, lap, compounds)
    return lr.compare_pit_now_vs_wait(
        n_laps, field_pace, degradation, compounds, lap, compound, age, mask, min_compounds,
        pit_loss, prices, kind_now, lr.neutral_hazard_by_lap(risk_ctx['neutral_hazard'], n_laps), risk_ctx['sc_beta'])


def live_skip_reason(compound, tyre_age, lap: int, laps_so_far: pd.DataFrame, risk_ctx: dict | None) -> str | None:
    """Why a flagged lap can't get a pit-now-vs-wait cost comparison, or
    None if it can. Reported on the flag instead of leaving it pending."""
    if risk_ctx is None:
        return "no usable pit-loss estimate for this race"
    if lr.wet_conditions(laps_so_far, lap):
        return (f"wet conditions (intermediate/wet tyres in use within {lr.WET_LOOKBACK_LAPS} laps) "
                f"-- wet-weather strategy is out of scope")
    if compound not in risk_ctx['compounds']:
        return f"{compound or 'unknown compound'} not modeled for this race"
    if tyre_age is None:
        return "tyre age unknown on this lap"
    return None


def ensure_topics(names: list) -> None:
    """Create the Kafka topics before subscribing. A topic auto-created by
    the producer's first send didn't exist yet when Spark subscribed, and
    that first message (e.g. an early VSC) was lost."""
    admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP)
    try:
        existing = set(admin.list_topics())
        missing = [NewTopic(n, num_partitions=1, replication_factor=1) for n in names if n not in existing]
        if missing:
            admin.create_topics(missing)
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def to_bronze_shape(batch: pd.DataFrame) -> pd.DataFrame:
    """Reshape the live Kafka message schema back into the same column
    names/dtypes build_race_features() expects from Bronze laps."""
    return pd.DataFrame({
        'Driver': batch['driver'], 'DriverNumber': batch['driver_number'], 'Team': batch['team'],
        'LapNumber': batch['lap_number'].astype(float),
        'Time': pd.to_timedelta(batch['session_time_s'], unit='s'),
        'LapTime': pd.to_timedelta(batch['lap_time_s'], unit='s'),
        'Compound': batch['compound'], 'TyreLife': batch['tyre_life'],
        'Position': batch['position'], 'TrackStatus': batch['track_status'],
        'PitInTime': pd.to_timedelta(batch['session_time_s'].where(batch['pit_in']), unit='s'),
        'PitOutTime': pd.to_timedelta(batch['session_time_s'].where(batch['pit_out']), unit='s'),
    })


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    race_id = sys.argv[1]

    emit = make_emitter(race_id)

    rivals, n_laps, event_name, live_ctx = load_static_context(race_id)
    if live_ctx['trigger'] is None:
        reason = "; ".join(live_ctx['problems'])
        print(f"Can't replay {race_id}: {reason}")
        emit({'type': 'error', 'message': f"Can't replay {race_id}: {reason}."})
        sys.exit(1)
    bundle = live_ctx['trigger']
    model, feature_cols, compound_categories = bundle['model'], bundle['feature_cols'], bundle['compound_categories']
    sc_prob_by_lap = bundle['sc_prob_by_lap']
    # The as-of model's own decision threshold, chosen on the last season of
    # the races it was trained on -- its probabilities are calibrated (no
    # class rebalancing), so this sits far below the old fixed 0.5.
    live_threshold = bundle['threshold']
    print(f"Loaded models as of {race_id} ({event_name}, {n_laps} laps): "
          f"trained on {live_ctx['n_prior_races']} earlier races, pit threshold P >= {live_threshold:.3f}")

    risk_ctx = live_ctx['risk']
    if risk_ctx is None:
        print("No tyre/pit-loss history before this race -- pit windows will open without a "
              "pit-now-vs-wait cost comparison.\n")
    else:
        risk_ctx['race_id'] = race_id
        print(f"Risk/reward context ready: green pit loss {risk_ctx['green_pit_loss']:.1f}s, "
              f"SC/VSC pit loss {risk_ctx['sc_pit_loss']:.1f}/{risk_ctx['vsc_pit_loss']:.1f}s "
              f"({risk_ctx['pit_loss_source']}), full-SC gap compression {risk_ctx['sc_beta']:.2f}\n")

    emit({'type': 'context', 'race_id': race_id, 'event_name': event_name, 'n_laps': n_laps,
          'n_prior_races': live_ctx['n_prior_races'], 'thin_history': live_ctx['thin_history'],
          'pit_threshold': live_threshold,
          'compounds_modeled': risk_ctx['compounds'] if risk_ctx else [],
          'green_pit_loss': risk_ctx['green_pit_loss'] if risk_ctx else None,
          'sc_pit_loss': risk_ctx['sc_pit_loss'] if risk_ctx else None,
          'vsc_pit_loss': risk_ctx['vsc_pit_loss'] if risk_ctx else None,
          'pit_loss_source': risk_ctx['pit_loss_source'] if risk_ctx else None,
          'circuit_seen': risk_ctx['circuit_seen'] if risk_ctx else False})

    # The default plan (Pirelli's prediction, else what finishers did here
    # before) and, per driver, the policy following it -- kept in step with
    # the stops the real car makes.
    plan = ds.default_plan(race_id, n_laps)
    followers = {}
    if plan is not None:
        print(f"Default plan: {len(plan['stops'])} stop(s) at laps {plan['stops']} "
              f"(windows {plan['windows']}), {'-'.join(plan['compounds'])} -- {plan['source']}\n")
        emit({'type': 'plan', 'stops': plan['stops'], 'windows': [list(w) for w in plan['windows']],
              'compounds': plan['compounds'], 'source': plan['source']})
    else:
        print("No default plan for this race (no Pirelli entry, no history here) -- plan calls disabled.\n")

    # Position-aware calls: overtaking model fitted on earlier races only.
    position_ctx = None
    if STRATEGY_MODE != 'time' and plan is not None and risk_ctx is not None and pos.EVENTS_FILE.exists():
        events = pd.read_csv(pos.EVENTS_FILE)
        prior_events = events[events['race'] < race_id]
        if len(prior_events):
            pos.FOLLOW_GAP_S = pos.follow_gap(prior_events)
            position_ctx = {'pm': pos.fit_pass_model(events, before_race=race_id), 'cache': {}}
            print(f"Position model ready: overtaking fitted on {prior_events['race'].nunique()} earlier races\n")

    seen_history = {'df': pd.DataFrame()}       # accumulated laps seen so far this race
    open_window = {}                              # driver -> id of their open pit window

    # The lap stream and the race-control stream run as separate Spark
    # queries and can fire concurrently; both read/update seen_history.
    lock = threading.Lock()

    def process_batch(batch_sdf, batch_id):
        with lock:
            process_lap_batch(batch_sdf, batch_id)
            react_to_due_neutralisations()

    # A deployment is only acted on once the lap stream has passed its
    # timestamp: the two streams are separate queries with their own
    # micro-batches, and in a fast replay (400x: one 2s lap batch = ~13
    # race-minutes) a deployment could otherwise be handled before laps
    # completed ahead of it had been ingested -- evaluating drivers from a
    # lap or two back. Laps completed after the deployment are then
    # excluded, so the decision uses exactly what was known at that moment.
    pending_neutralisations = []

    def process_rc_batch(batch_sdf, batch_id):
        events = batch_sdf.toPandas()
        if events.empty:
            return
        with lock:
            pending_neutralisations.extend(ev for _, ev in events.sort_values('session_time_s').iterrows())
            react_to_due_neutralisations()

    def react_to_due_neutralisations():
        if seen_history['df'].empty:
            return
        latest = seen_history['df']['Time'].max().total_seconds()
        while pending_neutralisations and pending_neutralisations[0]['session_time_s'] <= latest:
            react_to_neutralisation(pending_neutralisations.pop(0))

    def react_to_neutralisation(ev):
        """An SC/VSC was just deployed, mid-lap: for every running driver,
        "pit at the end of the lap you're on, or stay out?" -- the call a
        team makes within that lap. Decided from completed laps only, the
        lap in progress being the decision lap with the deployment's kind
        as the flag now out."""
        event_id = f"{ev['kind']}-{int(ev['lap'])}-{int(ev['session_time_s'])}"
        print(f"\n  >>> {ev['kind']} DEPLOYED (race-control lap {int(ev['lap'])}) -- evaluating pit-this-lap for every driver")
        emit({'type': 'neutralisation', 'event_id': event_id, 'kind': ev['kind'], 'lap': int(ev['lap'])})
        if risk_ctx is None or seen_history['df'].empty:
            return
        laps_so_far = seen_history['df'].drop_duplicates(['Driver', 'LapNumber'], keep='last')
        laps_so_far = laps_so_far[laps_so_far['Time'].dt.total_seconds() <= ev['session_time_s']]
        laps_so_far = tc.fuel_correct(laps_so_far, total_laps=n_laps)
        leader_lap = int(laps_so_far['LapNumber'].max())
        for driver, dl in laps_so_far.sort_values('LapNumber').groupby('Driver'):
            last = dl.iloc[-1]
            if last['LapNumber'] < leader_lap - 2:
                continue   # retired or long since lapped -- no live decision
            lap_now = int(last['LapNumber']) + 1   # the lap this driver is on
            if pd.notna(last['PitInTime']) or lap_now > n_laps:
                continue   # just pitted, or on the last lap
            compound = str(last['Compound']) if pd.notna(last['Compound']) else None
            age = float(last['TyreLife']) + 1 if pd.notna(last['TyreLife']) else None
            reason = live_skip_reason(compound, age, lap_now, laps_so_far, risk_ctx)
            result = None
            if reason is None:
                result = evaluate_pit_now_vs_wait(driver, str(last['Team']), lap_now, compound, int(age),
                                                  laps_so_far, n_laps, risk_ctx, kind_now=ev['kind'])
                if result is None:
                    reason = "no legal strategy can be solved from this state"
            if reason:
                emit({'type': 'sc_skipped', 'event_id': event_id, 'driver': driver, 'lap': lap_now, 'reason': reason})
                continue
            v = 'too close to call' if result['too_close'] else result['verdict']
            policy_pit = ds.PlanFollower.neutral_call(result['diff_s'])
            print(f"      {driver} (lap {lap_now}, {compound} age {int(age)}): {v} by {abs(result['diff_s']):.1f}s"
                  f" -> policy: {'PIT' if policy_pit else 'stay out'}")
            emit({'type': 'sc_decision', 'event_id': event_id, 'driver': driver, 'lap': lap_now,
                  'policy_pit': policy_pit, **result})

    def position_forecast():
        """(laps so far, the last lap every running car has completed) after
        making sure 15's forecast of the field from that lap is cached --
        shared by every car until the next lap completes; None if it can't
        be built."""
        hist = seen_history['df'].drop_duplicates(['Driver', 'LapNumber'], keep='last')
        last = hist.groupby('Driver')['LapNumber'].max()
        known = int(last[last >= last.max() - 1].min())
        cache = position_ctx['cache']
        if known not in cache:
            prepared = pos.prepare_laps(hist[hist['LapNumber'] <= known])
            degradation = ac.live_pace_lookup(risk_ctx['deg_bundle'], race_id, n_laps,
                                              hist['Time'].max(), risk_ctx['compounds'])
            if prepared is None or not degradation:
                return None
            teams = hist.groupby('Driver')['Team'].last()
            losses = {d: risk_ctx['green_pit_loss'] + float(risk_ctx['team_adj'].get(t, 0.0)) for d, t in teams.items()}
            replay = pos.RaceReplay(race_id, degradation, laps=prepared, n_laps=n_laps)
            cache.clear()
            cache[known] = replay.forecast_from(known, plan, losses)
            position_ctx['replay'], position_ctx['losses'] = replay, losses
        return hist, known

    def position_value(driver, team, decision_lap, follower, laps_so_far):
        """15's PositionDecider for the end of decision_lap, racing the
        cached forecast of the field."""
        got = position_forecast()
        if got is None:
            return None
        hist, known = got
        cache = position_ctx['cache']
        start = hist.loc[(hist['Driver'] == driver) & (hist['LapNumber'] == 1), 'Compound']
        if start.empty or position_ctx['replay'].degradation.get(start.iloc[0]) is None:
            return None
        decider = pos.PositionDecider(position_ctx['replay'], driver, plan, start.iloc[0],
                                      position_ctx['losses'].get(driver, risk_ctx['green_pit_loss']), None,
                                      position_ctx['pm'], (rivals.get(driver) or {}).get('rival_driver'),
                                      position_ctx['losses'], cache)
        return decider(decision_lap, list(follower.done), list(follower.pending), known_lap=known)

    strategies = {}   # driver -> {'future': planned stop in-laps, 'done': real stops so far}

    def strategy_call(driver, team, row, lap, decision_lap):
        st = strategies.setdefault(driver, {'future': [s for s in plan['stops'] if 2 < s < n_laps - 1], 'done': []})
        if row['pitted_this_lap'] and lap not in st['done']:
            # The real car stopped: it replaces the next planned stop if past
            # that stint's halfway point; the rest are re-spaced (as 14's
            # PlanFollower and 06's follow_strategy_policy).
            prev = st['done'][-1] if st['done'] else 0
            rest = st['future'][1:] if st['future'] and lap >= (prev + st['future'][0]) / 2 else st['future']
            st['done'].append(lap)
            st['future'] = [int(round(lap + (i + 1) * (n_laps - lap) / (len(rest) + 1))) for i in range(len(rest))]
        st['future'] = [s for s in st['future'] if s >= decision_lap]
        if not 2 < decision_lap < n_laps - 1:
            return
        near = bool(st['future']) and st['future'][0] - decision_lap <= pos.NEAR_STOP
        if (decision_lap - 3) % pos.REPLAN_EVERY == 0 or near:
            got = position_forecast()
            hist = got[0] if got else None
            dl = hist[hist['Driver'] == driver].sort_values('LapNumber') if hist is not None else None
            if got and dl is not None and len(dl) and position_ctx['replay'].degradation.get(dl['Compound'].iloc[0]):
                _, comps = pos.real_strategy_from(dl)
                if row['pitted_this_lap'] and len(comps) < len(st['done']) + 1:
                    guess = pos.choose_compounds(position_ctx['replay'].degradation, comps,
                                                 sorted(st['done'] + st['future']), n_laps)
                    comps = comps + [guess[len(comps)] if guess else comps[-1]]
                decider = pos.StrategyDecider(position_ctx['replay'], driver, plan,
                                              position_ctx['losses'].get(driver, risk_ctx['green_pit_loss']), None,
                                              position_ctx['pm'], (rivals.get(driver) or {}).get('rival_driver'),
                                              position_ctx['losses'], position_ctx['cache'])
                new = decider(decision_lap, list(st['done']), comps, list(st['future']), known_lap=got[1])
                if new != st['future']:
                    print(f"      strategy: {driver} re-plans stops {st['future']} -> {new}")
                    emit({'type': 'plan_update', 'driver': driver, 'lap': decision_lap,
                          'old': st['future'], 'new': new})
                    st['future'] = new
                near = bool(st['future']) and st['future'][0] - decision_lap <= pos.NEAR_STOP
        if near:
            nxt = st['future'][0]
            pit_now = nxt == decision_lap
            reason = 'planned stop' if pit_now else f"plan: stop at the end of lap {nxt}"
            print(f"      strategy: end of lap {decision_lap} -> {'PIT' if pit_now else 'stay out'} ({reason})")
            emit({'type': 'plan_call', 'driver': driver, 'lap': decision_lap, 'pit': pit_now, 'reason': reason,
                  'stop_index': len(st['done']), 'planned_lap': nxt, 'window': [nxt, nxt], 'diff_s': None})

    def process_lap_batch(batch_sdf, batch_id):
        batch = batch_sdf.toPandas()
        if batch.empty:
            return
        new_laps = to_bronze_shape(batch)
        seen_history['df'] = pd.concat([seen_history['df'], new_laps], ignore_index=True)
        laps_so_far = seen_history['df'].drop_duplicates(['Driver', 'LapNumber'], keep='last')

        # Scheduled race length, not the highest lap seen so far -- see
        # fuel_correct()/build_race_features() in 05.
        laps_so_far = tc.fuel_correct(laps_so_far, total_laps=n_laps)
        field_pace = tc.compute_field_pace(laps_so_far)
        laps_so_far = laps_so_far.join(field_pace, on='LapNumber')
        laps_so_far['RelativePace'] = laps_so_far['FuelCorrectedLapTime'] - laps_so_far['FieldPace']

        feats = tc.build_race_features(laps_so_far, race_id, event_name, rivals, n_laps=n_laps)
        # build_race_features only creates a column if at least one row has
        # a value for it (pd.DataFrame(list_of_dicts) behavior). Offline
        # this is never an issue -- a full race always has later laps mixed
        # in -- but the very first live micro-batch can legitimately be
        # nothing but lap-1 events for every driver, with no previous lap
        # for anyone yet, silently dropping every lagged/trend/rival column
        # instead of NaN-filling it. Backfill any that are missing.
        for col in feature_cols:
            if col not in feats.columns:
                feats[col] = float('nan')
        feats['circuit_sc_prob'] = feats['lap'].map(sc_prob_by_lap)
        feats['own_compound'] = pd.Categorical(feats['own_compound'], categories=compound_categories)

        # Only score the laps that actually arrived in this batch -- no
        # need to re-predict laps already handled in earlier batches.
        new_keys = set(zip(new_laps['Driver'], new_laps['LapNumber']))
        to_score = feats[feats.apply(lambda r: (r['driver'], r['lap']) in new_keys, axis=1)]
        if to_score.empty:
            return

        proba = model.predict_proba(to_score[feature_cols])[:, 1]
        scored = list(zip(to_score.iterrows(), proba))
        scored.sort(key=lambda item: (item[0][1]['driver'], item[0][1]['lap']))

        for (_, row), p in scored:
            # p = P(this driver pits at the end of the NEXT lap), from
            # everything known now that lap `lap` has ended (05 predicts one
            # lap ahead) -- so a call can be made before the stop happens.
            driver, lap = row['driver'], int(row['lap'])
            decision_lap = lap + 1
            driver_team = laps_so_far.loc[laps_so_far['Driver'] == driver, 'Team'].iloc[-1]
            tyre_age = float(row['own_tyre_age']) + 1 if pd.notna(row['own_tyre_age']) else None
            compound_str = str(row['own_compound']) if pd.notna(row['own_compound']) else None
            print(f"  lap {lap:2d}  {driver}  P(pit end of lap {decision_lap})={p:.2f}  {compound_str} age {row['own_tyre_age']}")
            emit({'type': 'lap_score', 'lap': lap, 'driver': driver, 'team': str(driver_team),
                  'proba': float(p), 'tyre_age': row['own_tyre_age'] if pd.notna(row['own_tyre_age']) else None,
                  'compound': compound_str})

            evaluated = {}   # decision lap -> cost-model result, shared by the plan call and the pit window

            def evaluate():
                if decision_lap not in evaluated:
                    reason = live_skip_reason(compound_str, tyre_age, decision_lap, laps_so_far, risk_ctx)
                    result = None
                    if reason is None:
                        result = evaluate_pit_now_vs_wait(driver, driver_team, decision_lap, compound_str,
                                                          int(tyre_age), laps_so_far, n_laps, risk_ctx)
                        if result is None:
                            reason = "no legal strategy can be solved from this state (too few laps left?)"
                    evaluated[decision_lap] = (result, reason)
                return evaluated[decision_lap]

            # Strategy mode: this car's own rest-of-race plan, re-planned by
            # 15's StrategyDecider against the forecast field.
            if plan is not None and STRATEGY_MODE == 'strategy' and position_ctx is not None:
                strategy_call(driver, driver_team, row, lap, decision_lap)
            # Otherwise the policy's call for the end of the next lap: follow
            # the plan unless the cost model is clearly sure inside the window.
            elif plan is not None:
                follower = followers.setdefault(driver, ds.PlanFollower(plan, n_laps))
                if row['pitted_this_lap']:
                    follower.record_stop(lap)
                if 2 < decision_lap < n_laps - 1 and follower.pending:
                    diff, call = None, None
                    if position_ctx is not None and follower.wants_evaluation(decision_lap):
                        diff = position_value(driver, driver_team, decision_lap, follower, laps_so_far)
                        if diff is not None:
                            call = follower.green_call(decision_lap, diff, margin=pos.POSITION_MARGIN, unit=' pts')
                    if call is None:   # no position-model call possible: the time-based one
                        diff = evaluate()[0]['diff_s'] if follower.wants_evaluation(decision_lap) and evaluate()[0] else None
                        call = follower.green_call(decision_lap, diff)
                    if call:
                        planned, (lo, hi) = follower.pending[0]
                        print(f"      plan: end of lap {decision_lap} -> {'PIT' if call[0] else 'stay out'} ({call[1]})")
                        emit({'type': 'plan_call', 'driver': driver, 'lap': decision_lap, 'pit': bool(call[0]),
                              'reason': call[1], 'stop_index': len(follower.done), 'planned_lap': planned,
                              'window': [lo, hi], 'diff_s': diff})

            if row['pitted_this_lap'] or p < live_threshold * 0.5:
                open_window.pop(driver, None)   # stopped, or the signal went quiet: re-arm
                continue
            # Same candidate window as every decision rule in 06 (no stop at
            # the end of laps 1-2 -- first-lap damage the classifier can't
            # see coming -- or in the final laps).
            if p < live_threshold or not (2 < decision_lap < n_laps - 1):
                continue

            # Pit window: opened when P first crosses the threshold, then
            # re-evaluated every lap until the car stops or the signal goes
            # quiet. The classifier says the window is open; the cost model
            # (10's expected-cost DP) says whether to stop at the end of the
            # next lap or keep going.
            if driver not in open_window:
                open_window[driver] = f"{driver}-{decision_lap}"
                print(f"      >>> pit window opens: P={p:.2f} for the end of lap {decision_lap}")
                emit({'type': 'pit_call', 'window_id': open_window[driver], 'driver': driver,
                      'lap': decision_lap, 'proba': float(p)})
            window_id = open_window[driver]
            result, reason = evaluate()
            if reason:
                print(f"      (lap {decision_lap} not evaluated: {reason})")
                emit({'type': 'skipped', 'window_id': window_id, 'lap': decision_lap, 'driver': driver,
                      'proba': float(p), 'reason': reason})
                continue
            v = 'too close to call' if result['too_close'] else result['verdict']
            print(f"      -> end of lap {decision_lap}: pit {result['pit_now_s']:.1f}s vs stay out "
                  f"{result['wait_s']:.1f}s expected ==> {v} by {abs(result['diff_s']):.1f}s")
            emit({'type': 'risk_comparison', 'window_id': window_id, 'lap': decision_lap, 'driver': driver,
                  'proba': float(p), **result})

    ensure_topics([TOPIC, TOPIC_RACE_CONTROL])
    spark = SparkSession.builder \
        .appName('f1-streaming-consumer') \
        .master('local[2]') \
        .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0') \
        .getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    raw = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', KAFKA_BOOTSTRAP) \
        .option('subscribe', TOPIC) \
        .option('startingOffsets', 'latest') \
        .load()

    parsed = raw.selectExpr('CAST(value AS STRING) as json') \
        .select(from_json(col('json'), LAP_SCHEMA).alias('d')) \
        .select('d.*') \
        .filter(f"race = '{race_id}'")

    parsed.writeStream.foreachBatch(process_batch).trigger(processingTime='2 seconds').start()

    rc_raw = spark.readStream.format('kafka') \
        .option('kafka.bootstrap.servers', KAFKA_BOOTSTRAP) \
        .option('subscribe', TOPIC_RACE_CONTROL) \
        .option('startingOffsets', 'latest') \
        .load()
    rc_parsed = rc_raw.selectExpr('CAST(value AS STRING) as json') \
        .select(from_json(col('json'), RACE_CONTROL_SCHEMA).alias('d')) \
        .select('d.*') \
        .filter(f"race = '{race_id}'")
    rc_parsed.writeStream.foreachBatch(process_rc_batch).trigger(processingTime='1 second').start()

    print(f"Listening on '{TOPIC}' and '{TOPIC_RACE_CONTROL}' for race={race_id} -- "
          f"start 08_streaming_producer.py now.\n")
    emit({'type': 'ready'})
    spark.streams.awaitAnyTermination()


if __name__ == '__main__':
    main()
