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

Live decision rule brings the offline backtest's core finding -- that the
model's probability *ranking* carries more signal than a fixed threshold
(06_backtest_evaluation.py) -- into a form that's actually causal: online
peak detection. A live system can't scan a whole completed race for the
best peaks the way the offline ranking does; it can only tell a value was
a local peak once it sees the *next* value come in lower. So a
recommendation here fires one lap after the actual high point, explicitly
naming that earlier lap ("peak was lap N") rather than claiming "pit
now" -- a real, stated one-lap detection lag inherent to true online
operation, not something to hide behind the printed output.

A stint can have more than one local peak (e.g. a brief VSC produces a
small early blip, then the real SC-triggered mass-pit signal several laps
later) -- an earlier peak firing must not block a later, separate one from
also being reported, or the rule buries its own best signal behind its
first false alarm. The rule that got this right, found by directly
investigating a specific case (2023_r08, MAG/PER): don't require the next
peak to be *taller* than the last one flagged -- their real mass pit stop
(lap 12, P=0.79) was honestly lower than the earlier false-alarm blip
(lap 7, P=0.92) that fired first, so a magnitude-based "only escalate if
bigger" rule still misses it. What actually distinguishes a separate,
later rise from just noise around the same decline is whether the signal
genuinely went quiet in between -- so the detector disarms after firing
and only re-arms once probability drops below half the threshold,
independent of how tall either peak was.

Scope simplification, stated plainly: each micro-batch recomputes
features over ALL laps seen so far in the race (correct live semantics --
only ever uses information available up to that point), rather than
maintaining true incremental state. Fine for a single-race proof-of-
concept (at most ~1500 lap events), not how a production system would do
it at scale.

Requires: a running Kafka broker (see README), JAVA_HOME set, and a
trained model at data/models/trigger_classifier.joblib (from
05_trigger_classifier.py).

Run with: JAVA_HOME=~/.local/jdk python notebooks/09_streaming_consumer.py <race_id>
"""
import sys
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC = 'f1-race-laps'
LIVE_THRESHOLD = 0.5
BRONZE_DIR = Path('data/bronze')

# Load 05_trigger_classifier.py's feature-building functions directly, so
# live and offline feature computation can never drift apart.
_spec = importlib.util.spec_from_file_location(
    'trigger_classifier_lib', Path(__file__).parent / '05_trigger_classifier.py')
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

LAP_SCHEMA = StructType([
    StructField('race', StringType()), StructField('driver', StringType()),
    StructField('driver_number', StringType()), StructField('team', StringType()),
    StructField('lap_number', LongType()), StructField('session_time_s', DoubleType()),
    StructField('lap_time_s', DoubleType()), StructField('compound', StringType()),
    StructField('tyre_life', DoubleType()), StructField('position', DoubleType()),
    StructField('track_status', StringType()), StructField('pit_in', BooleanType()),
    StructField('pit_out', BooleanType()),
])


def load_static_context(race_id: str):
    """Everything knowable before the race starts: rival identities, the
    circuit's SC-probability curve (leave-one-race-out, precomputed by
    05_trigger_classifier.py), the trained model, and the historical
    race's total lap count (known here because we're replaying a
    finished race; a genuinely live system would track this as the race
    unfolds instead)."""
    rivals_df = pd.read_csv('data/championship_rivals.csv')
    rivals = rivals_df[rivals_df['race'] == race_id].set_index('driver')[
        ['rival_driver', 'rival_team']].to_dict('index')

    features_df = pd.read_parquet('data/silver/trigger_classifier_features.parquet')
    race_features = features_df[features_df['race'] == race_id]
    sc_prob_by_lap = race_features.drop_duplicates('lap').set_index('lap')['circuit_sc_prob'].to_dict()

    bundle = joblib.load('data/models/trigger_classifier.joblib')
    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    n_laps = int(laps['LapNumber'].max())
    event_name = laps['EventName'].iloc[0]

    return rivals, sc_prob_by_lap, bundle, n_laps, event_name


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

    rivals, sc_prob_by_lap, bundle, n_laps, event_name = load_static_context(race_id)
    model, feature_cols, compound_categories = bundle['model'], bundle['feature_cols'], bundle['compound_categories']
    print(f"Loaded model + static context for {race_id} ({event_name}, {n_laps} laps)\n")

    seen_history = {'df': pd.DataFrame()}       # accumulated laps seen so far this race
    last_recommendation_lap = {}                 # driver -> last lap a recommendation fired, resets after a real stop
    last_pit_lap = {}                             # driver -> lap of their most recent actual stop
    prob_history = {}                             # driver -> [(lap, proba), ...] since their last stop
    armed = {}                                     # driver -> ready to flag a new peak (re-arms once the signal goes quiet)

    def process_batch(batch_sdf, batch_id):
        batch = batch_sdf.toPandas()
        if batch.empty:
            return
        new_laps = to_bronze_shape(batch)
        seen_history['df'] = pd.concat([seen_history['df'], new_laps], ignore_index=True)
        laps_so_far = seen_history['df'].drop_duplicates(['Driver', 'LapNumber'], keep='last')

        laps_so_far = tc.fuel_correct(laps_so_far)
        field_pace = tc.compute_field_pace(laps_so_far)
        laps_so_far = laps_so_far.join(field_pace, on='LapNumber')
        laps_so_far['RelativePace'] = laps_so_far['FuelCorrectedLapTime'] - laps_so_far['FieldPace']

        feats = tc.build_race_features(laps_so_far, race_id, event_name, rivals)
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
        # Sort by (driver, lap) so each driver's probabilities are processed
        # in order even when a batch mixes multiple drivers/laps together --
        # the peak-detection logic below depends on seeing them sequentially.
        scored = list(zip(to_score.iterrows(), proba))
        scored.sort(key=lambda item: (item[0][1]['driver'], item[0][1]['lap']))

        for (_, row), p in scored:
            driver, lap = row['driver'], row['lap']
            if row['label_pit']:
                last_pit_lap[driver] = lap
                prob_history[driver] = []   # fresh stint, forget the last one's curve

            # Online peak detection -- the causal equivalent of the offline
            # backtest's ranking approach (06_backtest_evaluation.py), which
            # can only work there because it sees the whole race at once. A
            # live system can't know today is the peak until it sees
            # tomorrow's value is lower, so this necessarily fires one lap
            # *after* the actual peak, flagging that earlier lap by number
            # rather than claiming "now" -- an honest, stated tradeoff of
            # true online operation, not hidden behind the printed output.
            history = prob_history.setdefault(driver, [])
            flag = ""
            if history:
                prev_lap, prev_p = history[-1]
                stint_start = last_pit_lap.get(driver, -1)
                if last_recommendation_lap.get(driver, -1) <= stint_start:
                    armed[driver] = True   # new stint -- ready to flag again

                # A later, separate peak in the same stint (e.g. a transient
                # VSC blip early on, then the real SC-triggered mass-pit
                # signal several laps after) must still get through. Trying
                # to require it be *stronger* than the first is wrong -- a
                # later peak can be the genuinely correct one even at a
                # lower raw probability (found exactly this case: 2023_r08
                # MAG/PER peaked at 0.92 on a false-alarm VSC blip, lap 7,
                # then again at a truthful, lower 0.79 at lap 12, matching
                # the field's real mass pit stop). So instead: once fired,
                # require the signal to genuinely go quiet (drop below half
                # the threshold) before arming for the next one -- that's
                # what distinguishes "still the same decline" from "a
                # separate, later rise", not how tall either peak was.
                if prev_p >= LIVE_THRESHOLD and p < prev_p and armed.get(driver, True):
                    flag = f"  <-- peak was lap {prev_lap:.0f} (P={prev_p:.2f}), consider pitting there"
                    last_recommendation_lap[driver] = lap
                    armed[driver] = False
            # Re-arm using THIS lap's own value, checked after this lap's
            # peak-check above so it only affects future comparisons.
            if p < LIVE_THRESHOLD * 0.5:
                armed[driver] = True
            history.append((lap, p))

            print(f"  lap {lap:2.0f}  {driver}  P(pit)={p:.2f}  tyre_age={row['own_tyre_age']:.0f} "
                  f"{row['own_compound']}{flag}")

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

    query = parsed.writeStream.foreachBatch(process_batch).trigger(processingTime='2 seconds').start()
    print(f"Listening on '{TOPIC}' for race={race_id} -- start 08_streaming_producer.py now.\n")
    query.awaitTermination()


if __name__ == '__main__':
    main()
