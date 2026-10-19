"""
Phase 6 -- streaming replay producer. Publishes one historical race's laps
to Kafka in real (accelerated) chronological order -- the literal
implementation of Psaltiras's declared future-work chapter this thesis is
built around: historical races pushed through Kafka in real time order,
consumed by Spark Structured Streaming, so a strategy model can be queried
mid-race exactly as a live system would use it.

Each Kafka message is one driver's completed lap, published at the moment
it "would have" completed in the real race, compressed by SPEED_FACTOR.
Ordering across drivers is by actual session-elapsed Time, not grouped by
driver or lap number -- a real live feed sees whichever car crosses the
line next, not one driver's whole race at once.

Safety car / VSC deployments are also published, to a second topic
(TOPIC_RACE_CONTROL), at the moment race control announced them -- mid-lap,
not when the next lap completes. The pit-or-stay call under a safety car is
made within that lap, and a car pitting at the end of it is already in the
pit lane by the time its lap record arrives, so reacting only to lap
records came a lap too late.

Requires a running Kafka broker (see README's "Streaming layer" section
for the docker run command) and JAVA_HOME set for the consumer side.

Run with: python notebooks/08_streaming_producer.py <race_id> [speed_factor]
  e.g.    python notebooks/08_streaming_producer.py 2023_r07 50
"""
import sys
import json
import time
import importlib.util
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer

_spec = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpl)

BRONZE_DIR = Path('data/bronze')
KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC = 'f1-race-laps'
TOPIC_RACE_CONTROL = 'f1-race-control'
DEFAULT_SPEED_FACTOR = 50  # 50x real time -- a ~90 min race replays in ~2 min


def load_race(race_id: str) -> pd.DataFrame:
    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    laps['TimeSeconds'] = laps['Time'].dt.total_seconds()
    return laps.sort_values('TimeSeconds')


def lap_to_message(row: pd.Series) -> dict:
    """JSON-serializable snapshot of one completed lap -- only fields a
    live timing feed would actually carry moment-to-moment."""
    return {
        'race': row['race_id'],
        'driver': row['Driver'],
        'driver_number': str(row['DriverNumber']),
        'team': row['Team'],
        'lap_number': int(row['LapNumber']),
        'session_time_s': float(row['TimeSeconds']),
        'lap_time_s': row['LapTime'].total_seconds() if pd.notna(row['LapTime']) else None,
        'compound': row['Compound'],
        'tyre_life': float(row['TyreLife']) if pd.notna(row['TyreLife']) else None,
        'position': float(row['Position']) if pd.notna(row['Position']) else None,
        'track_status': row['TrackStatus'],
        'pit_in': pd.notna(row['PitInTime']),
        'pit_out': pd.notna(row['PitOutTime']),
    }


def neutralisation_events(race_id: str, laps: pd.DataFrame) -> list:
    """SC/VSC deployments as (session_time_s, message). Race-control
    timestamps are wall-clock, lap times are session time; the chequered
    flag ties the two (it's shown as the leader completes the final lap).
    Without one (e.g. a race ended by red flag), a deployment is placed
    just after the leader completed the lap before."""
    rc = pd.read_parquet(BRONZE_DIR / f"{race_id}_race_control.parquet")
    deployed = rc[(rc['Category'] == 'SafetyCar') & (rc['Status'] == 'DEPLOYED')]
    flag = rc[rc['Message'].astype(str).str.contains('CHEQUERED FLAG')]
    leader_end = laps.groupby('LapNumber')['TimeSeconds'].min()
    offset = None
    if len(flag):
        offset = flag['Time'].iloc[0].timestamp() - float(leader_end.iloc[-1])
    events = []
    for _, r in deployed.iterrows():
        if offset is not None:
            t = r['Time'].timestamp() - offset
        else:
            t = float(leader_end.get(int(r['Lap']) - 1, leader_end.iloc[0])) + 1.0
        events.append((t, {'race': race_id, 'kind': 'VSC' if dpl.is_vsc_message(r['Message']) else 'SC',
                           'lap': int(r['Lap']), 'session_time_s': t, 'message': str(r['Message'])}))
    return events


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    race_id = sys.argv[1]
    speed_factor = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SPEED_FACTOR

    laps = load_race(race_id)
    laps['race_id'] = race_id
    event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id
    n_events = len(laps)
    duration_s = laps['TimeSeconds'].max() - laps['TimeSeconds'].min()
    print(f"Replaying {race_id} ({event_name}): {n_events} lap events over "
          f"{duration_s/60:.0f} real minutes, at {speed_factor:.0f}x -> "
          f"~{duration_s/speed_factor/60:.1f} minutes of replay")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8'),
    )

    # One timeline: lap completions and SC/VSC deployments, in session-time order.
    timeline = [(row['TimeSeconds'], 'lap', row) for _, row in laps.iterrows()]
    timeline += [(t, 'race_control', msg) for t, msg in neutralisation_events(race_id, laps)]
    timeline.sort(key=lambda x: x[0])
    print(f"  plus {sum(1 for _, k, _ in timeline if k == 'race_control')} SC/VSC deployments "
          f"on topic '{TOPIC_RACE_CONTROL}'")

    start_wall = time.time()
    start_sim = timeline[0][0]
    sent = 0
    for t, kind, item in timeline:
        sleep_for = start_wall + (t - start_sim) / speed_factor - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        if kind == 'race_control':
            producer.send(TOPIC_RACE_CONTROL, key=item['kind'], value=item)
            print(f"  {item['kind']} deployed on lap {item['lap']} @ t={t:.0f}s")
            continue
        msg = lap_to_message(item)
        producer.send(TOPIC, key=item['Driver'], value=msg)
        sent += 1
        if sent % 50 == 0 or sent == n_events:
            print(f"  [{sent}/{n_events}] lap {msg['lap_number']:2d} {msg['driver']} "
                  f"@ t={msg['session_time_s']:.0f}s")

    producer.flush()
    producer.close()
    print(f"Done -- published {sent} lap events to topic '{TOPIC}'")


if __name__ == '__main__':
    main()
