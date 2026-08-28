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

Requires a running Kafka broker (see README's "Streaming layer" section
for the docker run command) and JAVA_HOME set for the consumer side.

Run with: python notebooks/08_streaming_producer.py <race_id> [speed_factor]
  e.g.    python notebooks/08_streaming_producer.py 2023_r07 50
"""
import sys
import json
import time
import pandas as pd
from pathlib import Path
from kafka import KafkaProducer

BRONZE_DIR = Path('data/bronze')
KAFKA_BOOTSTRAP = 'localhost:9092'
TOPIC = 'f1-race-laps'
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

    start_wall = time.time()
    start_sim = laps['TimeSeconds'].iloc[0]
    sent = 0
    for _, row in laps.iterrows():
        target_wall = start_wall + (row['TimeSeconds'] - start_sim) / speed_factor
        sleep_for = target_wall - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

        msg = lap_to_message(row)
        producer.send(TOPIC, key=row['Driver'], value=msg)
        sent += 1
        if sent % 50 == 0 or sent == n_events:
            print(f"  [{sent}/{n_events}] lap {msg['lap_number']:2d} {msg['driver']} "
                  f"@ t={msg['session_time_s']:.0f}s")

    producer.flush()
    producer.close()
    print(f"Done -- published {sent} lap events to topic '{TOPIC}'")


if __name__ == '__main__':
    main()
