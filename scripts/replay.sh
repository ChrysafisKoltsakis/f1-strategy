#!/usr/bin/env bash
# Replay one race through Kafka + Spark without the dashboard, e.g.
#   scripts/replay.sh 2023_r08 150
# Events land in data/live_events/<race>.jsonl, logs in data/live_logs/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RACE="${1:?usage: scripts/replay.sh <race_id> [speed, default 150]}"
SPEED="${2:-150}"
EV="data/live_events/$RACE.jsonl"
mkdir -p data/live_logs
"$ROOT/scripts/kafka.sh" start
rm -f "$EV"
"$PY" notebooks/09_streaming_consumer.py "$RACE" > "data/live_logs/$RACE.consumer.log" 2>&1 &
CONSUMER=$!
trap 'kill $CONSUMER 2>/dev/null || true' EXIT
printf "Waiting for the consumer (Spark boot + as-of models, up to a few minutes)"
for _ in $(seq 1 300); do grep -qs '"ready"' "$EV" && break; printf "."; sleep 1; done
grep -qs '"ready"' "$EV" || { echo; echo "Consumer not ready, see data/live_logs/$RACE.consumer.log"; tail -15 "data/live_logs/$RACE.consumer.log"; exit 1; }
echo " ready. Replaying at ${SPEED}x..."
"$PY" notebooks/08_streaming_producer.py "$RACE" "$SPEED" > "data/live_logs/$RACE.producer.log" 2>&1
# done when the consumer has been quiet for 30 s
while [ $(( $(date +%s) - $(stat -c %Y "$EV") )) -lt 30 ]; do sleep 5; done
echo "Done: $(wc -l < "$EV") events in $EV"
