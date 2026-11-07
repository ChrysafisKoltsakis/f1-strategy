#!/usr/bin/env bash
# Start Kafka (if needed) and the live dashboard, then open http://localhost:8000
#   scripts/dashboard.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$ROOT/scripts/kafka.sh" start
echo "Dashboard: http://localhost:8000   (Ctrl+C to stop)"
exec "$PY" dashboard/server.py
