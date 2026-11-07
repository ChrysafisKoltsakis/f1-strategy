#!/usr/bin/env bash
# Rebuild everything in data/ in the right order (see README 'Rebuilding and running').
#   scripts/rebuild.sh                  data + models + time-scored backtest + position model
#   scripts/rebuild.sh --retrain-tyre   also re-run 03 even if a tyre model exists
#   scripts/rebuild.sh --policies       also the position-aware policies in 06 (~2 h more)
# 03 (tyre model) is skipped when data/models/degradation_model.joblib exists: it is
# frozen for the forward test (docs/forward_test.md). Pass --retrain-tyre to override.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
RETRAIN=0; POLICIES=0
for a in "$@"; do case $a in --retrain-tyre) RETRAIN=1;; --policies) POLICIES=1;; *) echo "unknown option $a"; exit 1;; esac; done

"$PY" -m fastf1 auth f1tv --status 2>&1 | grep -q "Subscription Status: active" \
  || echo "NOTE: not logged in to F1 TV; races after 2026 round 16 can't be downloaded. Log in: $PY -m fastf1 auth f1tv --authenticate"

run() { echo; echo "=== $1 ==="; shift; "$PY" "$@"; }
run "01 pull data"            notebooks/01_pull_bronze_data.py
run "07 championship rivals"  notebooks/07_championship_rivals.py
run "02 pit losses, SC/VSC"   notebooks/02_fixed_degradation_pitloss.py
if [ "$RETRAIN" = 1 ] || [ ! -f data/models/degradation_model.joblib ]; then
  run "03 tyre model"         notebooks/03_ml_degradation_model.py
else
  echo; echo "=== 03 tyre model: skipped (frozen; use --retrain-tyre to override) ==="
fi
run "04 DP baseline"          notebooks/04_dp_baseline.py
run "05 trigger classifier"   notebooks/05_trigger_classifier.py
if [ "$POLICIES" = 1 ]; then
  run "06 backtest + policies" notebooks/06_backtest_evaluation.py --position --strategy --strategy-guarded
else
  run "06 backtest"           notebooks/06_backtest_evaluation.py
fi
run "13 teammate validation"  notebooks/13_strategy_cost_validation.py
run "15 position model"       notebooks/15_position_model.py
echo; echo "Rebuild finished."
