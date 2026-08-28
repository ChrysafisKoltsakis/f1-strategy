# F1 Strategy

A data pipeline and strategy optimization engine for Formula 1 pit-stop decisions, built as a thesis project.

Ingests multi-season F1 timing/telemetry data (via [FastF1](https://github.com/theOehrly/Fast-F1)), builds per-circuit tyre-degradation and pit-loss models, and works toward a real-time strategy recommendation system: given the live state of a race, should a team pit now or hold.

## Status

Currently in the experimentation phase. Built so far, covering the full scope-guardrail MVP:

- **Bronze-layer ingestion** (`notebooks/01_pull_bronze_data.py`) — pulls race sessions (laps, weather, race-control, results) via FastF1 across multiple seasons (2022+) and lands them as Parquet. Idempotent — safe to re-run as new seasons/rounds become available. Currently 103 races (all of 2022-2025, 2026 to date).
- **Cleaning + degradation/pit-loss modeling** (`notebooks/02_fixed_degradation_pitloss.py`) — filters pit laps, standing-start laps, SC/VSC-restart-affected laps, and non-green-flag laps; detrends lap time against the field's per-lap median pace to separate tyre degradation from track evolution; estimates per-circuit green and SC/VSC pit-loss constants, plus a per-team pit-loss adjustment pooled across races.
- **ML degradation model** (`notebooks/03_ml_degradation_model.py`) — LightGBM model predicting relative pace (vs. field median) from tyre age, compound, weather, and circuit, validated with leave-one-circuit-out cross-validation (~0.72s MAE). Persisted (`data/models/degradation_model.joblib`) and actually consumed by the DP baseline, backtest, and live risk-comparison tool via a precomputed per-race (compound, lap, tyre age) prediction grid — captures non-linear/cliff wear and pools across circuits/seasons, unlike the simpler per-race linear fit those tools originally used.
- **DP hindsight-optimal baseline** (`notebooks/04_dp_baseline.py`) — dynamic program computing the mathematically optimal pit strategy per race per team, used as the evaluation ruler.
- **Trigger/strategy classifier** (`notebooks/05_trigger_classifier.py`) — LightGBM imitation-learning model predicting pit decisions from live race state (on-track gaps, tyre age, championship-rival state, momentum/trend features), with a causal-confusion audit on the feature set.
- **Championship-rival computation** (`notebooks/07_championship_rivals.py`) — cumulative standings and nearest drivers'/constructors' rival per race, feeding the classifier.
- **3-way backtest evaluation** (`notebooks/06_backtest_evaluation.py`) — actual vs. classifier-driven vs. DP-hindsight-optimal strategy, replayed through the same lap-time model.
- **Streaming layer** (`notebooks/08_streaming_producer.py` + `09_streaming_consumer.py`) — replays a historical race's laps through Kafka in real (accelerated) chronological order, consumed by Spark Structured Streaming, which re-invokes the trained classifier after each new lap for a live pit probability, then online peak-detects it into a recommended lap. When a peak fires, it also re-solves 10's pit-now-vs-wait DP live, using only laps observed so far (a causal field-pace forecast, not real future laps) plus the historical SC/VSC recurrence rate, so the printed output is a genuine live recommendation -- probability, cost, and risk together -- not just a probability. The literal implementation of Psaltiras's declared future-work chapter this thesis is built around.
- **Live risk/reward comparison** (`notebooks/10_live_risk_comparison.py`) — at a mid-race decision point (e.g. an SC/VSC just appeared), re-solves the DP baseline from the *current* live state to compare "pit now under the cheap SC/VSC window" vs. "wait for a possible later one", weighted by the real historical frequency of a race getting a second SC/VSC after the first — a genuine risk/reward tradeoff grounded in historical data, not a hand-built probability model. Also exposes a causal, live-safe version of its field-pace forecast that 09_streaming_consumer.py calls directly.
- **Per-lap risk/reward report** (`notebooks/11_pit_lap_risk_report.py`) — reports the classifier's top candidate pit laps for a driver side by side with their model confidence *and* actual predicted time cost, instead of collapsing to one recommendation. Flags when the top two candidates are a statistical toss-up (found on a real case: two laps separated by a 0.005 probability margin, where the "losing" lap was actually 5 seconds faster).
- **Live dashboard** (`dashboard/server.py`) — a local web viewer onto the real streaming system above: launches 08/09 as subprocesses for a chosen race and speed, and streams every event they produce (per-lap probability, peak flags, pit-now-vs-wait cost comparisons) to the browser over Server-Sent Events. A view onto the genuine Kafka/Spark pipeline, not a reimplementation of it. Functional first pass — visual design deliberately deferred.

Not yet built: proper design polish on the dashboard; wet-race (rain/intermediate) strategy is out of scope for the cost/risk comparison everywhere in this project, not just the dashboard.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/fastf1_cache
```

### Streaming layer (Kafka + Spark)

The streaming layer needs a JVM (for PySpark) and a running Kafka broker, neither of which are Python packages:

```bash
# Java 17 (user-local, no sudo needed)
mkdir -p ~/.local/jdk
curl -fsSL -o /tmp/jdk.tar.gz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk"
tar -xzf /tmp/jdk.tar.gz -C ~/.local/jdk --strip-components=1
export JAVA_HOME=~/.local/jdk PATH=~/.local/jdk/bin:$PATH   # add to shell profile to persist

# Kafka broker, single-node KRaft mode (no Zookeeper needed)
docker run -d --name f1-kafka -p 9092:9092 \
  -e KAFKA_NODE_ID=1 -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e CLUSTER_ID=f1strategy0001 \
  apache/kafka:latest
```

Then, in two terminals:

```bash
# terminal 1 -- start the consumer first, it needs to be listening before the producer sends anything
JAVA_HOME=~/.local/jdk PATH=~/.local/jdk/bin:$PATH .venv/bin/python notebooks/09_streaming_consumer.py 2023_r08

# terminal 2 -- replay a race (speed factor optional, default 50x)
.venv/bin/python notebooks/08_streaming_producer.py 2023_r08 100
```

The live risk/reward comparison doesn't need Kafka/Spark running -- it's a standalone query:

```bash
.venv/bin/python notebooks/10_live_risk_comparison.py 2023_r08 8   # race_id, decision lap
```

### Dashboard

A local web viewer onto the streaming layer above -- pick a race, click Start, watch live probabilities and pit-now-vs-wait recommendations arrive as the race replays. Needs the Kafka broker running (see above); it launches 08/09 itself, so no need to run them manually or set JAVA_HOME yourself.

```bash
.venv/bin/python dashboard/server.py
# open http://localhost:8000
```

## Layout

```
notebooks/   numbered exploration scripts, run in order (00, 01, 02, 03, ...)
dashboard/   local web viewer onto the live streaming system (server.py launches 08/09 and streams their output)
data/        generated Bronze/Silver/Gold outputs + FastF1 cache (gitignored, regenerated by the scripts)
```

`data/` is not checked in — it's fully regenerable by running the notebooks in order, and the FastF1 cache alone is ~180MB for 12 races.
