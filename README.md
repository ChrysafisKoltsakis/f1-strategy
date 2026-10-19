# F1 Strategy

A data pipeline and strategy optimization engine for Formula 1 pit-stop decisions, built as a thesis project.

Ingests multi-season F1 timing/telemetry data (via [FastF1](https://github.com/theOehrly/Fast-F1)), builds per-circuit tyre-degradation and pit-loss models, and works toward a real-time strategy recommendation system: given the live state of a race, should a team pit now or hold.

## Status

Currently in the experimentation phase. Built so far, covering the full scope-guardrail MVP:

- **Bronze-layer ingestion** (`notebooks/01_pull_bronze_data.py`) — pulls race sessions (laps, weather, race-control, results) via FastF1 for 2019-2026 and lands them as Parquet, plus official per-round championship standings and sprint results from [Jolpica](https://github.com/jolpica/jolpica-f1) (the Ergast successor, via FastF1's built-in client). Idempotent — safe to re-run as new seasons/rounds become available. Currently 168 races (all of 2019-2025, 2026 through round 16). 2018 is left out: its absolute compound names (ultrasoft, supersoft, ...) map to soft/medium/hard differently at every race; before 2018 there's no tyre data at all.
- **Cleaning + degradation/pit-loss modeling** (`notebooks/02_fixed_degradation_pitloss.py`) — filters pit laps, standing-start laps, SC/VSC-restart-affected laps, and non-green-flag laps; detrends lap time against the field's per-lap median pace to separate tyre degradation from track evolution; estimates per-circuit green and SC/VSC pit-loss constants, plus a per-team pit-loss adjustment pooled across races.
- **ML degradation model** (`notebooks/03_ml_degradation_model.py`) — LightGBM model predicting relative pace (vs. field median) from tyre age, compound, weather, and circuit, validated with leave-one-circuit-out cross-validation (~0.75s MAE over 2019-2026, with 2019-2021 down-weighted -- see below). Pace can only get worse with tyre age (monotone constraint). Race data are censored -- teams change tyres before they fall away, so observed wear even slows late in stints -- and two corrections that forced a drop-off from the data were rejected by the teammate validation (slope 0.93 -> 0.18 and 0.47): the data confirm the model where teams have run a strategy but can't prove an untried one (e.g. a one-stop in Spain 2024, where every team two-stopped) would fail. That knowledge comes from the pre-race default strategy instead; the tyre model keeps only a safety net against absurd extrapolation (drop-off beyond the longest stint ever run on that compound). Persisted (`data/models/degradation_model.joblib`) and actually consumed by the DP baseline, backtest, and live risk-comparison tool via a precomputed per-race (compound, lap, tyre age) prediction grid — captures non-linear/cliff wear and pools across circuits/seasons, unlike the simpler per-race linear fit those tools originally used.
- **DP hindsight-optimal baseline** (`notebooks/04_dp_baseline.py`) — dynamic program computing the mathematically optimal pit strategy per race per team, used as the evaluation ruler. Any stop may fit a fresh set of any compound (including the one already on, so e.g. medium → hard → hard is reachable), subject to the two-compound rule. Each driver's real strategy is replayed through the same model for comparison, counting a stop only when the tyre set actually changes (stops onto used sets included; pit-lane passes with no tyre change and free red-flag changes excluded), and lapped drivers are compared against the optimum over their own completed distance.
- **Trigger/strategy classifier** (`notebooks/05_trigger_classifier.py`) — LightGBM imitation-learning model predicting, from everything known when a lap ends, whether a driver pits at the end of the *next* lap (one lap ahead, so a live call arrives before the stop rather than after it), from race state (on-track gaps, tyre age, championship-rival state, momentum/trend features), with a causal-confusion audit on the feature set. Trained **without class rebalancing**, so its probabilities are calibrated (held-out calibration error 0.007 vs. 0.249 with `class_weight='balanced'`); the decision threshold is chosen on the last training season, never on the test races. Predicting a lap ahead is harder than predicting the lap itself (average precision 0.138 vs. 0.210 on the held-out races) -- the honest cost of a call that's usable in time. Also evaluated rolling-origin: each season 2022-2026 predicted by a model trained only on earlier races (average precision 0.12-0.15, calibration error ≤0.014 every season).
- **Championship-rival computation** (`notebooks/07_championship_rivals.py`) — official standings before each race (previous round's Jolpica standings plus that weekend's sprint points) and nearest drivers'/constructors' rival, feeding the classifier. Cross-checks that previous standings + sprint + race points reproduce the official standings after every round.
- **3-way backtest evaluation** (`notebooks/06_backtest_evaluation.py`) — actual vs. classifier-driven vs. DP-hindsight-optimal strategy, replayed through the same lap-time model over each driver's own distance, with the race's real safety cars priced everywhere (including the hindsight optimum; see 04 and the strategy-cost validation below), and bootstrap 95% intervals over races. On the rolling-origin predictions (78 races 2022-2026, 1252 driver-races), real teams lose a median 11.2s to the hindsight optimum. The classifier's multi-stop ranking rule is a paired 3.4s better than the teams (95% CI 2.0-5.0s), but it chooses laps after seeing the whole race's probabilities; its live-usable version, the threshold rule, is 2.1s *worse* (CI 1.1-3.7s). The **live policy** (below) is usable live and is 4.4s better than the teams (CI 2.7-6.2s, better in 72% of driver-races; 2.5s better, CI 0.1-5.1s, in the 24 races without SC/VSC). The default plan alone is 0.9s better (CI 0.0-2.2s). Scored in finishing position instead (see the position model), the policy's edge holds overall but not in green-flag races.
- **Default strategy and live policy** (`notebooks/14_default_strategy.py`) — the plan a car starts the race with: Pirelli's published pre-race prediction (`reference/pirelli_strategies.csv`, collected by hand with sources for the 2025-26 test races), else what finishers did at the circuit before (as of the race, recency-weighted: modal stop count and compounds, median stop laps with interquartile windows). Race data can't show an untried strategy failing (see the ML degradation model above), so this carries the teams' revealed knowledge of how many stops a circuit needs. `PlanFollower` is the live policy: follow the plan, and leave it only when the expected-cost model's margin beats its own error (1s). Under an SC/VSC it pits if that is clearly cheaper; under green it moves a stop only inside the plan's window. A stop that comes early or late re-spaces the rest of the plan to keep its stint proportions. Replayed lap by lap in 06 and run live in 09 (results above). Pirelli and history defaults are level over the 12 test races with both (median −0.3s for Pirelli, CI −3.1 to +5.4); Pirelli helps most at the first race of new rules (Australia 2026).
- **Position model** (`notebooks/15_position_model.py`) — what a strategy does to finishing position and points, which race time alone can't show (the undercut, holding a place a slower car can't take back). An overtaking model fitted on 56k close-following green laps: chance of getting past the car ahead within the lap by gap, the two cars' *potential* pace difference (pace offset + tyre model; observed lap times of a car stuck behind just repeat the car ahead's), tyre-age difference and circuit (Monaco hardest, then Montréal, Zandvoort, Imola; Austin and Shanghai easiest). Each season predicted from earlier races only: AUC 0.92-0.95, well calibrated, and per-circuit terms help every season. A counterfactual race then re-runs one car on a candidate strategy against every other car's real laps: free position changes through the pit lane, passes on track drawn from the overtaking model in both directions (fail and you're held 0.58s behind, the measured median), SC/VSC freezing the order, Monte Carlo over the draws. Re-racing every finisher's real strategy reproduces their real finishing position to 1.54 places on average (1.71 by race time alone). Position-scored backtest over the rolling races (1212 driver-races, same draws for every strategy): the live policy finishes 0.47 places better than the teams' real strategies (CI 0.16-0.80) and scores 0.52 more points (CI 0.04-1.02); but in the 24 green-flag races its 2.5s time advantage buys no positions (-0.22 points, CI -0.75 to +0.28) -- time saved without regard to track position doesn't convert. Objective with the championship rival (points + 1 x points gap to the nearest rival): +0.99 (CI -0.06 to +2.06). The other cars keep their real strategies and the simulator knows their real future laps, so it scores strategies after the fact. A **position-aware policy** decides with it live: inside a planned stop's window, every stop lap from now to the window's end is raced against a *forecast* field (each other car's laps so far, then its potential pace on the default plan re-spaced to its real stops), and pitting now must beat the best later lap by 0.25 objective points (`PositionDecider`; `06 --position` offline; `STRATEGY_MODE = 'window'` in `09`, the live default). It changes the strategy in 63% of driver-races but scores level with the time-based policy against the real field (+0.48 points vs the teams, CI 0.03-0.97, against +0.52), and in green-flag races both stay behind the teams (-0.35 and -0.22 points, CIs spanning zero). Given the other cars' real future laps instead of a forecast it does no better (-0.20), so the limit isn't forecasting the field but what the policy may change: only stop timing inside the default plan's windows, while in green-flag races the default plan itself is 0.36 points per driver behind the teams. Choosing among whole strategies is the next lever: `StrategyDecider` re-plans the rest of the race (0-3 more stops at balanced timing, shifted earlier to undercut or later to overcut, compounds by the tyre model) every few laps and near each stop (`06 --strategy`; `STRATEGY_MODE = 'strategy'` in `09`). Offline (1215 driver-races) it finishes 0.95 places better than the teams (CI 0.71-1.23) and scores 1.14 more points (CI 0.81-1.51), +0.98 in green-flag races -- but not credibly: 45% of that gain comes from the 31% of driver-races where it cuts stops (+2.21 points vs the time-based policy there, against +0.09 where it keeps the stop count), i.e. from long stints the tyre model can't price because teams never run tyres to failure (censoring, see the ML degradation model). The simulator scores with the same tyre model, so it can't catch this. Hence the live default stays `'window'`; a guard against fewer stops than the default plan, and a check of fewer-stop strategies against drivers who really ran them, come next.
- **Streaming layer** (`notebooks/08_streaming_producer.py` + `09_streaming_consumer.py`) — replays a historical race's laps through Kafka in real (accelerated) chronological order, consumed by Spark Structured Streaming, which re-invokes the trained classifier after each new lap for a live probability of pitting at the end of the next lap. Crossing the threshold opens a pit window a lap ahead of the stop (same lap 3 to n−2 window as the backtest), re-evaluated every lap until the car stops; an earlier peak-detection rule flagged stops a lap after the car was already in the pit lane. In Spain 2024 (no safety car), 36 of 42 real stops had a call on the board for that exact lap before it happened. Features use the scheduled race length, so a partial live race isn't mistaken for its final lap. Every lap a window is open, it also runs 10's pit-now-vs-wait comparison live, using only laps observed so far (a causal field-pace forecast, not real future laps) plus the historical per-lap SC/VSC rate, so the printed output is a genuine live recommendation -- probability, cost, and risk together -- not just a probability. SC/VSC deployments are also streamed, on a second topic at the moment race control announced them, and trigger an immediate "pit at the end of this lap, or stay out?" call for every running driver: lap records arrive only when a lap ends, by which time a car pitting under the safety car is already in the pit lane. In Canada 2023, every clear call at the lap-8 VSC matched the teams (all stayed out), and at the lap-12 safety car 8 of the 14 "pit NOW" calls matched; the 6 who stayed out had started on hards and ran long for track position, which the per-car model can't see. The consumer also runs the live policy per driver: it announces the default plan, then gives a pit-or-stay-out call for the end of each lap inside a planned stop's window, kept in step with the stops the real car makes (in Canada 2023, Verstappen's calls were laps 12 under the SC and 42, exactly his real stops). The literal implementation of Psaltiras's declared future-work chapter this thesis is built around.
- **Live risk/reward comparison** (`notebooks/10_live_risk_comparison.py`) — at a mid-race decision point, compares pitting at the end of this lap against staying out, by expected race time from the *current* live state. Every later lap carries the historical chances (from earlier races only) of a full SC and of a VSC being deployed then, making a stop that lap cheap at that kind's price, and the plan re-optimizes from whatever happens; a full SC also erases ~80% of the time lost before it, weighted into each lap's cost in expectation (exact for a DP, since deployments don't depend on our decisions). Published expected-value and simulation models price cheaper SC/VSC stops but not this bunching — so waiting for a cheap window is only worth what history says it's worth, including the risk it comes too late. Margins under 1s are reported as too close to call. Skips wet conditions (anyone on intermediates/wets in the last 3 laps) with a stated reason, waives the two-compound rule for drivers who've used wet tyres, and counts compounds already used towards it. Also exposes a causal, live-safe version of its field-pace forecast that 09_streaming_consumer.py calls directly.
- **Per-lap risk/reward report** (`notebooks/11_pit_lap_risk_report.py`) — reports the classifier's top candidate pit laps for a driver side by side with their model confidence *and* actual predicted time cost, instead of collapsing to one recommendation. Flags when the top two candidates are a statistical toss-up (motivated by a real case under an earlier model version: two laps separated by a 0.005 probability margin, where the "losing" lap was actually 5 seconds faster -- the retrained model no longer reproduces that particular case).
- **Strategy-cost validation** (`notebooks/13_strategy_cost_validation.py`) — checks that the pace model prices strategies correctly, using teammates (same car and race, often different strategies) as a natural experiment: the model's predicted time difference between teammates' real strategies vs. their real time difference over common laps, with each pair's driver-skill gap removed using laps where both ran the same tyres. Slope of real on predicted difference 0.91 overall (95% CI 0.68-1.14, 234 skill-adjusted pairs over 92 races) and 0.96 (CI 0.70-1.18) for strategies without a stop under SC/VSC — right-sized, not overstated (the optimizer's-curse concern). Safety cars needed three pieces, each measured from our own data: separate SC and VSC stop prices (0.21 and 0.68 of a green stop, over 857 individual stops), gap compression when a full SC comes out (gaps shrink to 0.21x, stable across seasons, from 646 car observations in 57 races -- the statistical equivalent of Heilmeier et al.'s "ghost" safety car), and only one cheap stop per SC/VSC period. That lifted the slope for pairs with a stop under SC/VSC from 0.44 to 0.81. (Race control writes 'VSC DEPLOYED' rather than 'VIRTUAL SAFETY CAR DEPLOYED' from mid-2026; VSCs are now recognised by either wording.)
- **As-of-race context** (`notebooks/12_as_of_context.py`) — everything the live tools (09, 10, the dashboard) use, built only from races strictly before the one being replayed: the trigger classifier and tyre model are retrained on earlier races (cached per race under `data/models/as_of/`, ~2s to build), pit loss comes from earlier visits to the circuit (or the median of all earlier races on a first visit) rather than the race's own stops, team pit-loss adjustments and SC/VSC rates use earlier races only, and the tyre model sees only the weather observed so far. So a replay shows what a team could have known that morning, not a model that has already seen the race. A race with no earlier history (2022 round 1) can't be replayed; one with fewer than 20 is flagged as thinly trained. The offline backtest is unaffected — it already holds out its test races.
- **Live dashboard** (`dashboard/server.py`) — a local web viewer onto the real streaming system above: launches 08/09 as subprocesses for a chosen race and speed, and streams every event they produce (per-lap probability, pit windows, pit-now-vs-wait cost comparisons, the default plan and the policy's calls) to the browser over Server-Sent Events. A view onto the genuine Kafka/Spark pipeline, not a reimplementation of it. Functional first pass — visual design deliberately deferred.

Not yet built: proper design polish on the dashboard; wet-race (rain/intermediate) strategy is out of scope for the cost/risk comparison everywhere in this project, not just the dashboard.

## Using the 2019-2021 seasons (previous rules)

Everything per-circuit (pit loss, tyre model, SC rates, default plans) is keyed by circuit, not event name: the same track runs under several names (Styrian/Austrian, 70th Anniversary/British, Brazilian/São Paulo, and 2026's Barcelona GP is the old Spanish GP's circuit), while the 2026 Spanish GP is a new circuit (Madrid) and the 2026 Bahrain GP ran in Kuala Lumpur. `01` writes each race's FastF1 schedule location to `data/circuits.csv` and `02`'s `circuit_of()` turns it into a circuit key.

2026 brought new chassis and power-unit rules and narrower tyres. Predicting 2026 rounds 5-16 from earlier races (models trained before rounds 5, 9 and 13, each predicting the next four), neither model benefits from treating 2026 as a new era: down-weighting pre-2026 data made both worse (classifier average precision 0.096 at *k* = 5 rising to 0.135 with no down-weighting; tyre-model error lowest with none), and a `regs_2026` flag made both slightly worse (classifier 0.135 vs 0.145 without; tyre model 1.063s vs 1.043s MAE). Both work relative to the field, which a rules change every car makes at once mostly leaves alone. 2026 still counts as its own era for usable tyre life and for which history the default plan draws on, and Pirelli's prediction covers the first 2026 visits.

2022 brought ground-effect cars and 18-inch tyres, so 2019-2021 data is only partly relevant. The tyre model and trigger classifier get a `regs_2022` era flag, and all of 2019-2021 together is weighted as if it were *k* current-era races (`OLD_ERA_RACES` in 03/05): a fixed-strength prior that current data outweighs as it accumulates. Pit loss and SC/VSC rates use every earlier season unweighted, since pit lanes and incident rates barely changed. *k* was chosen by experiment, comparing 2022-only training (*k* = 0), *k* = 5, 10, 20 and no down-weighting, both on the thesis test set and as-of across 2022 (each race trained only on races before it):

| | 2022 only | k=5 | k=10 | k=20 | unweighted |
|---|---|---|---|---|---|
| Tyre MAE, 2022 rounds 1-7 | 1.048s | 0.861s | 0.850s | 0.849s | 0.849s |
| Tyre MAE, 2022 rounds 8-15 | 0.765s | 0.716s | 0.711s | 0.722s | 0.723s |
| Tyre MAE, test set | 0.917s | 0.902s | 0.900s | 0.904s | 0.901s |
| Classifier avg. precision, 2022 rounds 1-7 | 0.139 | 0.219 | 0.239 | 0.234 | 0.220 |
| Classifier avg. precision, 2022 rounds 16-22 | 0.151 | 0.173 | 0.191 | 0.213 | 0.206 |
| Classifier avg. precision, test set | 0.218 | 0.216 | 0.227 | 0.211 | 0.206 |

Older data helps most early in 2022 and fades as the season goes on; *k* = 10 is best or within noise of best throughout, so both models use it. The test-set differences come from one test window and are small.

The thesis test set is pinned to races from 2025 round 20 on (`TEST_FROM_RACE` in 05), so adding seasons or new 2026 rounds doesn't change what the evaluation is scored on.

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

## Rebuilding and running

`data/` is not in git, so on a new machine everything is regenerated. Order (each step reads the previous ones' outputs):

```bash
.venv/bin/python notebooks/01_pull_bronze_data.py        # FastF1/Jolpica -> data/bronze, data/circuits.csv (re-run hourly past the API limit)
.venv/bin/python notebooks/07_championship_rivals.py     # official standings -> data/championship_rivals.csv
.venv/bin/python notebooks/02_fixed_degradation_pitloss.py   # pit losses, SC/VSC stops, gap compression
.venv/bin/python notebooks/03_ml_degradation_model.py    # silver laps + tyre model (data/models/degradation_model.joblib)
.venv/bin/python notebooks/04_dp_baseline.py             # hindsight-optimal DP + real-strategy replay
.venv/bin/python notebooks/05_trigger_classifier.py      # classifier features, pinned-test + rolling predictions
.venv/bin/python notebooks/06_backtest_evaluation.py     # time-scored backtest (add --position --strategy for the position-aware policies, ~1-2 h)
.venv/bin/python notebooks/13_strategy_cost_validation.py    # teammate validation of strategy pricing
.venv/bin/python notebooks/15_position_model.py          # overtaking model, simulator check, position-scored backtest (reads 06's output)
```

`14_default_strategy.py` (default plan, `PlanFollower`) and `12_as_of_context.py` (as-of models, cached per race under `data/models/as_of/`, delete that folder after retraining 03/05) are libraries used by the others; `14` also prints plans for given races. `15` writes `data/overtaking_events.csv` on first run (~2 min); delete it to rebuild after changing the tyre model.

Live modes: `STRATEGY_MODE` at the top of `09_streaming_consumer.py` -- `'time'` (cost model in the plan's windows), `'window'` (position model in the windows; the default), `'strategy'` (whole-race re-planning; see the caveat in the position-model section).

Where things stand and what's next: `docs/research_notes.md`, section 4f (the late/few-stops bias and the default plan) and section 6 (numbered next steps, with the latest results).

## Layout

```
notebooks/   numbered exploration scripts, run in order (00, 01, 02, 03, ...)
dashboard/   local web viewer onto the live streaming system (server.py launches 08/09 and streams their output)
docs/        research notes (related work, methods, risks, results, next steps) and references.bib
reference/   hand-collected inputs with sources (Pirelli's pre-race strategy predictions)
data/        generated Bronze/Silver/Gold outputs + FastF1 cache (gitignored, regenerated by the scripts)
```

`data/` is not checked in — it's fully regenerable by running the notebooks in order (`01` also writes `data/circuits.csv`, the circuit key everything per-circuit uses), and the FastF1 cache alone is ~1100MB for 168 races. A full rebuild needs more API calls than FastF1's 500-per-hour limit allows: `01` reports the remaining races as failed once it's hit (or exits outright, if the limit lands on a season-calendar fetch), so re-run it after an hour and it resumes where it stopped.
