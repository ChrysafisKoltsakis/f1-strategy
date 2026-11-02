# Forward test: 2026 rounds 17 onwards

Every result before this was computed on races that existed while the models and policies were being built. This test scores them on races nobody looked at during development. The protocol below was fixed and committed **before** the race data for these races was pulled (2026-11-02).

## Races

- **Already run, data not yet pulled:** 2026_r17 Singapore (11 Oct), 2026_r18 United States (25 Oct), 2026_r19 Mexico City (1 Nov).
- **Still to run:** 2026_r20 São Paulo (8 Nov), 2026_r21 Las Vegas, 2026_r22 Qatar, 2026_r23 Abu Dhabi. Each gets its Pirelli entry before its data is pulled. For São Paulo, the default plan and the strategy policy's planned stops for every car are also written down before the race.

## Frozen

- **Code:** the commit that adds this file. Nothing in 02–15 that affects decisions or scoring changes until the test is scored.
- **Default plans:** `reference/pirelli_strategies.csv` entries for 2026_r17–r19, Pirelli's first-listed option, collected 2026-11-02 (notes in the file).
- **Tyre model:** `data/models/degradation_model.joblib`, trained on races up to 2026_r16 (sha256 `b83475c8d16148a4…`). **03 is not re-run.**
- **Overtaking model inputs:** `data/overtaking_events.csv` (sha256 `610ddd3299a04e9d…`). The 2026 pass model is fitted on races before 2026 only.
- **Tyre-wear uncertainty:** `WEAR_SD = 0.45` (calibrated on 2022–2026_r16), `data/wear_ratios.csv` (sha256 `e54e0fa8d51edf1f…`).
- **Policy settings (15):** `RIVAL_WEIGHT = 1.0`, `DECISION_SIMS = 16`, `POSITION_MARGIN = 0.25`, `PLACE_TIE_BREAK = 0.1`, `MAX_STOPS = 3`, `TIMING_SHIFTS = (-3, 0, 3, 6)`, `REPLAN_EVERY = 4`, `NEAR_STOP = 3`, `MIN_STINT = 8`, `N_SIMS = 60`. Plus `CLOSE_CALL_S = 1.0` (10, 14), `PRIOR_ERA_RACES = {2022: 10, 2026: None}` (02), `RECENCY_HALF_LIFE_YEARS = 1.0` (14).

## What is re-run

- **01, 07:** pull the new races and championship standings.
- **02:** each race's own pit-loss and SC/VSC constants, used for scoring as in every earlier backtest.
- **04:** hindsight-optimal DP and real-strategy replay, with the frozen tyre model.
- **05:** rebuilds features. Its 2026 rolling predictions come from a model trained on races before 2026, so the new races aren't used to train anything.
- **06** (`--position --strategy --strategy-guarded`) and **15**'s position-scored backtest, both on the new races only.

## What is reported

For each new race and pooled, against the teams' real strategies (all re-raced through the same simulator, same random draws):

- expected finishing position, points, and points + rival gap;
- for: the default plan, the time-based live policy, the position-aware window policy, and the strategy policy unguarded and guarded (no fewer stops than the default plan);
- risk: SD and worst-10% of points, P(no points);
- for the strategy policy, which stop-count changes it made, and whether any real driver in that race ran that few stops.

Development-set reference (2022–2026_r16, 1215 driver-races): strategy policy +1.10 points per driver-race vs the teams (CI 0.82–1.41), guarded +0.47, time-based +0.47.

With 3–7 races the test can confirm or flag the development result, not establish it on its own; results are reported whichever way they go.
