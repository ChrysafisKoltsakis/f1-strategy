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

## Results: 2026_r17–r19 (scored 2026-11-02, after the red-flag fix)

50 driver-races in 3 races. Mean per driver-race vs the teams' real strategies, all re-raced through the same simulator with the same random draws. The 95% CIs are a bootstrap over only 3 races, so treat them as rough.

| Strategy | Points vs real | Places vs real |
|---|---|---|
| Default plan (Pirelli) | +0.37 (−0.19 to +0.83) | −0.13 |
| Time-based live policy | +0.74 (+0.49 to +0.91) | −0.47 |
| Position-aware window policy | +0.74 (+0.43 to +0.95) | −0.48 |
| Strategy policy (unguarded = guarded here) | **+0.82 (+0.40 to +1.06)** | −0.56 |

- **Per race, strategy policy, points per driver vs real:** Singapore +0.40, United States +1.01, Mexico City +1.05. All three go the same way.
- **Risk:** points SD +0.10 vs real (CI −0.03 to +0.23), worst-10% points +0.35, P(no points) −2 pp. Still ahead at λ = 3 (+0.53).
- **What this does not test:** Pirelli's plan was a one-stop in all three races, so the strategy policy never had a stop it could cut. Guarded and unguarded are identical, and in Mexico City it *added* stops (3 stops for 14 of 17 cars). The forward test confirms the policy's timing and SC/VSC side on unseen races. The development finding that its extra value comes from fewer stops is not tested here.
- **Development reference after the red-flag fix (2022–2026_r16, 1215 driver-races):** strategy policy +1.03 points (CI 0.78–1.31), guarded +0.43, time-based +0.40. Simulator coverage of real finishes at `WEAR_SD = 0.45`: 79% (11%/10% below/above).

## Deviations log

- **2026-11-02: FastF1 3.8.3 → 3.9.1 in `.venv`.** Needed to sign in to F1's live-timing archive, which has required an F1 TV login since ~15 Oct 2026. The races before 2026_r17 were parsed with 3.8.3, the forward-test races with 3.9.1. No model, setting or code in 02–15 changed.
- **2026-11-02: red-flag bug fixed in 15's simulator.** The first forward-test scoring put every driver 1st in Mexico City (2026_r19). The race had a red flag on laps 44–45, about 55 minutes stopped. The simulated car was timed from the field's median *lap time*, which excludes the stoppage, while the other cars' session times include it, so it "won" by about 2,200 s. The field is now timed by its median change in session time per lap. Races without a red flag move by under 1 s in total (Canada 2023: 0.9 s on one SC lap; Spain 2024: 0). Three development races in the backtest set had red flags (2023_r20, 2024_r08, 2026_r06; validation error there 4.76 places before the fix). Their policy decisions and the full development scoring were re-run with the fix, so the development reference above is updated below. No setting changed; `WEAR_SD` stays 0.45.
