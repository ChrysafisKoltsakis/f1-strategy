# Ideas to tackle: weather forecast, "is the driver pushing?", "is the driver in a battle?"

Working notes, written 2026-11-07. Nothing here is implemented. The rules in `CLAUDE.md` apply: test each change on two dry races (one repeated, e.g. Canada 2023 `2023_r08` / Spain 2024 `2024_r10`, one new), include an SC race when SC/VSC behaviour is touched, report negative results with race-bootstrap CIs, and choose nothing by looking at the test races. `docs/forward_test.md` freezes the tyre model and settings until it is scored, so write anything new to a separate path (e.g. `data/models/tuned/`) and leave `degradation_model.joblib` and the frozen backup alone.

## 0. What exists today

- **Weather:** the tyre model (`03`) takes `AirTemp, TrackTemp, Humidity, Rainfall` per lap, matched from FastF1's weather feed (about 1 reading per minute; also has `WindSpeed`, `WindDirection`, `Pressure`, unused). In live mode (`12_as_of_context.live_pace_lookup`) the latest observed reading is held flat for every remaining lap. There is no forecast. The backtest uses each lap's own measured weather, which is mild look-ahead.
- **Pushing:** nothing explicit. The classifier has `own_pace_trend`; the tyre model predicts a pooled average pace, not a driver's current effort.
- **Battle:** the classifier has gaps (`gap_ahead_s_*`, `gap_behind_s_*`, trends) and rival tyre ages; the position model (`15`) has overtaking probability by gap and potential-pace difference. There is no explicit "in a battle" flag.

---

## 1. Weather forecast

### What a forecast would change
- Track/air temperature drift over the race (the track usually cools late) shifts degradation.
- Rain arriving or stopping is the large effect: it invalidates dry-tyre pace entirely, and `10` currently just skips wet conditions.

### Step 0: measure the ceiling before building anything
Cheap experiment first. In the backtest, replace each lap's actual weather with (a) the race-start reading held flat (what live mode effectively does), and (b) a naive trend extrapolation. Compare the paired race-time and points difference against the actual-weather run. If (a) vs actual costs nothing, the forecast is not worth much for dry races and the rain case is the only reason to continue.

### Data options (decide which)
1. **Historical forecasts.** Ideal, but free APIs rarely keep what was forecast at the time. Open-Meteo has a historical-forecast API; check if it covers each circuit and date back to 2019. This is the honest way to backtest.
2. **Reanalysis / observed weather** (e.g. ERA5, Open-Meteo archive) as a stand-in "forecast with realistic error". Optimistic: this is hindsight weather, so label it as an upper bound.
3. **Derive from the FastF1 feed itself** as of the current lap: fit a short-horizon trend on the last N minutes of `TrackTemp`/`AirTemp`/`Rainfall`. No new data, fully causal, works live. Probably the best first version.
4. **Teams' own forecasts / radar:** not available.

### How to plug it in
- `predict_pace_grid` already takes per-lap weather (`lap_weather` indexed by lap). A forecast is just a non-flat `lap_weather` for the remaining laps; `live_pace_lookup` is the one place that holds it flat.
- Add a rain-probability input to `10` (which already carries SC/VSC probabilities per lap): chance of rain by lap, and a wet-tyre switch cost. Needs intermediate/wet pace, so first check how many wet laps the data has (`Rainfall > 0`, wet compounds); the tyre model is dry-only in practice.
- Evaluate on races with a real weather change (find them: `Rainfall` flips within the race, or `TrackTemp` range above some threshold). Dry-and-stable races will show nothing, and that is expected.

### Risks
- Few rain races, so wide CIs. Say so.
- Forecast error should be modelled, not assumed perfect, or the gain is overstated (same optimizer's-curse point as research notes 4a).

---

## 2. Is the driver pushing? (from the best lap of the stint)

### Idea
Compare a driver's current lap time to their own best lap on this tyre stint, after correcting for fuel burn and tyre age. If recent laps are well off that best, the driver is managing (tyres, fuel, traffic); if close to it, pushing. The distinction matters for the undercut (needs a pushing car with fresh tyres behind) and for the "fewer stops" finding: teams that manage pace are the ones who make a one-stop work, and the model can't see that (README: "tyre-safety margins, risk").

### Definition to try
- `stint_best` = the minimum of green, non-pit, non-SC/VSC `LapTime` in the stint so far (or a low quantile, to cut single-lap outliers; start with the 2nd-best lap).
- `push_gap` = (this lap's detrended pace) − (stint_best's detrended pace), using `RelativePace` from `02`/`03` (lap time minus field median) so track evolution and fuel burn are mostly cancelled. Raw gap to stint best is confounded by fuel burn (about 0.03s/lap early on), so correct for it, or compare on `RelativePace`.
- Flag `pushing` when `push_gap` is below a threshold, or use the continuous value. Don't fix the threshold by eye on test races: choose on training seasons (e.g. the quantile that separates laps followed by a faster lap from those that are not).

### Complications to check
- A stint's best lap is often early (fresh tyre) or lucky (clear air), so a long stint always looks like it is declining. Needs tyre age in the comparison: use the tyre model's expected degradation between lap `a` and lap `b` and compare observed vs expected.
- Traffic: a driver stuck behind a slower car is not "managing", just held up. Exclude laps with gap ahead under about 1.0s, or take that as the battle flag (section 3) and report separately.
- First/last laps of a stint, in/out laps, SC/VSC and yellow laps must be excluded (already filtered in `02`).
- Lifting and coasting for tyres can't be seen in lap times alone. Sector times (in the laps table) help: a driver who only lifts in one sector looks different from one who is slow throughout. Telemetry (throttle/brake/speed) would be definitive but is a much bigger data pull.

### Where it plugs in
- **Classifier (`05`):** add `push_gap` and its trend to `FEATURE_COLS`. A driver who stops pushing is probably about to pit or has already decided on a longer stint. Needs the causal-confusion audit already in `05`, because a slowing lap can be caused by a pending stop (an in-lap).
- **Tyre model / DP:** a per-driver pace-management factor, with a prior from team and circuit. Possibly the most valuable use, since wear spread (the `1 + 0.45·N(0,1)` factor in `15`) is currently one global number.
- **Position model:** replace "potential pace" offset with a driver's current effort, so a car that is managing is not assumed to be able to attack.

### Test
Does it help out of sample (rolling-origin, 05) or in the paired backtest? Report even if it does not. Also report how often the flag is on and the pushing share by team, to sanity-check that it is not just a team-pace proxy.

---

## 3. Is the driver in a battle?

### Idea
A driver is in a battle when a close car is ahead or behind and the gap persists over several laps, which is what changes strategy: they may pit to undercut, defend, or stay out to hold position.

### Definition to try (all from existing gap columns; the gaps already exist in `05`)
- `in_battle_ahead` = `gap_ahead_s` below about 1.0s (DRS range) for at least k consecutive laps; `in_battle_behind` likewise with `gap_behind_s`. Start with k = 2 and test 1 and 3. Don't tune on test races.
- Better, a **closing-rate** version: gap trend (already `gap_ahead_trend`) plus the potential-pace difference from `15`, so that it separates "stuck behind a faster/slower car" from "about to pass". The position model already estimates the chance of a pass within the lap by gap and potential pace, so that probability is a ready-made continuous battle score.
- Direction and who: is the car ahead/behind on older or fresher tyres? (`rival_tyre_age_*_prev` exists.) A battle with a car on older tyres ahead is an undercut chance; a battle with a fresher car behind is a defend situation.
- Distinguish **real battle** from **traffic** (e.g. a lapped car or a slower car with no chance of passing) by the potential-pace difference.

### Complications
- Gaps under SC/VSC are meaningless (bunched field): mask them.
- Lap-end gaps are coarse: a gap measured at the line misses a pass inside the lap. Sector times or position changes (`Position`) can refine this.
- The lapped cars case: gap to the car ahead on track is not necessarily the car ahead in the race.

### Where it plugs in
- **Classifier (`05`):** `in_battle_*` flags, battle length, and a pass-probability feature as inputs. Several gap features exist already, so the first question is whether the explicit flag adds anything beyond them (likely small). Honest negative result is fine.
- **Policy (`14` `PlanFollower` / `StrategyDecider`):** treat a battle as a reason to deviate from the plan's window (undercut if close to a car ahead on older tyres, don't pit into a queue). This is where it could matter for the green-flag races where the policy currently buys no positions.
- **Pit-loss:** pit-exit traffic. Where the car rejoins relative to others is what makes a stop cost positions. Already approximated in `15`.

### Test
Position-scored backtest (06 `--position` / `--strategy`), both overall and in the 24 green-flag races where the policy currently does not convert time into places.

---

## Suggested order

1. **Battle flag** first: no new data, everything needed is already in `05` and `15`, and it targets the known weak spot (green-flag races).
2. **Push / manage flag**: also from existing columns, but needs the fuel/tyre correction done carefully; likely feeds a per-driver wear factor.
3. **Weather**: run the Step 0 ceiling experiment early since it is cheap. Only build the forecast pipeline (and rain handling) if it shows something, or if wet races are in scope for the thesis.

## Decisions (2026-11-07)

- **Rain strategy is in scope.** Wet-tyre pace and a rain-probability input to `10` are part of the work, not an optional extra.
- **An external weather API is acceptable.** Open-Meteo's historical-forecast API (`historical-forecast-api.open-meteo.com`) responded from this machine for a 2019 date, so it can serve as the backtest "forecast"; the archive API (`archive-api.open-meteo.com`) is reachable too, for the hindsight upper bound. Needs circuit coordinates (`data/circuits.csv` may have them; otherwise add) and a documented source in the thesis. Check that its forecasts really are as-issued before calling the result a forecast.
- **"Pushing" goes in both places:** a classifier feature (`05`) and a per-driver wear/management factor in the tyre/position models.
- **Telemetry is wanted for the push flag.** None is cached yet: `data/fastf1_cache` holds 172 weather files and **0** `car_data` files, so every race would be a fresh pull. Telemetry loading is heavy and counts against the ~500 calls/hour FastF1 limit; races from Oct 2026 on need the user's F1 TV login (README). Plan: prototype on the two test races first, store only per-lap summaries (full-throttle share, lift-and-coast distance, top speed, brake count) in a new `data/silver/` file, and extend to all races only if the prototype separates pushing from managing better than lap and sector times do.

