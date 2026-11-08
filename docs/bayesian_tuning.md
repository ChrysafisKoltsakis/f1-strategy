# Bayesian hyperparameter tuning (not started)

Notes for adding Bayesian optimisation later. Nothing here is implemented; no tuning library is installed (`optuna` is the obvious choice, add it to `requirements.txt` when starting).

## Why it hasn't been done, and what to expect

Every hyperparameter so far is a hand-picked default, never searched. `notebooks/16_model_comparison.py` found that all the tree models tie (tyre MAE within ~0.01s of each other, classifier average precision 0.119-0.123, CIs overlapping), and that a tyre model with no tyre-age term is only 0.008s worse than the deployed one. So the held-out metrics are close to a noise floor, and tuning is unlikely to buy much on them. The more useful questions are on the downstream side (see "Objective" below). Report a null result honestly, like the others here.

## What is there to tune

Current values, so a search has a baseline to beat:

| Model | Where | Current |
|---|---|---|
| Tyre regressor (LightGBM) | `03_ml_degradation_model.py` `new_regressor()` | `n_estimators=300, learning_rate=0.05, max_depth=5, num_leaves=31, min_child_samples=20`, monotone on `TyreLife` |
| Pit classifier (LightGBM) | `05_trigger_classifier.py` `new_classifier()` | `n_estimators=300, learning_rate=0.05, num_leaves=31`, no class weights |
| Era prior | `OLD_ERA_RACES` in 03/05 | `10` (already chosen by experiment, see README table) |
| Policy margins | `14_default_strategy.py` / `15_position_model.py` | 1s cost margin (`PlanFollower`), 0.25 objective points (`PositionDecider`), wear spread 0.45, `CLIFF_S_PER_LAP2 = 0.05` |

Suggested search ranges for the LightGBM models: `learning_rate` 0.01-0.2 (log), `num_leaves` 8-128, `max_depth` 3-10, `min_child_samples` 5-200 (log), `n_estimators` 100-800 (or early stopping), `reg_lambda` 0-10, `subsample`/`colsample_bytree` 0.5-1.

## Rules to keep it honest

- **Nothing is chosen by looking at the test races** (CLAUDE.md). Tune only on races before `TEST_FROM_RACE = '2025_r20'`. The forward-test races (2026 r17-r19, and r20-r23 once scored) are out of bounds too.
- **Forward test freeze.** `docs/forward_test.md` freezes the tyre model and settings until it is scored; do not retrain or overwrite `data/models/degradation_model.joblib` or the frozen backup `degradation_model_frozen_r16.joblib` in the meantime. Write tuned models to a separate path (e.g. `data/models/tuned/`) and compare there.
- **Folds must match the use.** Tyre model: leave-one-circuit-out (as 03), or grouped 5-fold by circuit inside the search because it needs far fewer fits (`16_model_comparison.py tyre --cv 5`). Classifier: rolling-origin by season (as 05), with the threshold chosen on the validation block. Never split by row.
- **Compare trials on identical folds.** In `16`'s 5-fold run, MAE varies about 0.04s between folds and up to 0.008s between fold assignments, while the models differ by about 0.01s. Fix one assignment for the whole search, compare paired differences, and confirm the winner with leave-one-circuit-out: the two schemes agreed on the only clear gap but not on every smaller one.
- **Nested evaluation.** Tuning on the same folds you report on overstates the gain. Tune on earlier seasons and report on a later one, or use an inner/outer split by circuit or season.
- **Compare with CIs.** Use the same race/circuit bootstrap as 16 (paired difference vs. the deployed model). A gain whose CI spans zero is a tie.
- **Keep the monotone constraint** on the tyre model unless it is being tested explicitly; it matters downstream for the censored-tyre-data reasons in 03's comments.

## Objective

Tuning hyperparameters to minimise MAE or maximise average precision is the cheap option, but 16 shows those barely move. Better, in order of cost:

1. Tyre model: leave-one-circuit-out MAE (minutes per trial). Weak signal.
2. Classifier: rolling-origin average precision plus calibration error as a constraint (the probabilities drive live decisions, so keep ECE low).
3. Downstream: tune against `06`'s time-scored rolling backtest ("vs real" median seconds), as 16's `downstream()` does. Slower, but it is what the thesis claims. The position-scored backtests (`06 --position --strategy`) take 1-2 hours per run and are not practical inside a search.

## Sketch

```python
import optuna
def objective(trial):
    params = dict(learning_rate=trial.suggest_float('lr', 0.01, 0.2, log=True), ...)
    # fit with leave-one-circuit-out (reuse 03/16), return weighted MAE
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=50)
```

Reuse `16_model_comparison.py`'s `Est` wrapper, `load_silver()` and `classifier_frame()` so folds and features match the existing comparison. Log every trial (params and score) to `data/model_comparison/tuning_*.csv`, and report the number of trials, since the best of many trials is optimistically biased.

## Results

(none yet)
