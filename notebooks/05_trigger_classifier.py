"""
Phase 4 -- trigger/strategy classifier (imitation learning). The hard,
novel part of the thesis: predict whether a team pits on a given lap, by
learning from what real teams actually did across every race pulled (2022-2026),
rather than a hand-built hazard-rate/Bayesian trigger model -- see the
project's ML-not-stats framing decision. GBM-first (LightGBM), not a
sequence model, per the build plan.

Label: given everything known when lap L ends, will this driver pit at the
end of lap L+1 (label_pit_next)? Predicting one lap ahead is what makes the
live system usable: a lap's record only arrives when the lap ends, so a
model of "pits at the end of THIS lap" fired after the car was already in
the pit lane. All features of row L are known when lap L ends, including
this lap's own position/gaps and whether the car pitted at its end. Positive rate is small (~3.4% of laps), but the
model is deliberately trained WITHOUT class rebalancing: rebalancing (class
weights, SMOTE) doesn't improve how well rare events are ranked but inflates
the predicted probabilities (Van Den Goorbergh et al. 2022; Carriero et al.
2024 -- see docs/research_notes.md), and these probabilities drive live
decisions. The imbalance is handled where it belongs, in the decision
threshold, chosen on a validation season inside the training data (never
on the test races). Balanced weighting and SMOTE are still trained, for
comparison only.

--- Causal-confusion audit (de Haan et al., "Causal Confusion in
Imitation Learning") ---
Two feature groups are treated differently on purpose:
  1. Own tyre age/compound and track status on THIS lap are safe to use
     as-is: they describe the state the driver is racing under *before*
     any pit decision for this lap is made, and track status is exogenous
     (no individual driver's pit call changes whether the Safety Car is
     out). Not affected by the label.
  2. Position and gaps-to-rivals on THIS lap are NOT safe to use as-is --
     if a driver pits during this lap, their own recorded Position/gap
     for this lap already reflects the time lost in the pit lane, so it
     would partly encode the outcome of the decision rather than a cause
     of it (classic causal-confusion setup: the model could learn "large
     gap opening up" as a *symptom* of already having pitted). These
     features are therefore computed from the *previous* lap's state
     (lagged by one lap) instead.
None of the surveyed F1-specific pit-decision papers (Fathi et al.,
Heilmeier's VSE, Deep-Racing -- see the prior-art scan) mention auditing
for this, so beyond correctness it's a legitimate rigor claim.

Must split by race, not by row, for train/test -- otherwise the model
can cheat by seeing other laps from a race it's evaluated on.

Run with: python notebooks/05_trigger_classifier.py
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             precision_recall_curve, average_precision_score, brier_score_loss)
from imblearn.over_sampling import SMOTENC
import importlib.util

_spec = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpl)

BRONZE_DIR = Path('data/bronze')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
TREND_WINDOW = 3  # how many non-blue-flagged prior laps to use for trend features
TREND_LOOKBACK = 6  # how far back to search to find TREND_WINDOW clean laps


# Seasons the classifier learns from (features are still built for every
# season, e.g. for 12_as_of_context.py's circuit SC/VSC rates), and the
# first race of the held-out test set. The test set is pinned rather than
# "the most recent 20% of races" so adding older seasons, or new 2026
# rounds, doesn't silently change what the thesis evaluation is scored on.
TRAIN_FROM_SEASON = 2019
TEST_FROM_RACE = '2025_r20'
LABEL = 'label_pit_next'   # pits at the end of the NEXT lap -- see docstring

# Older regulation eras are down-weighted as a fading prior: 02's
# era_weights / PRIOR_ERA_RACES, shared with the tyre model.
era_weights = dpl.era_weights

FEATURE_COLS = [
    # Regulation-era flag (02's era_flags). No regs_2026: predicting 2026
    # rounds 5-16 from earlier races, the classifier without it was better
    # (average precision 0.145 vs 0.135), as was the tyre model (see 03).
    'regs_2022',
    'own_tyre_age', 'own_compound', 'lap_number', 'race_progress', 'laps_remaining',
    'sc_vsc_active', 'yellow_active', 'n_compounds_used_so_far', 'n_stops_so_far',
    'position_prev', 'gap_ahead_s_prev', 'gap_behind_s_prev',
    'rival_tyre_age_ahead_prev', 'rival_tyre_age_behind_prev', 'circuit_sc_prob',
    'own_pace_trend', 'gap_ahead_trend', 'gap_behind_trend',
    'own_blue_flagged_this_lap', 'n_blue_flag_laps_recent', 'n_recent_pit_activity',
    # This lap's own state: known when the lap ends, i.e. before the
    # next-lap decision being predicted (excluded when the label was "pits
    # at the end of THIS lap", where it would have encoded the outcome).
    'position_now', 'gap_ahead_s_now', 'gap_behind_s_now', 'pitted_this_lap',
    'champ_rival_gap_s_prev', 'champ_rival_tyre_age_prev', 'champ_rival_pitted_recently',
    'team_rival_gap_s_prev', 'team_rival_tyre_age_prev', 'team_rival_pitted_recently',
]


# The decision threshold (and any recalibration) is chosen on the last
# VALIDATION_RACES races of whatever the model is trained on -- one season.
VALIDATION_RACES = 24


def new_classifier(balanced: bool = False) -> lgb.LGBMClassifier:
    """The deployed setup (no class rebalancing) -- shared with
    12_as_of_context.py so its as-of models match. balanced=True is the
    previous class_weight='balanced' setup, kept for comparison."""
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight='balanced' if balanced else None, random_state=0, verbosity=-1,
    )


def split_validation(rows: pd.DataFrame) -> tuple:
    """(fit rows, validation rows): the validation block is the last
    VALIDATION_RACES races (fewer when there's little history), so the
    threshold is chosen on races later than the ones the model fit on --
    the same forward-in-time relationship as training vs. test."""
    races = sorted(rows['race'].unique())
    n_val = min(VALIDATION_RACES, max(1, len(races) // 4))
    val_races = set(races[-n_val:])
    return rows[~rows['race'].isin(val_races)], rows[rows['race'].isin(val_races)]


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1 -- applied to validation predictions only."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def fit(model, rows: pd.DataFrame):
    rows = rows.dropna(subset=[LABEL])
    model.fit(rows[FEATURE_COLS], rows[LABEL].astype(int), sample_weight=era_weights(rows['race']),
              categorical_feature=['own_compound'])
    return model


def train_with_threshold(rows: pd.DataFrame, balanced: bool = False) -> tuple:
    """(model, threshold, validation predictions): threshold chosen on the
    validation block with a model fit on the rest, then the model is refit
    on all rows so the most recent season isn't wasted. Shared with
    12_as_of_context.py."""
    rows = rows.dropna(subset=[LABEL])
    fit_rows, val_rows = split_validation(rows)
    val_proba = fit(new_classifier(balanced), fit_rows).predict_proba(val_rows[FEATURE_COLS])[:, 1]
    threshold = best_f1_threshold(val_rows[LABEL].values, val_proba)
    return fit(new_classifier(balanced), rows), threshold, (val_rows[LABEL].values, val_proba)


def is_sc_or_vsc(track_status) -> bool:
    return isinstance(track_status, str) and any(c in track_status for c in '467')


def is_yellow(track_status) -> bool:
    return isinstance(track_status, str) and '2' in track_status and '4' not in track_status


def fuel_correct(laps: pd.DataFrame, total_laps: int | None = None) -> pd.DataFrame:
    """total_laps must be the scheduled race length when `laps` is only a
    live partial race -- defaulting to the highest lap seen would treat the
    latest lap as the last one (empty tank) on every live update."""
    total_laps = total_laps or laps['LapNumber'].max()
    laps = laps.copy()
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def load_blue_flagged_laps(race_id: str) -> set:
    """(DriverNumber, Lap) pairs shown a blue flag during that lap, from
    race-control messages. On short/high-density tracks, a backmarker can
    be blue-flagged for several consecutive laps while letting the leaders
    through -- real pace loss with nothing to do with tyres or strategy,
    which would otherwise corrupt the pace/gap trend features below."""
    rc_path = BRONZE_DIR / f"{race_id}_race_control.parquet"
    if not rc_path.exists():
        return set()
    rc = pd.read_parquet(rc_path)
    blue = rc[rc['Flag'] == 'BLUE']
    return set(zip(blue['RacingNumber'].astype(str), blue['Lap'].astype(int)))


def build_race_features(laps: pd.DataFrame, race_id: str, event_name: str, rivals: dict,
                        n_laps: int | None = None) -> pd.DataFrame:
    """n_laps must be the scheduled race length when `laps` is a live
    partial race (09_streaming_consumer.py); otherwise race_progress and
    laps_remaining would say every live lap is the final lap."""
    laps = laps.copy()
    laps['TrackStatus'] = laps['TrackStatus'].astype(str)
    n_laps = n_laps or int(laps['LapNumber'].max())
    laps['TimeSeconds'] = laps['Time'].dt.total_seconds()
    team_drivers = laps.groupby('Team')['Driver'].unique().apply(list).to_dict()

    blue_flags = load_blue_flagged_laps(race_id)
    laps['BlueFlagged'] = laps.apply(
        lambda r: (str(r['DriverNumber']), int(r['LapNumber'])) in blue_flags, axis=1)

    # Global per-lap pit-stop count (any driver) -- used for the "pit window
    # just opened, others are reacting" feature.
    pit_counts = laps.groupby('LapNumber')['PitInTime'].apply(lambda s: s.notna().sum())

    # Per-lap-number lookup: Position -> row, for gap/rival lookups.
    by_lap = {ln: sub.set_index('Position') for ln, sub in laps.groupby('LapNumber')}

    # Pass 1: instantaneous per-(driver, lap) state -- gap/rival/pace as of
    # that lap itself, not yet lagged. Built once, reused both for the
    # lagged single-lap features and for trend windows further back.
    state = {}
    for _, row in laps.iterrows():
        d, ln, pos = row['Driver'], row['LapNumber'], row['Position']
        lap_table = by_lap[ln]
        ahead = lap_table[lap_table.index == pos - 1]
        behind = lap_table[lap_table.index == pos + 1]
        state[(d, ln)] = {
            'lap': ln,
            'position': pos,
            'time': row['TimeSeconds'],
            'tyre_age': row['TyreLife'],
            'pitted': pd.notna(row['PitInTime']),
            'gap_ahead_s': (row['TimeSeconds'] - ahead['TimeSeconds'].iloc[0]) if len(ahead) else None,
            'gap_behind_s': (behind['TimeSeconds'].iloc[0] - row['TimeSeconds']) if len(behind) else None,
            'rival_tyre_age_ahead': ahead['TyreLife'].iloc[0] if len(ahead) else None,
            'rival_tyre_age_behind': behind['TyreLife'].iloc[0] if len(behind) else None,
            'relative_pace': row['RelativePace'],
            'blue_flagged': row['BlueFlagged'],
        }

    def trend_history(driver, ln):
        """Up to TREND_WINDOW most-recent prior laps, skipping any that
        were blue-flagged (own_blue_flagged_this_lap/n_blue_flag_laps_recent
        still capture blue-flag laps directly; this just keeps them out of
        the *trend* calculation so a lapped backmarker's forced pace loss
        doesn't get misread as tyres falling off or a closing gap)."""
        history = []
        for k in range(1, TREND_LOOKBACK + 1):
            s = state.get((driver, ln - k))
            if s is None:
                break
            if not s['blue_flagged']:
                history.append(s)
            if len(history) >= TREND_WINDOW:
                break
        return history

    def resolve_rival_car(rival_team, ln):
        """Which of the rival team's (up to two) cars is the relevant one
        to compare against at this lap -- picks whichever is currently
        better-placed on track, since which of their two cars is 'the'
        threat can change through a race and isn't knowable in advance."""
        candidates = [(d, state[(d, ln)]['position']) for d in team_drivers.get(rival_team, [])
                      if (d, ln) in state]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    rows = []
    for driver, dlaps in laps.groupby('Driver'):
        dlaps = dlaps.sort_values('LapNumber')
        n_stops_so_far = 0
        compounds_seen = set()

        for _, row in dlaps.iterrows():
            ln = row['LapNumber']
            label = int(pd.notna(row['PitInTime']))
            compounds_seen.add(row['Compound'])

            feat = {
                'race': race_id, 'event': event_name, 'circuit': dpl.circuit_of(race_id),
                'driver': driver, 'lap': ln,
                'label_pit': label,
                **{f.lower().replace('regs', 'regs_'): v for f, v in dpl.era_flags(race_id).items()},
                'own_tyre_age': row['TyreLife'],
                'own_compound': row['Compound'],
                'lap_number': ln,
                'race_progress': ln / n_laps,
                'laps_remaining': n_laps - ln,
                'sc_vsc_active': int(is_sc_or_vsc(row['TrackStatus'])),
                'yellow_active': int(is_yellow(row['TrackStatus'])),
                'n_compounds_used_so_far': len(compounds_seen),
                'n_stops_so_far': n_stops_so_far,
                'own_blue_flagged_this_lap': int(row['BlueFlagged']),
                'n_blue_flag_laps_recent': sum(
                    1 for k in range(1, 4) if state.get((driver, ln - k), {}).get('blue_flagged')),
                'n_recent_pit_activity': int(pit_counts.get(ln - 1, 0) + pit_counts.get(ln - 2, 0)),
                'position_now': state[(driver, ln)]['position'],
                'gap_ahead_s_now': state[(driver, ln)]['gap_ahead_s'],
                'gap_behind_s_now': state[(driver, ln)]['gap_behind_s'],
                'pitted_this_lap': label,
            }

            # Lagged (previous-lap) position/gap features -- see causal-confusion
            # audit above. Lap 1 has no previous lap, so these stay NaN
            # (LightGBM handles missing values natively).
            prev_state = state.get((driver, ln - 1))
            if prev_state is not None:
                feat['position_prev'] = prev_state['position']
                feat['gap_ahead_s_prev'] = prev_state['gap_ahead_s']
                feat['gap_behind_s_prev'] = prev_state['gap_behind_s']
                feat['rival_tyre_age_ahead_prev'] = prev_state['rival_tyre_age_ahead']
                feat['rival_tyre_age_behind_prev'] = prev_state['rival_tyre_age_behind']

            # Trend features: newest vs. oldest of the last TREND_WINDOW
            # clean (non-blue-flagged) prior laps, normalized to a per-lap
            # rate. Still strictly prior to this lap, same causal-confusion
            # discipline as the single-lag features above.
            history = trend_history(driver, ln)
            if len(history) >= 2:
                newest, oldest = history[0], history[-1]
                span = newest['lap'] - oldest['lap']
                if span > 0:
                    if pd.notna(newest['relative_pace']) and pd.notna(oldest['relative_pace']):
                        feat['own_pace_trend'] = (newest['relative_pace'] - oldest['relative_pace']) / span
                    if newest['gap_ahead_s'] is not None and oldest['gap_ahead_s'] is not None:
                        feat['gap_ahead_trend'] = (newest['gap_ahead_s'] - oldest['gap_ahead_s']) / span
                    if newest['gap_behind_s'] is not None and oldest['gap_behind_s'] is not None:
                        feat['gap_behind_trend'] = (newest['gap_behind_s'] - oldest['gap_behind_s']) / span

            # Championship-rival features: who this driver/team is actually
            # fighting in the standings, which can be far apart on track
            # from the on-track gap/rival features above. Same causal-
            # confusion discipline -- lagged to the previous lap, since the
            # rival's current-lap state could itself be mid-reaction to
            # something that happens this same lap.
            rival_info = rivals.get(driver, {})
            rival_driver = rival_info.get('rival_driver')
            if rival_driver and prev_state is not None:
                rival_prev = state.get((rival_driver, ln - 1))
                if rival_prev is not None:
                    feat['champ_rival_gap_s_prev'] = prev_state['time'] - rival_prev['time']
                    feat['champ_rival_tyre_age_prev'] = rival_prev['tyre_age']
                    feat['champ_rival_pitted_recently'] = int(
                        any(state.get((rival_driver, ln - k), {}).get('pitted') for k in (1, 2)))

            rival_team = rival_info.get('rival_team')
            if rival_team and prev_state is not None:
                rival_car = resolve_rival_car(rival_team, ln - 1)
                if rival_car is not None:
                    rival_prev = state[(rival_car, ln - 1)]
                    feat['team_rival_gap_s_prev'] = prev_state['time'] - rival_prev['time']
                    feat['team_rival_tyre_age_prev'] = rival_prev['tyre_age']
                    feat['team_rival_pitted_recently'] = int(
                        any(state.get((rival_car, ln - k), {}).get('pitted') for k in (1, 2)))

            rows.append(feat)

            if label:
                n_stops_so_far += 1

    out = pd.DataFrame(rows).sort_values(['driver', 'lap'])
    # Next-lap label, only where the driver's next lap exists (not their
    # last or retirement lap) -- those rows have no target and are dropped
    # from training/evaluation.
    nxt = out.groupby('driver')[['lap', 'label_pit']].shift(-1)
    out[LABEL] = nxt['label_pit'].where(nxt['lap'] == out['lap'] + 1)
    return out.reset_index(drop=True)


def add_circuit_sc_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-race-out empirical SC/VSC-active rate per circuit, by
    race-progress decile -- excludes the row's own race so it can't leak
    that race's own incidents into its own feature. Falls back to the
    global (all-events) decile average for events seen only once."""
    df = df.copy()
    df['progress_decile'] = (df['race_progress'] * 10).clip(0, 9).astype(int)

    global_rate = df.groupby('progress_decile')['sc_vsc_active'].mean()

    event_decile_race_rate = (
        df.groupby(['circuit', 'progress_decile', 'race'])['sc_vsc_active'].mean().reset_index()
    )

    probs = np.zeros(len(df))
    for (circuit, decile), grp in df.groupby(['circuit', 'progress_decile']):
        other_races = event_decile_race_rate[
            (event_decile_race_rate['circuit'] == circuit) &
            (event_decile_race_rate['progress_decile'] == decile)
        ]
        idx = grp.index
        this_races = df.loc[idx, 'race'].unique()
        rate_per_race = other_races.set_index('race')['sc_vsc_active']
        leave_one_out = []
        for r in this_races:
            others = rate_per_race.drop(index=r, errors='ignore')
            leave_one_out.append(others.mean() if len(others) else global_rate.get(decile, np.nan))
        race_to_rate = dict(zip(this_races, leave_one_out))
        probs[idx] = df.loc[idx, 'race'].map(race_to_rate).values

    df['circuit_sc_prob'] = probs
    df['circuit_sc_prob'] = df['circuit_sc_prob'].fillna(df['progress_decile'].map(global_rate))
    df = df.drop(columns=['progress_decile'])
    return df


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: laps grouped into equal-size bins by predicted
    probability, mean prediction vs. observed pit rate per bin."""
    d = pd.DataFrame({'pred': proba, 'actual': y_true})
    d['bin'] = pd.qcut(d['pred'].rank(method='first'), n_bins, labels=False)
    return d.groupby('bin').agg(mean_pred=('pred', 'mean'), observed=('actual', 'mean'), n=('actual', 'size'))


def expected_calibration_error(y_true, proba) -> float:
    t = calibration_table(y_true, proba)
    return float((t['n'] * (t['mean_pred'] - t['observed']).abs()).sum() / t['n'].sum())


def pit_window_metrics(test_df: pd.DataFrame, pred) -> tuple:
    """Precision/recall when a prediction within +/-1 lap of a real stop
    counts as a hit."""
    eval_df = test_df[['race', 'driver', 'lap', LABEL]].rename(columns={LABEL: 'label_pit'})
    eval_df['pred'] = pred
    hits_recall = total_actual = hits_precision = total_pred = 0
    for _, grp in eval_df.groupby(['race', 'driver']):
        actual_laps = set(grp.loc[grp['label_pit'] == 1, 'lap'])
        pred_laps = set(grp.loc[grp['pred'] == 1, 'lap'])
        total_actual += len(actual_laps)
        hits_recall += sum(any(abs(al - pl) <= 1 for pl in pred_laps) for al in actual_laps)
        total_pred += len(pred_laps)
        hits_precision += sum(any(abs(al - pl) <= 1 for al in actual_laps) for pl in pred_laps)
    return (hits_precision / total_pred if total_pred else 0.0,
            hits_recall / total_actual if total_actual else 0.0)


def evaluate(pred_proba, test_df, label, threshold) -> dict:
    """Held-out metrics at a threshold chosen BEFORE seeing the test races.
    Average precision (precision-recall AUC) measures ranking without any
    threshold; Brier score and calibration error measure whether the
    probabilities mean what they say."""
    y_true = test_df[LABEL].values
    pred = (pred_proba >= threshold).astype(int)
    window_p, window_r = pit_window_metrics(test_df, pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    r = {'label': label, 'threshold': threshold,
         'avg_precision': average_precision_score(y_true, pred_proba),
         'brier': brier_score_loss(y_true, pred_proba),
         'ece': expected_calibration_error(y_true, pred_proba),
         'mean_proba': float(pred_proba.mean()),
         'precision': precision_score(y_true, pred, zero_division=0),
         'recall': recall_score(y_true, pred, zero_division=0),
         'f1': f1_score(y_true, pred, zero_division=0),
         'window_precision': window_p, 'window_recall': window_r,
         'pred_proba': pred_proba, 'pred': pred}
    print(f"\n=== {label} (held-out races) ===")
    print(f"  average precision {r['avg_precision']:.3f} | Brier {r['brier']:.4f} | calibration error {r['ece']:.4f} "
          f"| mean predicted P {r['mean_proba']:.3f} vs. real pit rate {y_true.mean():.3f}")
    print(f"  at threshold {threshold:.3f} (chosen on the validation season): precision {r['precision']:.3f}, "
          f"recall {r['recall']:.3f}, F1 {r['f1']:.3f}  (TP={tp} FP={fp} FN={fn})")
    print(f"  +/-1-lap pit window: precision {window_p:.3f}, recall {window_r:.3f}")
    return r


ROLLING_SEASONS = [2022, 2023, 2024, 2025, 2026]


def prediction_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Predictions in the shape 06/11 consume: 'lap' is the lap whose END
    the pit is predicted for (row lap + 1, the next lap), 'label_pit'
    whether the driver really pitted then."""
    o = rows[['race', 'event', 'driver']].copy()
    o['lap'] = rows['lap'] + 1
    o['label_pit'] = rows[LABEL].astype(int)
    return o


def rolling_origin_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin evaluation: each season in ROLLING_SEASONS is predicted
    by a model trained only on races before it (threshold chosen on the
    season before), the protocol recommended for data whose distribution
    shifts over time (Tashman 2000; Cerqueira et al. 2019). Complements the
    single pinned test block with several forward-in-time estimates and
    their spread, and gives 06 a ~100-race backtest instead of ~15."""
    in_training_seasons = df['race'].str[:4].astype(int) >= TRAIN_FROM_SEASON
    out, rows = [], []
    for season in ROLLING_SEASONS:
        test = df[(df['race'].str[:4].astype(int) == season) & df[LABEL].notna()]
        train = df[(df['race'] < f"{season}_r00") & in_training_seasons]
        if test.empty or train.empty:
            continue
        model, threshold, _ = train_with_threshold(train)
        proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
        y = test[LABEL].values
        pred = (proba >= threshold).astype(int)
        window_p, window_r = pit_window_metrics(test, pred)
        rows.append({'season': season, 'races': test['race'].nunique(), 'train_races': train['race'].nunique(),
                     'threshold': threshold, 'avg_precision': average_precision_score(y, proba),
                     'brier': brier_score_loss(y, proba), 'calib_error': expected_calibration_error(y, proba),
                     'mean_P': proba.mean(), 'pit_rate': y.mean(),
                     'f1': f1_score(y, pred, zero_division=0),
                     'window_precision': window_p, 'window_recall': window_r})
        o = prediction_frame(test)
        o['pred_proba'], o['pred'], o['season'] = proba, pred, season
        out.append(o)
    print("\n=== Rolling-origin evaluation: each season predicted by a model trained on earlier races only ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return pd.concat(out, ignore_index=True)


def main():
    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    print(f"Building features from {len(lap_files)} races...")

    rivals_df = pd.read_csv('data/championship_rivals.csv')
    rivals_by_race = {
        race: grp.set_index('driver')[['rival_driver', 'rival_team']].to_dict('index')
        for race, grp in rivals_df.groupby('race')
    }

    all_rows = []
    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps = laps[laps['Compound'].isin(DRY_COMPOUNDS + ['INTERMEDIATE', 'WET'])]
        if laps.empty:
            continue
        event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id
        laps = fuel_correct(laps)
        field_pace = compute_field_pace(laps)
        laps = laps.join(field_pace, on='LapNumber')
        laps['RelativePace'] = laps['FuelCorrectedLapTime'] - laps['FieldPace']
        rivals = rivals_by_race.get(race_id, {})
        all_rows.append(build_race_features(laps, race_id, event_name, rivals))

    df = pd.concat(all_rows, ignore_index=True)
    df = add_circuit_sc_prob(df)
    df['own_compound'] = df['own_compound'].astype('category')

    print(f"\n{len(df)} (driver, lap) rows across {df['race'].nunique()} races")
    pos_rate = df[LABEL].mean()
    print(f"Positive rate (pits at the end of the next lap): {pos_rate*100:.2f}% "
          f"({int(df[LABEL].sum())} of {int(df[LABEL].notna().sum())} rows with a next lap)")

    feature_cols = FEATURE_COLS

    # Race-level (not row-level) split: hold out every race from
    # TEST_FROM_RACE on (originally the most recent ~20%), so evaluation
    # reflects generalizing to unseen, later races, not memorizing laps
    # from races partially seen in training.
    in_training_seasons = df['race'].str[:4].astype(int) >= TRAIN_FROM_SEASON
    has_next = df[LABEL].notna()   # rows with a next lap to predict
    train_df = df[(df['race'] < TEST_FROM_RACE) & in_training_seasons & has_next]
    test_df = df[(df['race'] >= TEST_FROM_RACE) & has_next]
    print(f"\nTrain: {train_df['race'].nunique()} races, {len(train_df)} rows")
    print(f"Test:  {test_df['race'].nunique()} races, {len(test_df)} rows (most recent races, held out)")

    fit_rows, val_rows = split_validation(train_df)
    print(f"Validation season (threshold + recalibration chosen here): {val_rows['race'].min()} to "
          f"{val_rows['race'].max()} ({val_rows['race'].nunique()} races)")

    # --- Deployed setup: no class rebalancing ---
    model_a, threshold_a, _ = train_with_threshold(train_df)
    result_a = evaluate(model_a.predict_proba(test_df[feature_cols])[:, 1], test_df,
                        "Unweighted (deployed)", threshold_a)

    # --- Comparison: class_weight='balanced' (the previous setup) ---
    model_bal, threshold_bal, (val_y, val_bal) = train_with_threshold(train_df, balanced=True)
    proba_bal = model_bal.predict_proba(test_df[feature_cols])[:, 1]
    result_bal = evaluate(proba_bal, test_df, "class_weight='balanced'", threshold_bal)

    # --- Comparison: balanced + isotonic recalibration, the other standard
    # fix (map fitted on the validation season's predictions, then applied
    # to the refit model -- a close approximation, used for comparison only).
    iso = IsotonicRegression(out_of_bounds='clip').fit(val_bal, val_y)
    result_iso = evaluate(iso.predict(proba_bal), test_df, "Balanced + isotonic recalibration",
                          best_f1_threshold(val_y, iso.predict(val_bal)))

    # --- Comparison: SMOTENC oversampling (Fathi et al.'s approach) ---
    # SMOTE can't handle NaN or raw categoricals, so: impute the lagged
    # position/gap columns with the TRAIN median only (never touch test
    # statistics), and pass own_compound's integer category codes through
    # SMOTENC's categorical_features so it's resampled correctly rather
    # than being treated as a continuous variable. Oversampling is fit on
    # training rows only, threshold chosen on the validation season.
    cat_idx = [feature_cols.index('own_compound')]

    def smote_model(rows):
        X = rows[feature_cols].copy()
        X['own_compound'] = X['own_compound'].cat.codes
        medians = X.median(numeric_only=True)
        X_res, y_res = SMOTENC(categorical_features=cat_idx, random_state=0).fit_resample(
            X.fillna(medians), rows[LABEL].astype(int))
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   random_state=0, verbosity=-1)
        return model.fit(X_res, y_res, categorical_feature=cat_idx), medians

    def smote_proba(model_and_medians, rows):
        model, medians = model_and_medians
        X = rows[feature_cols].copy()
        X['own_compound'] = X['own_compound'].cat.codes
        return model.predict_proba(X.fillna(medians))[:, 1]

    threshold_smote = best_f1_threshold(val_rows[LABEL].values,
                                        smote_proba(smote_model(fit_rows), val_rows))
    result_smote = evaluate(smote_proba(smote_model(train_df), test_df), test_df,
                            "SMOTENC oversampling", threshold_smote)

    results = (result_a, result_bal, result_iso, result_smote)
    print(f"\n=== Summary (held-out races; thresholds chosen on the validation season) ===")
    print(f"  {'':34s} avg prec.  Brier   calib.err  mean P   F1     window prec.  window recall")
    for r in results:
        print(f"  {r['label']:34s} {r['avg_precision']:.3f}     {r['brier']:.4f}  {r['ece']:.4f}     "
              f"{r['mean_proba']:.3f}   {r['f1']:.3f}  {r['window_precision']:.3f}         {r['window_recall']:.3f}")
    print(f"\nReliability, unweighted (deployed) model -- mean predicted P vs. observed pit rate:")
    print(calibration_table(test_df[LABEL].values, result_a['pred_proba']).round(4).to_string())
    print(f"\nReliability, class_weight='balanced':")
    print(calibration_table(test_df[LABEL].values, result_bal['pred_proba']).round(4).to_string())

    importances = pd.Series(model_a.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature importance (deployed setup):")
    print(importances.to_string())

    df.to_parquet('data/silver/trigger_classifier_features.parquet', index=False)
    test_out = prediction_frame(test_df)
    test_out['pred_proba'] = result_a['pred_proba']
    test_out['pred'] = result_a['pred']
    test_out['pred_proba_balanced'] = result_bal['pred_proba']
    test_out['pred_proba_smote'] = result_smote['pred_proba']
    test_out.to_csv('data/trigger_classifier_test_predictions.csv', index=False)
    rolling = rolling_origin_predictions(df)
    rolling.to_csv('data/trigger_classifier_rolling_predictions.csv', index=False)
    print(f"\nSaved data/silver/trigger_classifier_features.parquet ({len(df)} rows)")
    print(f"Saved data/trigger_classifier_test_predictions.csv ({len(test_out)} rows)")
    print(f"Saved data/trigger_classifier_rolling_predictions.csv ({len(rolling)} rows, "
          f"{rolling['race'].nunique()} races)")

    # Deployed model: retrained on ALL races (not just the train split),
    # with its own threshold chosen the same way (on the last season), for
    # anything downstream that needs live predictions rather than a
    # held-out evaluation. Live replays use 12_as_of_context.py's as-of
    # models instead, trained identically on races before the replayed one.
    deploy_df = df[in_training_seasons]
    final_model, final_threshold, _ = train_with_threshold(deploy_df)
    Path('data/models').mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': final_model, 'feature_cols': feature_cols, 'threshold': final_threshold,
                 'compound_categories': df['own_compound'].cat.categories.tolist()},
                'data/models/trigger_classifier.joblib')
    print(f"Saved data/models/trigger_classifier.joblib (trained on all {len(deploy_df)} rows "
          f"from {TRAIN_FROM_SEASON} on, threshold {final_threshold:.3f})")


if __name__ == '__main__':
    main()
