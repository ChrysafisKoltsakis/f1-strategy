"""
Phase 4 -- trigger/strategy classifier (imitation learning). The hard,
novel part of the thesis: predict whether a team pits on a given lap, by
learning from what real teams actually did across 103 races (2022-2026),
rather than a hand-built hazard-rate/Bayesian trigger model -- see the
project's ML-not-stats framing decision. GBM-first (LightGBM), not a
sequence model, per the build plan.

Label: did this driver pit at the end of this (driver, lap) -- i.e. is
PitInTime set on this lap. Positive rate is expected to be small (~2-3%
of laps), so this is trained with LightGBM's built-in class weighting
rather than resampling.

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
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTENC

BRONZE_DIR = Path('data/bronze')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
TREND_WINDOW = 3  # how many non-blue-flagged prior laps to use for trend features
TREND_LOOKBACK = 6  # how far back to search to find TREND_WINDOW clean laps


def is_sc_or_vsc(track_status) -> bool:
    return isinstance(track_status, str) and any(c in track_status for c in '467')


def is_yellow(track_status) -> bool:
    return isinstance(track_status, str) and '2' in track_status and '4' not in track_status


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
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


def build_race_features(laps: pd.DataFrame, race_id: str, event_name: str, rivals: dict) -> pd.DataFrame:
    laps = laps.copy()
    laps['TrackStatus'] = laps['TrackStatus'].astype(str)
    n_laps = int(laps['LapNumber'].max())
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
                'race': race_id, 'event': event_name, 'driver': driver, 'lap': ln,
                'label_pit': label,
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

    return pd.DataFrame(rows)


def add_circuit_sc_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-race-out empirical SC/VSC-active rate per event, by
    race-progress decile -- excludes the row's own race so it can't leak
    that race's own incidents into its own feature. Falls back to the
    global (all-events) decile average for events seen only once."""
    df = df.copy()
    df['progress_decile'] = (df['race_progress'] * 10).clip(0, 9).astype(int)

    global_rate = df.groupby('progress_decile')['sc_vsc_active'].mean()

    event_decile_race_rate = (
        df.groupby(['event', 'progress_decile', 'race'])['sc_vsc_active'].mean().reset_index()
    )

    probs = np.zeros(len(df))
    for (event, decile), grp in df.groupby(['event', 'progress_decile']):
        other_races = event_decile_race_rate[
            (event_decile_race_rate['event'] == event) &
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


def evaluate(pred_proba, test_df, label):
    """Threshold sweep + best-F1 summary + +/-1-lap pit-window tolerance,
    shared between the class-weighted and SMOTE model comparisons."""
    y_true = test_df['label_pit'].values
    print(f"\n=== {label}: precision/recall by threshold (held-out races) ===")
    best_f1, best_t = -1, 0.5
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pred_t = (pred_proba >= t).astype(int)
        p, r = precision_score(y_true, pred_t, zero_division=0), recall_score(y_true, pred_t, zero_division=0)
        f1 = f1_score(y_true, pred_t, zero_division=0)
        print(f"  t={t:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    pred = (pred_proba >= best_t).astype(int)
    precision, recall = precision_score(y_true, pred), recall_score(y_true, pred)
    print(f"\n{label} -- best-F1 threshold: {best_t:.2f}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {best_f1:.3f}")
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    eval_df = test_df[['race', 'driver', 'lap', 'label_pit']].copy()
    eval_df['pred'] = pred
    tol_hits_recall, tol_total_actual = 0, 0
    tol_hits_precision, tol_total_pred = 0, 0
    for (_, _), grp in eval_df.groupby(['race', 'driver']):
        grp = grp.sort_values('lap')
        actual_laps = set(grp.loc[grp['label_pit'] == 1, 'lap'])
        pred_laps = set(grp.loc[grp['pred'] == 1, 'lap'])
        for al in actual_laps:
            tol_total_actual += 1
            if any(abs(al - pl) <= 1 for pl in pred_laps):
                tol_hits_recall += 1
        for pl in pred_laps:
            tol_total_pred += 1
            if any(abs(al - pl) <= 1 for al in actual_laps):
                tol_hits_precision += 1
    tol_recall = tol_hits_recall / tol_total_actual if tol_total_actual else 0
    tol_precision = tol_hits_precision / tol_total_pred if tol_total_pred else 0
    print(f"\n{label} -- +/-1-lap-tolerant (pit-window) evaluation, threshold={best_t:.2f}:")
    print(f"  Precision: {tol_precision:.3f}  Recall: {tol_recall:.3f}")

    return {
        'label': label, 'threshold': best_t, 'exact_precision': precision, 'exact_recall': recall,
        'exact_f1': best_f1, 'window_precision': tol_precision, 'window_recall': tol_recall,
        'pred_proba': pred_proba, 'pred': pred,
    }


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
    pos_rate = df['label_pit'].mean()
    print(f"Positive rate (pitted this lap): {pos_rate*100:.2f}% ({df['label_pit'].sum()} of {len(df)})")

    feature_cols = [
        'own_tyre_age', 'own_compound', 'lap_number', 'race_progress', 'laps_remaining',
        'sc_vsc_active', 'yellow_active', 'n_compounds_used_so_far', 'n_stops_so_far',
        'position_prev', 'gap_ahead_s_prev', 'gap_behind_s_prev',
        'rival_tyre_age_ahead_prev', 'rival_tyre_age_behind_prev', 'circuit_sc_prob',
        'own_pace_trend', 'gap_ahead_trend', 'gap_behind_trend',
        'own_blue_flagged_this_lap', 'n_blue_flag_laps_recent', 'n_recent_pit_activity',
        'champ_rival_gap_s_prev', 'champ_rival_tyre_age_prev', 'champ_rival_pitted_recently',
        'team_rival_gap_s_prev', 'team_rival_tyre_age_prev', 'team_rival_pitted_recently',
    ]

    # Race-level (not row-level) split: hold out the most recent ~20% of races
    # by year/round as the test set, so evaluation reflects generalizing to
    # unseen races, not memorizing laps from races partially seen in training.
    races_sorted = sorted(df['race'].unique())
    n_test = max(1, int(len(races_sorted) * 0.2))
    test_races = set(races_sorted[-n_test:])
    train_df = df[~df['race'].isin(test_races)]
    test_df = df[df['race'].isin(test_races)]
    print(f"\nTrain: {train_df['race'].nunique()} races, {len(train_df)} rows")
    print(f"Test:  {test_df['race'].nunique()} races, {len(test_df)} rows (most recent races, held out)")

    # --- Model A: class_weight='balanced' (the original baseline) ---
    model_a = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        class_weight='balanced', random_state=0, verbosity=-1,
    )
    model_a.fit(train_df[feature_cols], train_df['label_pit'], categorical_feature=['own_compound'])
    proba_a = model_a.predict_proba(test_df[feature_cols])[:, 1]
    result_a = evaluate(proba_a, test_df, "Model A: class_weight='balanced'")

    # --- Model B: SMOTENC oversampling (Fathi et al.'s approach) ---
    # SMOTE can't handle NaN or raw categoricals, so: impute the lagged
    # position/gap columns with the TRAIN median only (never touch test
    # statistics), and pass own_compound's integer category codes through
    # SMOTENC's categorical_features so it's resampled correctly rather
    # than being treated as a continuous variable. Oversampling is fit on
    # the training rows only -- the test set stays untouched/real so
    # evaluation is still honest.
    smote_cols = feature_cols.copy()
    X_train = train_df[smote_cols].copy()
    X_test = test_df[smote_cols].copy()
    X_train['own_compound'] = X_train['own_compound'].cat.codes
    X_test['own_compound'] = X_test['own_compound'].cat.codes

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    cat_idx = [smote_cols.index('own_compound')]
    smote = SMOTENC(categorical_features=cat_idx, random_state=0)
    X_res, y_res = smote.fit_resample(X_train, train_df['label_pit'])
    print(f"\nSMOTE resampled train set: {len(X_train)} -> {len(X_res)} rows "
          f"({y_res.mean()*100:.1f}% positive)")

    model_b = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbosity=-1,
    )
    model_b.fit(X_res, y_res, categorical_feature=cat_idx)
    proba_b = model_b.predict_proba(X_test)[:, 1]
    result_b = evaluate(proba_b, test_df, "Model B: SMOTENC oversampling")

    print(f"\n=== Summary: class_weight='balanced' vs. SMOTE ===")
    print(f"  {'':30s}  exact F1   window precision   window recall")
    for r in (result_a, result_b):
        print(f"  {r['label']:30s}  {r['exact_f1']:.3f}      {r['window_precision']:.3f}             {r['window_recall']:.3f}")

    importances = pd.Series(model_a.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nFeature importance (Model A):")
    print(importances.to_string())

    df.to_parquet('data/silver/trigger_classifier_features.parquet', index=False)
    test_out = test_df[['race', 'event', 'driver', 'lap', 'label_pit']].copy()
    test_out['pred_proba_balanced'] = result_a['pred_proba']
    test_out['pred_balanced'] = result_a['pred']
    test_out['pred_proba_smote'] = result_b['pred_proba']
    test_out['pred_smote'] = result_b['pred']
    test_out.to_csv('data/trigger_classifier_test_predictions.csv', index=False)
    print(f"\nSaved data/silver/trigger_classifier_features.parquet ({len(df)} rows)")
    print(f"Saved data/trigger_classifier_test_predictions.csv ({len(test_out)} rows)")


if __name__ == '__main__':
    main()
