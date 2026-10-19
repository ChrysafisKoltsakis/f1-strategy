"""
Fixed degradation-curve and pit-loss estimation, addressing the issues
found in 00_explore_one_race.py on 2023 Spain and in the first pass at
this script across the half-season:

  1. SOFT compound showed negative degradation -- likely track evolution
     (whole field getting faster early in the race) confounded with tyre
     wear. Fix: detrend each lap against the field's median pace on that
     lap number FIRST (RelativePace = FuelCorrectedLapTime - FieldPace),
     then fit tyre-age effect on the residual. This avoids the collinearity
     a first attempt at LapTime ~ TyreAge + FieldPace ran into (a driver's
     own lap time is part of the field median that same lap, so the two
     regressors are nearly collinear -- the fit "worked" but produced an
     intercept/coefficient pair that were only valid together, and pit-loss
     prediction silently dropped the FieldPace term, producing nonsense
     80-200s "pit losses").

  2. Outlier laps (85-89s vs ~78s baseline) survived the green-flag
     filter -- these were lap 1 (standing start, not pit-related) and
     laps immediately after a SC/VSC restart (pack bunching under a
     nominally-green status). Fix: explicitly drop LapNumber == 1 and
     the 2 laps following any non-green TrackStatus lap, plus an IQR
     trim per compound as a statistical backstop. Also discovered
     TrackStatus is not single-character -- FastF1 concatenates flags
     active during a lap (e.g. "124") -- so exact-match '1' is required,
     not a substring check.

  3. Pit-loss was estimated by comparing the pit lap's raw time to
     green pace, which conflated several effects, and a first fix still
     produced 50-200s "pit losses" because of the collinearity bug above.
     Now: pair each driver's in-lap (PitInTime set) with their out-lap
     (PitOutTime set, following lap), restrict to stops where both laps
     were run under a pure green flag, and compare against
         FieldPace[lap] + (intercept + slope * TyreAge)
     i.e. the detrended degradation curve added back onto that lap's
     actual field pace -- not a global average.

Run across all cached Bronze races: python notebooks/02_fixed_degradation_pitloss.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path('data/bronze')
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
VALID_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}

# Everything per-circuit (pit loss, tyre model, SC rates, default plans) is
# keyed by circuit, not event name: Styrian/Austrian, 70th Anniversary/
# British, Brazilian/Sao Paulo, Mexican/Mexico City and 2026's Barcelona GP/
# earlier Spanish GPs are the same tracks, while the 2026 Spanish GP is a
# new one (Madrid) and the 2026 Bahrain GP ran in Kuala Lumpur. FastF1's
# schedule location (01 writes it to data/circuits.csv) identifies the
# track, once its own renames are merged; the Sakhir GP used Bahrain's outer
# loop, a different lap.
# Regulation eras, by first season: 2022 (ground-effect cars, 18-inch
# tyres) and 2026 (new chassis and power-unit rules, narrower tyres). Used
# for era-matched usable tyre life (03) and default-plan history (14), and
# by the models as flags (era_flags) and a fading prior over older eras
# (era_weights).
ERA_STARTS = (2022, 2026)
# Once races of a new era are in the training data, everything before it
# together counts as this many new-era races, however many it really is: a
# fixed-strength prior that new data progressively outweighs. None = no
# down-weighting. 2022: k = 10 was best or within noise of best for both
# the tyre model and the trigger classifier, against 0, 5, 20 and none.
# 2026: predicting rounds 5-16 from earlier races, any down-weighting made
# both models worse (classifier average precision 0.096 at k = 5 rising to
# 0.135 with none; tyre-model MAE best with none), and neither model gains
# from a 2026 flag -- plausibly because both work relative to the field
# (pace vs the field median; gaps, positions, tyre ages), which a rules
# change every car makes at once mostly leaves alone.
PRIOR_ERA_RACES = {2022: 10, 2026: None}


def era_of(race_id: str) -> int:
    """First season of the regulation era a race belongs to (2019 = pre-2022)."""
    year = int(race_id[:4])
    return max([2019] + [e for e in ERA_STARTS if year >= e])


def era_flags(race_id: str) -> dict:
    """{'Regs2022': 0/1, 'Regs2026': 0/1} -- the model features."""
    year = int(race_id[:4])
    return {f'Regs{e}': int(year >= e) for e in ERA_STARTS}


def era_weights(race_ids: pd.Series, prior_races: dict | None = None) -> np.ndarray:
    """Per-row training weights implementing PRIOR_ERA_RACES, nested: for
    each era whose races are in the data, all rows before it are scaled
    so they total k races' worth (counting each earlier race at the
    weight it already has, so pre-2022 stays down-weighted within the
    pre-2026 block). Only relative weights matter, so a training set
    with no races of the newest era is left as it was."""
    prior_races = PRIOR_ERA_RACES if prior_races is None else prior_races
    races = race_ids.astype(str).values
    years = np.array([int(r[:4]) for r in races])
    w = np.ones(len(races))
    for start in ERA_STARTS:
        k = prior_races.get(start)
        before = years < start
        if k is None or not before.any() or before.all():
            continue
        per_race = pd.Series(w[before], index=races[before]).groupby(level=0).first()
        w[before] *= k / per_race.sum()
    return w


CIRCUITS_FILE = Path('data/circuits.csv')
LOCATION_ALIASES = {'Miami Gardens': 'Miami', 'Monte Carlo': 'Monaco', 'Marina Bay': 'Singapore',
                    'Yas Marina': 'Yas Island'}
EVENT_CIRCUIT_OVERRIDES = {'Sakhir Grand Prix': 'Sakhir (outer)'}
_CIRCUITS = None


def circuit_of(race_id: str) -> str:
    """Canonical circuit key for a race id ('2026_r14' -> 'Madrid')."""
    global _CIRCUITS
    if _CIRCUITS is None:
        c = pd.read_csv(CIRCUITS_FILE)
        key = c['location'].replace(LOCATION_ALIASES)
        key = c['event'].map(EVENT_CIRCUIT_OVERRIDES).fillna(key)
        _CIRCUITS = dict(zip(c['race'], key))
    return _CIRCUITS[race_id]


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    total_laps = laps['LapNumber'].max()
    laps = laps.copy()
    laps['TrackStatus'] = laps['TrackStatus'].astype(str)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
    laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG
    return laps


def compute_field_pace(laps: pd.DataFrame) -> pd.Series:
    """Median fuel-corrected lap time across the whole field, per lap number."""
    return laps.groupby('LapNumber')['FuelCorrectedLapTime'].median().rename('FieldPace')


def clean_laps(laps: pd.DataFrame, field_pace: pd.Series) -> pd.DataFrame:
    """Green-flag, non-pit, non-restart-affected, IQR-trimmed laps, detrended vs field pace."""
    laps = laps.join(field_pace, on='LapNumber')
    laps['RelativePace'] = laps['FuelCorrectedLapTime'] - laps['FieldPace']

    non_green_laps = set(laps.loc[laps['TrackStatus'] != '1', 'LapNumber'].unique())
    restart_affected = set()
    for ln in non_green_laps:
        restart_affected.update([ln + 1, ln + 2])

    clean = laps[
        laps['PitInTime'].isna() &
        laps['PitOutTime'].isna() &
        laps['RelativePace'].notna() &
        laps['TyreLife'].notna() &
        laps['Compound'].isin(VALID_COMPOUNDS) &
        (laps['LapNumber'] != 1) &
        (~laps['LapNumber'].isin(restart_affected)) &
        (laps['TrackStatus'] == '1')
    ].copy()

    trimmed = []
    for compound, sub in clean.groupby('Compound'):
        q1, q3 = sub['RelativePace'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        trimmed.append(sub[sub['RelativePace'].between(lo, hi)])
    return pd.concat(trimmed) if trimmed else clean.iloc[0:0]


def fit_degradation(clean: pd.DataFrame) -> dict:
    """
    RelativePace ~ TyreAge, i.e. "seconds off the field's median pace on that
    lap, as a function of tyre age" -- with track evolution already removed by
    the field-pace detrend, so this is a single-variable fit with no
    collinearity risk. Also reports the median of independent per-stint
    slopes as a check the pooled number isn't an artifact of a few stints.
    """
    results = {}
    for compound, sub in clean.groupby('Compound'):
        if len(sub) < 15 or sub['TyreLife'].nunique() < 2:
            continue

        slope, intercept = np.polyfit(sub['TyreLife'], sub['RelativePace'], 1)

        stint_slopes = []
        for _, stint_sub in sub.groupby(['Driver', 'Stint']):
            if len(stint_sub) < 5 or stint_sub['TyreLife'].nunique() < 2:
                continue
            s_slope, _ = np.polyfit(stint_sub['TyreLife'], stint_sub['RelativePace'], 1)
            stint_slopes.append(s_slope)

        results[compound] = {
            'n_laps': len(sub),
            'tyre_age_slope': slope,
            'relative_intercept': intercept,
            'n_stints': len(stint_slopes),
            'median_stint_slope': np.median(stint_slopes) if stint_slopes else None,
        }
    return results


def collect_pit_stop_losses(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> list:
    """
    Pair in-laps with out-laps per driver, restricted to stops run entirely
    under green flag (SC/VSC/red-flag stops have in/out times inflated by
    the incident itself, unrelated to pit-lane cost, and badly distort the
    estimate if included). Compare each lap's actual time against
    FieldPace[lap] + degradation curve at that tyre age -- the field's
    actual pace that lap, adjusted for tyre wear, not a flat average.
    Returns one record per stop (Driver, Team, loss) rather than an
    aggregate, so callers can pool per-race (circuit constant) or across
    races (per-team crew-speed adjustment).
    """
    laps = laps.join(field_pace, on='LapNumber')
    green = laps['TrackStatus'] == '1'
    team_col = 'Team' if 'Team' in laps.columns else None
    cols = ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']
    if team_col:
        cols.append(team_col)

    in_laps = laps[laps['PitInTime'].notna() & green][cols]
    out_laps = laps[laps['PitOutTime'].notna() & green][cols]

    def expected_pace(row):
        deg = degradation.get(row['Compound'])
        if deg is None or pd.isna(row['FieldPace']) or pd.isna(row['TyreLife']):
            return None
        return row['FieldPace'] + deg['relative_intercept'] + deg['tyre_age_slope'] * row['TyreLife']

    records = []
    for _, in_lap in in_laps.iterrows():
        match = out_laps[
            (out_laps['Driver'] == in_lap['Driver']) &
            (out_laps['LapNumber'] == in_lap['LapNumber'] + 1)
        ]
        if match.empty or pd.isna(in_lap['FuelCorrectedLapTime']):
            continue
        out_lap = match.iloc[0]
        if pd.isna(out_lap['FuelCorrectedLapTime']):
            continue

        expected_in = expected_pace(in_lap)
        expected_out = expected_pace(out_lap)
        if expected_in is None or expected_out is None:
            continue

        loss = (in_lap['FuelCorrectedLapTime'] - expected_in) + (out_lap['FuelCorrectedLapTime'] - expected_out)
        records.append({
            'driver': in_lap['Driver'], 'team': in_lap[team_col] if team_col else None, 'loss': loss,
        })
    return records


MIN_PLAUSIBLE_GREEN_PIT_LOSS = 12.0  # seconds -- pit-lane transit time alone rarely drops below this


def estimate_pit_loss(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> dict:
    """Green-flag pit loss. A physically implausible median (found via
    the DP baseline exploiting it into a 47-stop "optimal" strategy once
    a more expressive degradation model made the DP aggressive enough to
    actually chase the exploit) means the in-lap/out-lap decomposition
    got corrupted for this race -- e.g. a wet/disrupted race where the
    "expected pace" baseline itself is unreliable (Emilia Romagna 2022,
    caught this way, is a race already flagged elsewhere in this project
    for exactly that kind of disruption) -- not a genuinely fast pit
    lane. Rejected outright rather than silently trusted, same principle
    as the >=30-lap gate on degradation curves elsewhere in this file."""
    losses = [r['loss'] for r in collect_pit_stop_losses(laps, field_pace, degradation)]
    if not losses:
        return {'n_stops': 0, 'median_pit_loss': None}
    median_loss = float(np.median(losses))
    if median_loss < MIN_PLAUSIBLE_GREEN_PIT_LOSS:
        return {'n_stops': len(losses), 'median_pit_loss': None,
                'rejected_implausible_median': median_loss}
    return {'n_stops': len(losses), 'median_pit_loss': median_loss, 'std': float(np.std(losses))}


def _is_sc_or_vsc(track_status) -> bool:
    """True if full Safety Car ('4') or Virtual Safety Car ('6'/'7') is
    active per FastF1's concatenated per-lap TrackStatus codes. Red flag
    ('5') is deliberately excluded even if it co-occurs -- a red-flag stop
    is a different regime (near-zero cost, field fully stationary) that
    would corrupt an SC/VSC pit-loss estimate if pooled in."""
    return isinstance(track_status, str) and any(c in track_status for c in '467') and '5' not in track_status


def is_vsc_message(message) -> bool:
    """Race-control wording changed in 2026: 'VIRTUAL SAFETY CAR DEPLOYED'
    became 'VSC DEPLOYED', which a check for 'VIRTUAL' alone misread as a
    full safety car (and inflated the full-SC gap compression from ~0.21 to
    0.54, since VSCs don't bunch the field)."""
    m = str(message).upper()
    return 'VIRTUAL' in m or 'VSC' in m


def neutralised_laps(laps: pd.DataFrame) -> dict:
    """{lap: 'SC' or 'VSC'} for laps on which most of the field ran under a
    full safety car or a virtual one (red flags excluded) -- the end of such
    a lap is a cheap moment to pit."""
    st = laps['TrackStatus'].astype(str)
    flags = laps.assign(sc=st.apply(lambda s: '4' in s and '5' not in s),
                        neut=st.apply(_is_sc_or_vsc)).groupby('LapNumber')[['sc', 'neut']].mean()
    return {int(l): ('SC' if r['sc'] > 0.5 else 'VSC') for l, r in flags.iterrows() if r['neut'] > 0.5}


def sc_in_laps(laps: pd.DataFrame) -> frozenset:
    """Laps on which most of the field ran under SC or VSC."""
    return frozenset(neutralised_laps(laps))


SC_RATIO_MIN_STOPS = 5  # races with fewer measured SC/VSC stops don't inform the pooled ratio


def sc_pit_loss_ratio(before_race: str | None = None) -> float:
    """Pooled SC/VSC-to-green pit-loss ratio (both kinds together): median
    over races with at least SC_RATIO_MIN_STOPS measured SC/VSC stops (only
    races before `before_race`, for live use). Per-race estimates are far
    too noisy to use directly -- from -12s to +57s, often from one or two
    stops -- and near-zero or negative values let the hindsight DP pit on
    every lap of a safety car for fresh tyres (up to 15 "optimal" stops)."""
    sc = pd.read_csv('data/sc_pit_loss_summary.csv')
    green = pd.read_csv('data/pit_loss_summary.csv')[['race', 'median_pit_loss']]
    m = sc.merge(green, on='race', suffixes=('_sc', '_green')).dropna(subset=['median_pit_loss_sc', 'median_pit_loss_green'])
    m = m[m['n_stops'] >= SC_RATIO_MIN_STOPS]
    if before_race is not None:
        m = m[m['race'] < before_race]
    return float((m['median_pit_loss_sc'] / m['median_pit_loss_green']).median()) if len(m) else 0.5


def sc_pit_loss_for(race_id: str, green_pit_loss: float, before_race: str | None = None) -> float:
    """Pooled SC/VSC pit loss for a race (used live, where the kind of a
    future neutralisation isn't known): green pit loss x the pooled ratio."""
    return green_pit_loss * sc_pit_loss_ratio(before_race)


def neutralised_pit_ratios(before_race: str | None = None) -> dict:
    """{'SC': ratio, 'VSC': ratio} of pit loss to the circuit's green pit
    loss, median over every individual measured stop of that kind (the
    median keeps the many noisy single stops from mattering)."""
    stops = pd.read_csv('data/neutralised_pit_stops.csv')
    if before_race is not None:
        stops = stops[stops['race'] < before_race]
    ratio = stops['loss'] / stops['circuit_pit_loss']
    return {k: float(ratio[stops['kind'] == k].median()) for k in ('SC', 'VSC')}


def collect_sc_gap_compression(race_id: str, laps: pd.DataFrame) -> list:
    """For each full safety car: every same-lap car's time gap to the field
    median on the last green lap before it vs. on the restart lap. Cars that
    pitted in between are left out (their gap moved for another reason)."""
    rc_path = BRONZE_DIR / f"{race_id}_race_control.parquet"
    if not rc_path.exists():
        return []
    rc = pd.read_parquet(rc_path)
    sc = rc[(rc['Category'] == 'SafetyCar') & ~rc['Message'].apply(is_vsc_message)].sort_values('Time')
    t = laps.assign(t=laps['Time'].dt.total_seconds())
    obs, start = [], None
    for _, r in sc.iterrows():
        if r['Status'] == 'DEPLOYED':
            start = r['Lap']
        elif r['Status'] in ('ENDING', 'IN THIS LAP') and pd.notna(start) and pd.notna(r['Lap']) and start is not None:
            before, after = int(start) - 1, int(r['Lap']) + 1
            start = None
            if before < 2:
                continue
            pitted = set(t.loc[t['LapNumber'].between(before, after) & t['PitInTime'].notna(), 'Driver'])
            gaps = {}
            for lap in (before, after):
                x = t[t['LapNumber'] == lap].set_index('Driver')['t']
                gaps[lap] = x - x.median()
            d = pd.concat([gaps[before].rename('before'), gaps[after].rename('after')], axis=1).dropna()
            d = d[(d['before'].abs() < 60) & ~d.index.isin(pitted)]
            obs += [{'race': race_id, 'driver': drv, 'gap_before': v['before'], 'gap_after': v['after']}
                    for drv, v in d.iterrows()]
    return obs


def sc_gap_compression(before_race: str | None = None) -> float:
    """beta: a car's time gap to the field median after a full safety car
    is ~beta x the gap before it (slope through the origin). Heilmeier et
    al.'s race simulation gets the same effect from a "ghost" safety car
    that resets gaps to ~0.8s per position; this is its statistical
    equivalent for a model that doesn't track positions."""
    obs = pd.read_csv('data/sc_gap_compression_obs.csv')
    if before_race is not None:
        obs = obs[obs['race'] < before_race]
    return float((obs['gap_before'] * obs['gap_after']).sum() / (obs['gap_before'] ** 2).sum())


def neutralisation(laps: pd.DataFrame, sc_price: float, vsc_price: float, beta: float) -> dict:
    """Everything the strategy replays and the hindsight DP need to price a
    race's real safety cars: per-lap kind, the pit price under each kind,
    and the laps after which accumulated gaps compress by beta -- the last
    green lap before each full-SC run, matching how both were measured:
    beta from gaps just before deployment (cars that then pitted excluded),
    and SC stop prices from stops made during safety cars, which already
    include the catch-up to the queue. (Compressing at the restart instead
    double-counted that catch-up, halving SC stops to ~2s and making the
    hindsight DP pit twice under one safety car.)"""
    kinds = neutralised_laps(laps)
    compress_after = {l - 1 for l, k in kinds.items() if k == 'SC' and kinds.get(l - 1) != 'SC' and l > 1}
    # Each consecutive run of SC (or VSC) laps is one neutralisation period.
    period, pid = {}, -1
    for l in sorted(kinds):
        if not (l - 1 in kinds and kinds[l - 1] == kinds[l]):
            pid += 1
        period[l] = pid
    return {'kinds': kinds, 'period': period, 'price': {'SC': sc_price, 'VSC': vsc_price},
            'compress_after': frozenset(compress_after), 'beta': beta}


def stop_price(neutral: dict | None, in_lap: int, green_price: float, last_cheap_period: int = -1) -> tuple:
    """(price, last_cheap_period) for a stop at the end of in_lap. Only the
    first stop in a neutralisation period gets that period's cheap price;
    another stop under the same safety car pays the green price. Real
    teams don't stop twice under one safety car -- each stop drops them down
    the queue, a track-position cost a per-car time model doesn't see -- and
    without the rule the hindsight DP made up to 4 stops under a single
    Monaco safety car."""
    if neutral is None:
        return green_price, last_cheap_period
    kind = neutral['kinds'].get(int(in_lap))
    if not kind or neutral['period'][int(in_lap)] == last_cheap_period:
        return green_price, last_cheap_period
    return neutral['price'][kind], neutral['period'][int(in_lap)]


def estimate_sc_pit_loss(laps: pd.DataFrame, field_pace: pd.Series, degradation: dict) -> dict:
    """
    Same in-lap/out-lap decomposition as estimate_pit_loss, but for stops
    taken while SC/VSC is active on the in-lap (the lap the driver commits
    to pitting) -- this is the separate, much cheaper pit-loss constant
    real teams rely on when deciding to "pit under the SC". FieldPace
    already reflects the SC/VSC-slowed pace for that lap number (it's
    computed from the whole unfiltered field, not just green laps), so the
    same actual-vs-expected decomposition isolates the pit-lane-specific
    cost on top of the SC slowdown, same as the green-flag estimate does.
    """
    laps = laps.join(field_pace, on='LapNumber')
    sc_mask = laps['TrackStatus'].apply(_is_sc_or_vsc)

    in_laps = laps[laps['PitInTime'].notna() & sc_mask][
        ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']]
    out_laps = laps[laps['PitOutTime'].notna()][
        ['Driver', 'LapNumber', 'FuelCorrectedLapTime', 'Compound', 'TyreLife', 'FieldPace']]

    def expected_pace(row):
        deg = degradation.get(row['Compound'])
        if deg is None or pd.isna(row['FieldPace']) or pd.isna(row['TyreLife']):
            return None
        return row['FieldPace'] + deg['relative_intercept'] + deg['tyre_age_slope'] * row['TyreLife']

    losses, stops = [], []
    for _, in_lap in in_laps.iterrows():
        match = out_laps[
            (out_laps['Driver'] == in_lap['Driver']) &
            (out_laps['LapNumber'] == in_lap['LapNumber'] + 1)
        ]
        if match.empty or pd.isna(in_lap['FuelCorrectedLapTime']):
            continue
        out_lap = match.iloc[0]
        if pd.isna(out_lap['FuelCorrectedLapTime']):
            continue

        expected_in = expected_pace(in_lap)
        expected_out = expected_pace(out_lap)
        if expected_in is None or expected_out is None:
            continue

        loss = (in_lap['FuelCorrectedLapTime'] - expected_in) + (out_lap['FuelCorrectedLapTime'] - expected_out)
        losses.append(loss)
        # Full SC ('4') and VSC ('6'/'7') save very different amounts (a
        # calibrated Monte Carlo engine fits ~0.45 vs ~0.65 of green pit
        # loss -- Santillana 2026), so each stop is tagged by type.
        status = str(laps.loc[in_lap.name, 'TrackStatus'])
        stops.append({'kind': 'SC' if '4' in status else 'VSC', 'loss': loss})

    if not losses:
        return {'n_stops': 0, 'median_pit_loss': None, 'stops': []}
    return {'n_stops': len(losses), 'median_pit_loss': float(np.median(losses)), 'std': float(np.std(losses)),
            'stops': stops}


MIN_TEAM_STOPS = 15


def team_pit_loss_adjustment(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Each team's pooled median deviation from the circuit-wide pit-loss
    constant, for teams with at least MIN_TEAM_STOPS stops. Shared with
    12_as_of_context.py, which applies it to earlier races only."""
    stops_df = stops_df.assign(deviation=stops_df['loss'] - stops_df['circuit_pit_loss'])
    team_adj = stops_df.groupby('team')['deviation'].agg(['median', 'count']).reset_index()
    team_adj.columns = ['team', 'adjustment_s', 'n_stops']
    return team_adj[team_adj['n_stops'] >= MIN_TEAM_STOPS].sort_values('adjustment_s')


def main():
    lap_files = sorted(BRONZE_DIR.glob('*_laps.parquet'))
    print(f"Processing {len(lap_files)} races\n")

    all_degradation_rows = []
    all_pit_loss_rows = []
    all_sc_pit_loss_rows = []
    all_neutralised_stops = []  # per stop under SC/VSC, by kind, for the typed SC/VSC price ratios
    all_compression_obs = []    # per car per full SC: gap to field median before vs. after
    all_stop_records = []  # per-stop (team, loss, circuit baseline) for the team-adjustment pass

    for f in lap_files:
        race_id = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f)
        laps = fuel_correct(laps)
        event_name = laps['EventName'].iloc[0] if 'EventName' in laps else race_id

        field_pace = compute_field_pace(laps)
        clean = clean_laps(laps, field_pace)
        degradation = fit_degradation(clean)
        stop_records = collect_pit_stop_losses(laps, field_pace, degradation)
        pit_loss = estimate_pit_loss(laps, field_pace, degradation)
        sc_pit_loss = estimate_sc_pit_loss(laps, field_pace, degradation)

        if pit_loss['median_pit_loss'] is not None:
            for r in stop_records:
                all_stop_records.append({**r, 'race': race_id, 'circuit_pit_loss': pit_loss['median_pit_loss']})
            for r in sc_pit_loss['stops']:
                all_neutralised_stops.append({**r, 'race': race_id, 'circuit_pit_loss': pit_loss['median_pit_loss']})
        all_compression_obs += collect_sc_gap_compression(race_id, laps)

        print(f"=== {race_id}: {event_name} ===")
        print(f"  {len(laps)} total laps -> {len(clean)} clean laps after filtering "
              f"({100*len(clean)/len(laps):.0f}%)")
        for compound, d in degradation.items():
            stint_slope_str = (f"{d['median_stint_slope']:+.3f}" if d['median_stint_slope'] is not None
                                else "n/a")
            print(f"  {compound:8s}: pooled slope {d['tyre_age_slope']:+.3f} s/lap, "
                  f"median per-stint slope {stint_slope_str} s/lap "
                  f"(n={d['n_laps']} laps, {d['n_stints']} stints)")
            all_degradation_rows.append({
                'race': race_id, 'event': event_name, 'compound': compound, **d
            })
        if pit_loss['median_pit_loss'] is not None:
            print(f"  Pit loss (green): {pit_loss['median_pit_loss']:.2f}s "
                  f"(std {pit_loss['std']:.2f}, n={pit_loss['n_stops']} stops)")
            all_pit_loss_rows.append({'race': race_id, 'event': event_name, **pit_loss})
        elif 'rejected_implausible_median' in pit_loss:
            print(f"  Pit loss (green): REJECTED, implausible median "
                  f"{pit_loss['rejected_implausible_median']:.2f}s from n={pit_loss['n_stops']} stops "
                  f"(likely a disrupted race corrupting the expected-pace baseline)")
        else:
            print("  Pit loss (green): no valid stops")
        if sc_pit_loss['median_pit_loss'] is not None:
            print(f"  Pit loss (SC/VSC): {sc_pit_loss['median_pit_loss']:.2f}s "
                  f"(std {sc_pit_loss['std']:.2f}, n={sc_pit_loss['n_stops']} stops)")
            all_sc_pit_loss_rows.append({'race': race_id, 'event': event_name, **sc_pit_loss})
        else:
            print("  Pit loss (SC/VSC): no valid stops")
        print()

    deg_df = pd.DataFrame(all_degradation_rows)
    pit_df = pd.DataFrame(all_pit_loss_rows)
    sc_pit_df = pd.DataFrame(all_sc_pit_loss_rows)
    deg_df.to_csv('data/degradation_summary.csv', index=False)
    pit_df.to_csv('data/pit_loss_summary.csv', index=False)
    sc_pit_df.to_csv('data/sc_pit_loss_summary.csv', index=False)
    print(f"Saved data/degradation_summary.csv ({len(deg_df)} rows)")
    print(f"Saved data/pit_loss_summary.csv ({len(pit_df)} rows)")
    print(f"Saved data/sc_pit_loss_summary.csv ({len(sc_pit_df)} rows)")
    pd.DataFrame(all_neutralised_stops).to_csv('data/neutralised_pit_stops.csv', index=False)
    pd.DataFrame(all_compression_obs).to_csv('data/sc_gap_compression_obs.csv', index=False)
    ratios = neutralised_pit_ratios()
    print(f"SC/VSC pit loss as a share of green (median over individual stops): SC {ratios['SC']:.2f}, "
          f"VSC {ratios['VSC']:.2f} ({len(all_neutralised_stops)} stops)")
    print(f"Gap compression through a full safety car: gap after = {sc_gap_compression():.2f} x gap before "
          f"({len(all_compression_obs)} car observations)")

    # Team-specific pit-loss adjustment: a team's per-race stop count is far
    # too small (2 cars, 1-2 stops each) to fit a stable per-(circuit, team)
    # constant directly. Pit-crew execution speed is also largely
    # circuit-independent (only the pit-lane transit time is circuit-
    # specific), so instead pool each stop's deviation from that race's own
    # circuit-wide pit-loss constant across ALL of a team's stops, at every
    # circuit -- giving a much bigger, more stable sample per team. Applied
    # downstream as: team_pit_loss(circuit, team) = circuit_pit_loss(circuit)
    # + team_adjustment(team).
    stops_df = pd.DataFrame(all_stop_records)
    # Per-stop records too, so 12_as_of_context.py can recompute the
    # adjustment from only the races before a given one.
    stops_df.to_csv('data/pit_stop_losses.csv', index=False)
    team_adj = team_pit_loss_adjustment(stops_df)
    team_adj.to_csv('data/team_pit_loss_adjustment.csv', index=False)
    print(f"\nTeam pit-loss adjustment (negative = faster than field average):")
    print(team_adj.to_string(index=False))
    print(f"Saved data/team_pit_loss_adjustment.csv ({len(team_adj)} teams, "
          f"min {MIN_TEAM_STOPS} stops to be included)")


if __name__ == '__main__':
    main()
