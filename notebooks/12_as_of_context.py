"""
As-of-race context for the live tools (09_streaming_consumer.py and the
dashboard on top of it, and 10_live_risk_comparison.py's main()):
everything they need, built only from races strictly before the one being
replayed -- what a team would actually have had the morning of that race.

Replaying a historical race with models trained on every race flatters
the result: the trigger classifier, tyre model, pit-loss constants and
SC/VSC rates would all have seen how that very race (and later ones)
played out. Instead, as of race R:
  - trigger classifier (05): retrained on earlier races' features, with
    the circuit SC/VSC-probability feature recomputed from earlier races
    only (05's own version is leave-one-race-out over ALL races), and its
    decision threshold chosen on the last season of those earlier races.
  - tyre model (03): retrained on earlier races' clean laps, and queried
    with the latest observed weather held flat for the laps still to
    come -- not the race's real future weather -- and stint number at the
    earlier races' median.
  - green pit loss: median over earlier visits to this circuit (02's
    circuit_of: by track, not event name); for a
    first visit, the median over all earlier races. SC and VSC pit loss:
    green x each kind's pooled ratio over earlier races' stops (02's
    neutralised_pit_ratios); full-SC gap compression beta from earlier
    races' safety cars (02's sc_gap_compression). Previously the
    race's own measured pit loss, i.e. from the very stops being replayed.
  - team pit-loss adjustment: 02's pooled deviation, over earlier races'
    stops only.
  - SC and VSC hazards: 10's build_neutral_hazard(before_race=R).
  - compounds: all three dry compounds (every race has all three
    available), not "the compounds this race turned out to use".
Trained models are cached per race under data/models/as_of/. The offline
backtest (04/06) is unaffected: it already holds out its test races.

Needs 02, 03 and 05 to have been run -- they write the per-stop, silver
and feature tables this reads.
"""
import importlib.util
import joblib
import pandas as pd
from pathlib import Path


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dpl = _load('degradation_pitloss_lib', '02_fixed_degradation_pitloss.py')
dm = _load('degradation_model_lib', '03_ml_degradation_model.py')
tc = _load('trigger_classifier_lib', '05_trigger_classifier.py')
lr = _load('live_risk_lib', '10_live_risk_comparison.py')

BRONZE_DIR = Path('data/bronze')
CACHE_DIR = Path('data/models/as_of')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
FEW_PRIOR_RACES = 20  # below this, flag the replay's models as thinly trained


def prior_races(race_id: str) -> list:
    """Race ids strictly before race_id (ids sort chronologically)."""
    return sorted(f.stem.replace('_laps', '') for f in BRONZE_DIR.glob('*_laps.parquet')
                  if f.stem.replace('_laps', '') < race_id)


def circuit_sc_prob_curve(prior: pd.DataFrame, circuit: str, n_laps: int) -> dict:
    """{lap: P(SC/VSC active)} for this circuit from earlier races, by
    race-progress decile -- the same quantity 05's add_circuit_sc_prob()
    computes, falling back to the all-circuit decile rate for a first visit."""
    decile = (prior['race_progress'] * 10).clip(0, 9).astype(int)
    global_rate = prior.groupby(decile)['sc_vsc_active'].mean()
    event_rows = prior[prior['circuit'] == circuit]
    event_rate = (event_rows.groupby([decile[event_rows.index], 'race'])['sc_vsc_active'].mean()
                  .groupby(level=0).mean())
    curve = {}
    for lap in range(1, n_laps + 1):
        d = min(int(lap / n_laps * 10), 9)
        curve[lap] = float(event_rate.get(d, global_rate.get(d, 0.0)))
    return curve


def trigger_classifier_as_of(race_id: str, n_laps: int) -> dict | None:
    cache = CACHE_DIR / f"{race_id}_trigger.joblib"
    if cache.exists():
        return joblib.load(cache)
    feats = pd.read_parquet('data/silver/trigger_classifier_features.parquet')
    prior = feats[feats['race'] < race_id].drop(columns='circuit_sc_prob').reset_index(drop=True)
    if prior.empty:
        return None
    # Circuit SC/VSC rates use every earlier season; the classifier itself
    # trains on the same seasons as 05's.
    prior = tc.add_circuit_sc_prob(prior)
    prior = prior[prior['race'].str[:4].astype(int) >= tc.TRAIN_FROM_SEASON].reset_index(drop=True)
    if prior.empty:
        return None
    prior['own_compound'] = prior['own_compound'].astype('category')
    # Same procedure as 05's deployed model: threshold chosen on the last
    # season of these earlier races, then refit on all of them.
    model, threshold, _ = tc.train_with_threshold(prior)
    bundle = {'model': model, 'feature_cols': tc.FEATURE_COLS, 'threshold': threshold,
              'compound_categories': prior['own_compound'].cat.categories.tolist(),
              'sc_prob_by_lap': circuit_sc_prob_curve(prior, dpl.circuit_of(race_id), n_laps),
              'n_races': int(prior['race'].nunique())}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache)
    return bundle


def degradation_model_as_of(race_id: str) -> dict | None:
    cache = CACHE_DIR / f"{race_id}_degradation.joblib"
    if cache.exists():
        return joblib.load(cache)
    silver = pd.read_parquet(dm.SILVER_DIR / 'laps_ml_features.parquet')
    prior = silver[(silver['race'] < race_id)
                   & (silver['race'].str[:4].astype(int) >= dm.TRAIN_FROM_SEASON)].copy()
    if prior.empty:
        return None
    for c in dm.CAT_FEATURES:
        prior[c] = prior[c].astype('category')
    model = dm.new_regressor()
    model.fit(prior[dm.FEATURE_COLS], prior['RelativePace'], sample_weight=dm.era_weights(prior['race']),
              categorical_feature=dm.CAT_FEATURES)
    bundle = {'model': model, 'feature_cols': dm.FEATURE_COLS,
              'compound_categories': prior['Compound'].cat.categories.tolist(),
              'circuit_categories': prior['Circuit'].cat.categories.tolist(),
              'median_stint': float(prior['Stint'].median()),
              'n_races': int(prior['race'].nunique())}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache)
    return bundle


def live_pace_lookup(bundle: dict, race_id: str, n_laps: int, now: pd.Timedelta, compounds: list) -> dict:
    """{compound: {(lap, age): relative pace}} using only weather observed
    up to session time `now`, held flat for every lap."""
    weather = pd.read_parquet(BRONZE_DIR / f"{race_id}_weather.parquet").sort_values('Time')
    observed = weather[weather['Time'] <= now]
    latest = (observed if len(observed) else weather).iloc[-1 if len(observed) else 0]
    cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
    lap_weather = pd.DataFrame([latest[cols].to_dict()] * n_laps, index=range(1, n_laps + 1))
    circuit = dpl.circuit_of(race_id)
    return dm.predict_pace_grid(bundle, circuit, n_laps, lap_weather, bundle['median_stint'], compounds,
                                dpl.era_flags(race_id), dm.usable_tyre_life(circuit, compounds, before_race=race_id))


def pit_losses_as_of(race_id: str) -> dict | None:
    pit = pd.read_csv('data/pit_loss_summary.csv')
    prior = pit[pit['race'] < race_id].dropna(subset=['median_pit_loss'])
    if prior.empty:
        return None
    same = prior[prior['race'].map(dpl.circuit_of) == dpl.circuit_of(race_id)]
    if len(same):
        green = float(same['median_pit_loss'].median())
        source = f"median of {len(same)} earlier visit(s) to this circuit"
    else:
        green = float(prior['median_pit_loss'].median())
        source = "first visit to this circuit -- median of all earlier races"

    # SC and VSC: green x each kind's pooled ratio over earlier races'
    # individual stops -- per-race and per-circuit estimates are too noisy
    # (see 02's neutralised_pit_ratios).
    ratios = dpl.neutralised_pit_ratios(before_race=race_id)
    return {'green_pit_loss': green, 'sc_pit_loss': green * ratios['SC'], 'vsc_pit_loss': green * ratios['VSC'],
            'pit_loss_source': source}


def team_adjustment_as_of(race_id: str) -> pd.Series:
    stops = pd.read_csv('data/pit_stop_losses.csv')
    adj = dpl.team_pit_loss_adjustment(stops[stops['race'] < race_id])
    return adj.set_index('team')['adjustment_s']


def live_context(race_id: str, event_name: str, n_laps: int) -> dict:
    """Everything the live tools need as of race_id. 'trigger' and
    'risk' are None when there are no earlier races to learn from (or no
    pit-loss history), with 'problems' saying why."""
    n_prior = len(prior_races(race_id))
    trigger = trigger_classifier_as_of(race_id, n_laps)
    deg = degradation_model_as_of(race_id)
    losses = pit_losses_as_of(race_id)
    problems = []
    if trigger is None:
        problems.append("no earlier races to train the trigger classifier on")
    risk = None
    if deg is None or losses is None:
        problems.append("no earlier races to estimate tyre wear and pit loss from")
    else:
        risk = {**losses, 'deg_bundle': deg, 'event_name': event_name,
                'compounds': [c for c in DRY_COMPOUNDS if c in deg['compound_categories']],
                'team_adj': team_adjustment_as_of(race_id),
                'neutral_hazard': lr.build_neutral_hazard(before_race=race_id),
                'sc_beta': dpl.sc_gap_compression(before_race=race_id),
                'circuit_seen': dpl.circuit_of(race_id) in deg['circuit_categories']}
    return {'trigger': trigger, 'risk': risk, 'n_prior_races': n_prior,
            'thin_history': n_prior < FEW_PRIOR_RACES, 'problems': problems}
