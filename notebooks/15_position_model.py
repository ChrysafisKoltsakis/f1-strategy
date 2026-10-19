"""
Position model: what a strategy does to finishing position and points,
not just race time.

The cost model (04/06/10) scores a strategy by the car's own race time, so
track position has no value in it: stopping early to jump a car (the
undercut), or staying out to keep a place a slower car can't take back,
are invisible. This adds the missing layer, in the spirit of Heilmeier et
al.'s lap-wise race simulation, but with every rate measured from our own
data:

  1. Overtaking. Every green lap where a car ran within 2.5s of the car
     ahead (no pit stop by either, no SC/VSC, lap 4+): did it get past by
     the end of the lap? A logistic model of that, by gap, pace advantage
     (difference in the two cars' potential pace -- pace offset + tyre
     model for the tyres each was on -- not their observed lap times,
     which for a car stuck behind just repeat the car ahead's), tyre-age
     difference and circuit (one intercept per circuit, shrunk towards the
     average by L2 -- an unseen circuit gets the average), fitted on earlier
     races only for each season it's used on.
  2. Counterfactual race. One car ("ego") runs a candidate strategy; every
     other car replays its real laps. Ego's lap times are the field's
     median lap that lap + ego's own pace offset (median over its clean
     laps of real minus predicted relative pace) + the tyre model's
     relative pace for its compound and tyre age, with stops priced as in
     06 (typed SC/VSC prices, one cheap stop per neutralisation, deficit
     compression under a full SC). Positions change freely through the pit
     lane; on track, getting past the car ahead needs a pass, drawn from
     the overtaking model, and a failed one leaves ego held FOLLOW_GAP_S
     behind it. The same applies the other way: a car behind that would
     get past ego on track has to pass it, or loses the time stuck behind
     (kept for the rest of the race). Monte Carlo over the pass draws.
  3. Points. Finishing position -> points (25-18-15-12-10-8-6-4-2-1), and
     the championship rival's position and points in the same simulated
     race (it can lose a place to ego, or be held up by it).

Validated first: simulating each driver's REAL strategy should reproduce
its real finishing position -- see main().

Limitation: the other cars don't react to ego (they keep their real
strategies), and the simulator knows their real future laps -- fine for
scoring a strategy after the fact, not something a live decision could
know.

Run with: python notebooks/15_position_model.py
"""
import importlib.util
import copy
import math
import zlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dpl = _load('degradation_pitloss_lib', '02_fixed_degradation_pitloss.py')
dm = _load('degradation_model_lib', '03_ml_degradation_model.py')
ds = _load('default_strategy_lib', '14_default_strategy.py')

BRONZE_DIR = Path('data/bronze')
EVENTS_FILE = Path('data/overtaking_events.csv')
DRY = {'SOFT', 'MEDIUM', 'HARD'}
POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
CLOSE_GAP_S = 2.5          # closer than this at a lap's end counts as racing the car ahead
FOLLOW_GAP_S = 0.5         # a car that fails to pass sits this far behind (overwritten from data in main)
N_SIMS = 60


# ---------------------------------------------------------------------------
# 1. Overtaking
# ---------------------------------------------------------------------------

def _race_frame(race_id: str) -> pd.DataFrame | None:
    """Laps with session time / lap time / relative pace, or None for a race
    with wet tyres in it."""
    return prepare_laps(pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet"))


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame | None:
    """Bronze-shaped laps (a whole race, or live: the laps so far) ->
    the columns the simulator uses; None if wet tyres were used."""
    if laps['Compound'].isin(['INTERMEDIATE', 'WET']).any():
        return None
    laps = laps.copy()
    laps['LapNumber'] = laps['LapNumber'].astype(int)
    laps['t'] = laps['Time'].dt.total_seconds()
    laps['lt'] = laps['LapTime'].dt.total_seconds()
    laps['rel'] = laps['lt'] - laps.groupby('LapNumber')['lt'].transform('median')
    laps['pit'] = laps['PitInTime'].notna() | laps['PitOutTime'].notna()
    laps['green'] = laps['TrackStatus'].astype(str) == '1'
    return laps


def race_pass_events(replay: 'RaceReplay') -> list:
    """Close-following green laps in one race. Pace advantage is each car's
    *potential* relative pace (its pace offset + the tyre model for its
    compound and tyre age), the same quantity the simulator uses -- not
    the observed recent lap times, which for a car stuck behind another
    just repeat that car's pace."""
    if not replay.ok:
        return []
    laps, pot = replay.laps, replay.potential()
    by = {d: g.set_index('LapNumber') for d, g in laps.groupby('Driver')}
    green = laps.groupby('LapNumber')['green'].all()
    rows = []
    for lap in range(4, replay.n_laps + 1):
        if not green.get(lap, False):
            continue
        prev = laps[(laps['LapNumber'] == lap - 1)].dropna(subset=['Position', 't']).sort_values('Position')
        drivers, ts = prev['Driver'].tolist(), prev['t'].tolist()
        for i in range(1, len(drivers)):
            a, b = drivers[i], drivers[i - 1]
            gap = ts[i] - ts[i - 1]
            if not 0 < gap < CLOSE_GAP_S or lap not in by[a].index or lap not in by[b].index:
                continue
            ra, rb = by[a].loc[lap], by[b].loc[lap]
            if ra['pit'] or rb['pit'] or by[a].loc[lap - 1, 'pit'] or by[b].loc[lap - 1, 'pit']:
                continue
            if pd.isna(ra['Position']) or pd.isna(rb['Position']):
                continue
            rows.append({'race': replay.race_id, 'circuit': replay.circuit, 'lap': lap, 'gap': gap,
                         'pace_adv': pot.loc[b, lap] - pot.loc[a, lap],
                         'age_diff': rb['TyreLife'] - ra['TyreLife'],
                         'passed': int(ra['Position'] < rb['Position']),
                         'gap_after': ra['t'] - rb['t']})
    return rows


def collect_pass_events() -> pd.DataFrame:
    """Uses the full tyre model (03) for potential pace, so events in an
    early season are described with a model that has seen later ones --
    a mild leak into the pass model's inputs (not its labels)."""
    ml_bundle = dm.load_degradation_model()
    rows = []
    for f in sorted(BRONZE_DIR.glob('*_laps.parquet')):
        race_id = f.stem.replace('_laps', '')
        degradation = dm.build_ml_pace_lookup(race_id, ml_bundle, ['SOFT', 'MEDIUM', 'HARD'])
        if degradation:
            rows += race_pass_events(RaceReplay(race_id, degradation))
    ev = pd.DataFrame(rows).dropna(subset=['pace_adv'])
    ev['age_diff'] = ev['age_diff'].fillna(0)
    ev.to_csv(EVENTS_FILE, index=False)
    return ev


def _design(gap, pace_adv, age_diff) -> np.ndarray:
    g = np.asarray(gap, dtype=float)
    p = np.clip(np.asarray(pace_adv, dtype=float), -2, 3)
    a = np.clip(np.asarray(age_diff, dtype=float), -30, 30) / 10
    return np.column_stack([g, g ** 2, p, p ** 2, p * g, p * g ** 2, a])


def fit_pass_model(events: pd.DataFrame, before_race: str | None = None, C: float = 1.0) -> dict:
    """Logistic P(pass this lap); per-circuit intercepts via one-hot columns
    under the same L2 penalty, so thin circuits shrink to the average."""
    ev = events if before_race is None else events[events['race'] < before_race]
    circuits = sorted(ev['circuit'].unique())
    X = np.hstack([_design(ev['gap'], ev['pace_adv'], ev['age_diff']),
                   (ev['circuit'].values[:, None] == np.array(circuits)[None, :]).astype(float)])
    model = LogisticRegression(C=C, max_iter=2000).fit(X, ev['passed'])
    n_num = X.shape[1] - len(circuits)
    return {'coef': model.coef_[0][:n_num], 'intercept': float(model.intercept_[0]),
            'circuit_effect': dict(zip(circuits, model.coef_[0][n_num:])), 'n_events': len(ev)}


def pass_prob_arr(pm: dict, circuit: str, gap, pace_adv, age_diff) -> np.ndarray:
    """pass_prob over arrays of any (matching) shape."""
    g = np.asarray(gap, dtype=float)
    shape = g.shape
    x = _design(g.ravel(), np.asarray(pace_adv, dtype=float).ravel(), np.asarray(age_diff, dtype=float).ravel())
    z = pm['intercept'] + x @ pm['coef'] + pm['circuit_effect'].get(circuit, 0.0)
    return (1.0 / (1.0 + np.exp(-z))).reshape(shape)


def pass_prob(pm: dict, circuit: str, gap: float, pace_adv: float, age_diff: float) -> float:
    x = _design([gap], [pace_adv], [age_diff])[0]
    z = pm['intercept'] + float(x @ pm['coef']) + pm['circuit_effect'].get(circuit, 0.0)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# 2. Counterfactual race
# ---------------------------------------------------------------------------

class RaceReplay:
    """One race's real timing for every car, plus what's needed to run a
    counterfactual strategy for any one of them."""

    def __init__(self, race_id: str, degradation: dict, neutral: dict | None = None,
                 laps: pd.DataFrame | None = None, n_laps: int | None = None):
        """The whole real race, or with `laps` (prepared, e.g. only the laps
        run so far) and the scheduled `n_laps`, a partial one -- see
        forecast_from()."""
        self.race_id, self.degradation, self.neutral = race_id, degradation, neutral
        self.circuit = dpl.circuit_of(race_id)
        partial = laps is not None
        laps = laps if partial else _race_frame(race_id)
        self.ok = laps is not None
        self._offsets = {}
        if not self.ok:
            return
        self.n_laps = int(n_laps or laps['LapNumber'].max())
        self.field = laps.groupby('LapNumber')['lt'].median().reindex(range(1, self.n_laps + 1))
        self.drivers = sorted(laps['Driver'].unique())
        self.T = laps.pivot_table(index='Driver', columns='LapNumber', values='t').reindex(
            index=self.drivers, columns=range(1, self.n_laps + 1))
        self.lt = laps.pivot_table(index='Driver', columns='LapNumber', values='lt').reindex(
            index=self.drivers, columns=range(1, self.n_laps + 1))
        self.pit = laps.pivot_table(index='Driver', columns='LapNumber', values='pit', aggfunc='max').reindex(
            index=self.drivers, columns=range(1, self.n_laps + 1)).fillna(False).astype(bool)
        self.age = laps.pivot_table(index='Driver', columns='LapNumber', values='TyreLife').reindex(
            index=self.drivers, columns=range(1, self.n_laps + 1))
        self.laps = laps
        if partial:
            last = int(laps['LapNumber'].max())
            self.finished = set(laps.loc[laps['LapNumber'] >= last - 1, 'Driver'])   # still running
            self.real_position = {}
            return
        res = pd.read_parquet(BRONZE_DIR / f"{race_id}_results.parquet")
        self.finished = set(res.loc[res['Status'].isin(['Finished', 'Lapped']) | res['Status'].astype(str).str.startswith('+'),
                                    'Driver'])
        self.real_position = res.set_index('Driver')['Position'].to_dict()

    def pace_offset(self, driver: str) -> float:
        """Driver/car pace vs the tyre model: median over clean green laps
        of real relative pace minus predicted."""
        if driver in self._offsets:
            return self._offsets[driver]
        d = self.laps[(self.laps['Driver'] == driver) & self.laps['green'] & ~self.laps['pit']
                      & self.laps['Compound'].isin(DRY) & (self.laps['LapNumber'] > 1) & self.laps['TyreLife'].notna()]
        diffs = []
        for c, l, a, rel in zip(d['Compound'], d['LapNumber'], d['TyreLife'], d['rel']):
            pred = self.degradation.get(c, {}).get((int(l), int(a)))
            if pred is not None and pd.notna(rel):
                diffs.append(rel - pred)
        self._offsets[driver] = float(np.median(diffs)) if diffs else 0.0
        return self._offsets[driver]

    def potential(self) -> pd.DataFrame:
        """Every car's potential relative pace by lap on the tyres it really
        ran: pace offset + tyre-model prediction (NaN where unknown)."""
        if getattr(self, '_potential', None) is None:
            pot = np.full((len(self.drivers), self.n_laps), np.nan)
            row = {d: i for i, d in enumerate(self.drivers)}
            off = {d: self.pace_offset(d) for d in self.drivers}
            L = self.laps.dropna(subset=['TyreLife'])
            for d, c, l, a in zip(L['Driver'], L['Compound'], L['LapNumber'], L['TyreLife']):
                pred = self.degradation.get(c, {}).get((int(l), int(a)))
                if pred is not None and 1 <= l <= self.n_laps:
                    pot[row[d], int(l) - 1] = off[d] + pred
            self._potential = pd.DataFrame(pot, index=self.drivers, columns=range(1, self.n_laps + 1))
        return self._potential

    def forecast_from(self, lap: int, plan: dict | None, pit_loss_by_driver: dict) -> 'RaceReplay':
        """The race as it would be forecast at the end of `lap`: every car's
        real laps up to it, then for each car still running its potential
        pace (pace offset from its laps so far + tyre model) on the default
        plan, re-spaced to the stops it has really made (14's PlanFollower),
        stopping at its team's pit loss; no future SC/VSC. Lap times are
        held at the field's recent median -- the same for every car, so it
        doesn't change who's ahead."""
        past = self.laps[self.laps['LapNumber'] <= lap]
        neutral = None
        if self.neutral is not None:
            kinds = {l: k for l, k in self.neutral['kinds'].items() if l <= lap}
            neutral = {**self.neutral, 'kinds': kinds,
                       'period': {l: v for l, v in self.neutral['period'].items() if l <= lap},
                       'compress_after': frozenset(l for l in self.neutral['compress_after'] if l <= lap)}
        fc = RaceReplay(self.race_id, self.degradation, neutral, laps=past, n_laps=self.n_laps)
        n = self.n_laps
        recent = past[past['green']].groupby('LapNumber')['lt'].median()
        field_now = float(recent.iloc[-5:].median()) if len(recent) else float(fc.field.dropna().median())
        fc.field.loc[lap + 1:] = field_now
        pot = fc.potential().values.copy()
        T, LT, AGE, PIT = fc.T.values.copy(), fc.lt.values.copy(), fc.age.values.copy(), fc.pit.values.copy()
        median_loss = float(np.median(list(pit_loss_by_driver.values()))) if pit_loss_by_driver else 22.0
        for i, d in enumerate(fc.drivers):
            if d not in fc.finished or np.isnan(T[i, lap - 1]) or np.isnan(AGE[i, lap - 1]):
                fc.finished.discard(d)
                continue
            dl = past[past['Driver'] == d].sort_values('LapNumber')
            real_stops, comps_so_far = real_strategy_from(dl)
            follower = ds.PlanFollower(plan, n) if plan else None
            for st in real_stops:
                if follower:
                    follower.record_stop(st)
            future = sorted(max(st, lap) for st, _ in follower.pending) if follower else []
            comps = stint_compounds(comps_so_far, len(future), plan['compounds'] if plan else ['HARD'])
            off = fc.pace_offset(d)
            loss = pit_loss_by_driver.get(d, median_loss)
            t, c, a, stint = T[i, lap - 1], comps[len(comps_so_far) - 1], AGE[i, lap - 1], len(comps_so_far) - 1
            for l in range(lap + 1, n + 1):
                extra = 0.0
                if (l - 1) in future:
                    stint, a = stint + 1, 0.0
                    c, extra = comps[stint], loss
                    PIT[i, l - 2] = PIT[i, l - 1] = True
                a += 1
                rel = rel_at(self.degradation, c, l, a)
                if rel is None:   # a compound the race's tyre model doesn't cover: nearest one it does
                    rel = next((v for c2 in ('HARD', 'MEDIUM', 'SOFT')
                                if (v := rel_at(self.degradation, c2, l, a)) is not None), 0.0)
                t += field_now + off + rel + extra
                T[i, l - 1], LT[i, l - 1], AGE[i, l - 1], pot[i, l - 1] = t, field_now + off + rel + extra, a, off + rel
        cols = range(1, n + 1)
        fc.T, fc.lt = pd.DataFrame(T, fc.drivers, cols), pd.DataFrame(LT, fc.drivers, cols)
        fc.age, fc.pit = pd.DataFrame(AGE, fc.drivers, cols), pd.DataFrame(PIT, fc.drivers, cols)
        fc._potential = pd.DataFrame(pot, fc.drivers, cols)
        return fc

    def ego_potential(self, driver: str, stops: list, compounds: list, pit_loss: float, offset: float):
        """Ego's free-air lap times from lap 2 on, for in-laps `stops` and
        stint compounds `compounds` (compounds[0] = start). Returns
        (lap time, pit-lap flags, tyre age, potential relative pace), each
        an array indexed by lap."""
        n = self.n_laps
        lap_time, pit_lap, ages = np.full(n + 1, np.nan), np.zeros(n + 1, bool), np.zeros(n + 1)
        pot_rel = np.full(n + 1, np.nan)
        stop_set = {int(s) + 1: i + 1 for i, s in enumerate(sorted(stops))}   # out-lap -> stint index
        start_age = self.age.loc[driver, 1] if pd.notna(self.age.loc[driver, 1]) else 1.0
        stint, age, last_p = 0, start_age - 1, -1
        deficit = 0.0
        for lap in range(2, n + 1):
            extra = 0.0
            if lap in stop_set:
                stint, age = stop_set[lap], 0
                price, last_p = dpl.stop_price(self.neutral, lap - 1, pit_loss, last_p)
                extra += price
                pit_lap[lap - 1] = pit_lap[lap] = True
            age += 1
            rel = rel_at(self.degradation, compounds[min(stint, len(compounds) - 1)], lap, age)
            if rel is None:
                return None
            lt = float(self.field.get(lap, np.nan)) + offset + rel + extra
            if self.neutral is not None:
                # Under a full SC the deficit built up before it mostly
                # disappears, as in 06's replay (02's sc_gap_compression).
                deficit += offset + rel + extra
                if lap in self.neutral['compress_after']:
                    lt += (self.neutral['beta'] - 1.0) * deficit
                    deficit *= self.neutral['beta']
            lap_time[lap], ages[lap], pot_rel[lap] = lt, age, offset + rel
        return lap_time, pit_lap, ages, pot_rel

    def simulate(self, driver: str, stops: list, compounds: list, pit_loss: float, pm: dict,
                 rival: str | None = None, n_sims: int = N_SIMS, seed: int = 0, offset: float | None = None) -> dict | None:
        """Monte Carlo finishing position (and points) of `driver` on the
        candidate strategy, and of `rival` in the same races."""
        if driver not in self.finished or pd.isna(self.T.loc[driver, 1]):
            return None
        offset = self.pace_offset(driver) if offset is None else offset
        pot = self.ego_potential(driver, stops, compounds, pit_loss, offset)
        if pot is None:
            return None
        lap_time, ego_pit, ego_age, ego_pot_rel = pot
        n, S = self.n_laps, n_sims
        others = [d for d in self.drivers if d != driver]
        T = self.T.loc[others].values                       # others' cumulative times
        pit = self.pit.loc[others].values
        age = self.age.loc[others].values
        pot_o = self.potential().loc[others].values        # others' potential relative pace by lap
        idx = {d: i for i, d in enumerate(others)}
        rng = np.random.default_rng(seed)
        neutral_laps = set(self.neutral['kinds']) if self.neutral else set()
        compress_after = self.neutral['compress_after'] if self.neutral else frozenset()
        ego_laps = int(self.T.loc[driver].notna().sum())    # ego's real distance (lapped stays lapped)
        done = (~np.isnan(T)).sum(axis=1)
        fin = np.array([d in self.finished for d in others])
        # All Monte Carlo draws at once: rows = draws, columns = other cars.
        delay = np.zeros((S, len(others)))                 # time others lose stuck behind ego
        C = np.full(S, float(self.T.loc[driver, 1]))
        C_hist = np.full((S, n + 1), np.nan)
        C_hist[:, 1] = C
        for lap in range(2, n + 1):
            j = lap - 1                                    # column of this lap
            prev_C, C_pot = C, C + lap_time[lap]
            Tprev, Tnow = T[:, j - 1][None, :] + delay, T[:, j][None, :] + delay
            on_track = (~np.isnan(T[:, j]) & ~np.isnan(T[:, j - 1]) & ~pit[:, j] & ~pit[:, j - 1])[None, :]
            neutral = lap in neutral_laps or (lap - 1) in neutral_laps
            if not ego_pit[lap]:
                k_rel = np.nan_to_num(pot_o[:, j])
                k_age = age[:, j]
                age_gap = np.where(np.isnan(k_age), 0.0, k_age - ego_age[lap])
                # Cars ego would get past on track this lap: nearest first,
                # stopping at the first failed pass (at most three a lap).
                cand = on_track & (Tprev < prev_C[:, None]) & (Tnow > C_pot[:, None]) \
                    & (prev_C[:, None] - Tprev <= CLOSE_GAP_S)
                active = np.ones(S, bool)
                for _ in range(3):
                    has = active & cand.any(axis=1)
                    if not has.any():
                        break
                    k = np.where(cand, Tprev, -np.inf).argmax(axis=1)
                    rows = np.arange(S)
                    p = np.zeros(S) if neutral else pass_prob_arr(
                        pm, self.circuit, prev_C - Tprev[rows, k], k_rel[k] - ego_pot_rel[lap], age_gap[k])
                    fail = has & (rng.random(S) >= p)
                    C_pot = np.where(fail, np.maximum(C_pot, Tnow[rows, k] + FOLLOW_GAP_S), C_pot)
                    active &= ~fail
                    cand[rows, k] = False
                # Cars that would get past ego on track this lap.
                behind = on_track & (Tprev > prev_C[:, None]) & (Tnow < C_pot[:, None]) \
                    & (Tprev - prev_C[:, None] <= CLOSE_GAP_S)
                if behind.any():
                    p = np.zeros(behind.shape) if neutral else pass_prob_arr(
                        pm, self.circuit, Tprev - prev_C[:, None],
                        ego_pot_rel[lap] - k_rel[None, :] + 0 * Tprev, (-age_gap)[None, :] + 0 * Tprev)
                    stuck = behind & (rng.random(behind.shape) >= p)
                    delay += np.where(stuck, C_pot[:, None] + FOLLOW_GAP_S - Tnow, 0.0)
            if lap in compress_after:
                delay *= self.neutral['beta']   # an SC erases most of the time lost behind ego too
            C = C_pot
            C_hist[:, lap] = C
        # Classified order: laps completed, then time at that distance.
        last = np.where(done > 0, done - 1, 0)
        final = np.where(done[None, :] > 0, T[np.arange(len(others)), last][None, :] + delay, np.inf)
        ego_t = C_hist[:, ego_laps]
        ahead_of_ego = fin[None, :] & ((done[None, :] > ego_laps) | ((done[None, :] == ego_laps) & (final < ego_t[:, None])))
        fin_pos = 1 + ahead_of_ego.sum(axis=1)
        pts = np.vectorize(lambda q: POINTS.get(int(q), 0))
        out = {'exp_position': float(fin_pos.mean()), 'exp_points': float(pts(fin_pos).mean()),
               'positions': fin_pos.tolist()}
        r = idx.get(rival) if rival is not None else None
        if r is not None and fin[r]:
            dr, fr = done[r], final[:, r]
            ahead_of_r = fin[None, :] & (np.arange(len(others)) != r)[None, :] & (
                (done[None, :] > dr) | ((done[None, :] == dr) & (final < fr[:, None])))
            ego_ahead = (ego_laps > dr) | ((ego_laps == dr) & (ego_t < fr))
            rival_pos = 1 + ahead_of_r.sum(axis=1) + ego_ahead
            out['rival_exp_position'] = float(rival_pos.mean())
            out['rival_exp_points'] = float(pts(rival_pos).mean())
        return out


def rel_at(degradation: dict, compound: str, lap: int, age: float) -> float | None:
    """Tyre-model relative pace; ages past the precomputed grid use its
    last age (the safety-net drop-off is already steep there)."""
    grid = degradation.get(compound)
    if not grid:
        return None
    return grid.get((int(lap), int(min(age, dm.MAX_AGE_GRID))))


def stint_compounds(comps_so_far: list, n_future: int, plan_compounds: list) -> list:
    """Compounds so far + one per future stint: the plan's compound for
    that stint, but an unused one while the two-compound rule isn't met."""
    comps, used = list(comps_so_far), set(comps_so_far)
    for _ in range(n_future):
        i = len(comps)
        want = plan_compounds[min(i, len(plan_compounds) - 1)]
        if len(used) < 2 and want in used:
            want = next(c for c in ('HARD', 'MEDIUM', 'SOFT') if c not in used)
        comps.append(want)
        used.add(want)
    return comps


def real_strategy(replay: RaceReplay, driver: str) -> tuple:
    """(stop in-laps, stint compounds) the driver really ran, with the same
    tyre-set-change rule as 04/14 (red-flag changes free)."""
    return real_strategy_from(replay.laps[replay.laps['Driver'] == driver].sort_values('LapNumber'))


def real_strategy_from(d: pd.DataFrame) -> tuple:
    prev = d.shift()
    new_set = ((d['Compound'] != prev['Compound'])
               | (d['TyreLife'] - prev['TyreLife'] != d['LapNumber'] - prev['LapNumber']))
    red = prev['TrackStatus'].astype(str).str.contains('5')
    stops = d[new_set & ~red].iloc[1:]
    return [int(l) - 1 for l in stops['LapNumber']], [d['Compound'].iloc[0]] + list(stops['Compound'])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def follow_gap(events: pd.DataFrame) -> float:
    """Median gap a quicker car (>0.5s/lap) that failed to pass was left at."""
    stuck = events[(events['passed'] == 0) & (events['gap'] < 1.0) & (events['pace_adv'] > 0.5)]
    return float(stuck['gap_after'].median())


def validate_pass_model(events: pd.DataFrame):
    """Each season 2023-2026 predicted by a model fitted on earlier races
    only, against the same model without circuit intercepts."""
    rows = []
    for season in range(2023, 2027):
        test = events[events['race'].str[:4].astype(int) == season]
        start = f"{season}_r00"
        if test.empty:
            continue
        pm = fit_pass_model(events, before_race=start)
        pm_flat = {**pm, 'circuit_effect': {}}
        flat = fit_pass_model(events[events['race'] < start].assign(circuit='all'))
        p = np.array([pass_prob(pm, c, g, a, d) for c, g, a, d in
                      zip(test['circuit'], test['gap'], test['pace_adv'], test['age_diff'])])
        p0 = np.array([pass_prob(flat, 'all', g, a, d) for g, a, d in
                       zip(test['gap'], test['pace_adv'], test['age_diff'])])
        rows.append({'season': season, 'events': len(test), 'pass_rate': test['passed'].mean(),
                     'auc': roc_auc_score(test['passed'], p), 'auc_no_circuit': roc_auc_score(test['passed'], p0),
                     'brier': brier_score_loss(test['passed'], p), 'brier_no_circuit': brier_score_loss(test['passed'], p0),
                     'mean_p': p.mean()})
    print("Overtaking model, each season predicted from earlier races only:")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


# Every pass succeeds: finishing order by modelled race time alone.
FREE_PASSING = {'coef': np.zeros(7), 'intercept': 50.0, 'circuit_effect': {}}


def race_setup(race_id: str, dp_race: pd.DataFrame, ml_bundle: dict, beta: float) -> dict:
    """Tyre-model lookup and one RaceReplay per team (stop prices differ
    by team), keyed by driver."""
    degradation = dm.build_ml_pace_lookup(race_id, ml_bundle, ['SOFT', 'MEDIUM', 'HARD'])
    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    replays = {}
    for driver, r in dp_race.iterrows():
        neutral = dpl.neutralisation(laps, float(r['team_sc_pit_loss_s']), float(r['team_vsc_pit_loss_s']), beta)
        replays[driver] = (RaceReplay(race_id, degradation, neutral), float(r['team_pit_loss_s']))
    return replays


def validate_simulator(events: pd.DataFrame, dp: pd.DataFrame, races: list, n_sims: int = 20) -> pd.DataFrame:
    """Simulate every finisher's REAL strategy and compare with where they
    really finished; the pass model for each season is fitted on earlier
    races only."""
    ml_bundle = dm.load_degradation_model()
    beta = dpl.sc_gap_compression()
    rows, models = [], {}
    for race_id in races:
        season = race_id[:4]
        if season not in models:
            models[season] = fit_pass_model(events, before_race=f"{season}_r00")
        dp_race = dp[dp['race'] == race_id].set_index('driver')
        if dp_race.empty:
            continue
        setups = race_setup(race_id, dp_race, ml_bundle, beta)
        for driver, (replay, pit_loss) in setups.items():
            if not replay.ok or driver not in replay.finished:
                continue
            stops, comps = real_strategy(replay, driver)
            offset = replay.pace_offset(driver)
            sim = replay.simulate(driver, stops, comps, pit_loss, models[season], n_sims=n_sims, offset=offset)
            free = replay.simulate(driver, stops, comps, pit_loss, FREE_PASSING, n_sims=1, offset=offset)
            if sim is None or free is None:
                continue
            rows.append({'race': race_id, 'driver': driver, 'real_position': replay.real_position.get(driver),
                         'sim_position': sim['exp_position'], 'free_position': free['exp_position']})
    out = pd.DataFrame(rows).dropna()
    for col, label in [('sim_position', 'with overtaking model'), ('free_position', 'race time only (free passing)')]:
        err = (out[col] - out['real_position']).abs()
        rho = out.groupby('race').apply(lambda g: g[col].corr(g['real_position'], method='spearman'), include_groups=False)
        print(f"  {label:32s} position MAE {err.mean():.2f}, within 1 place {(err <= 1).mean():.0%}, "
              f"median per-race Spearman {rho.median():.3f}")
    return out


RIVAL_WEIGHT = 1.0   # objective = E[points] + RIVAL_WEIGHT x E[points - championship rival's points]
# The position-aware policy: inside a planned stop's window, every stop lap
# from now to the window's end is raced against the forecast field
# (forecast_from), and pitting now has to beat the best later lap by
# POSITION_MARGIN in the objective -- below that, Monte Carlo noise and
# model error, so the plan stands. A small per-place term breaks ties
# outside the points, where the objective is flat.
DECISION_SIMS = 16
POSITION_MARGIN = 0.25
PLACE_TIE_BREAK = 0.1


def decision_value(sim: dict) -> float:
    pts = sim['exp_points']
    return pts + RIVAL_WEIGHT * (pts - sim.get('rival_exp_points', 0.0)) - PLACE_TIE_BREAK * sim['exp_position']


# The strategy-level policy (StrategyDecider): instead of only timing the
# default plan's stops, re-plan the whole rest of the race -- 0 to MAX_STOPS
# more stops, each count at the balanced ("freshest possible") timing and
# shifted earlier (undercut) or later (overcut) by TIMING_SHIFTS -- every
# REPLAN_EVERY laps and on every lap within NEAR_STOP of the next stop.
MAX_STOPS = 3
TIMING_SHIFTS = (-3, 0, 3, 6)
REPLAN_EVERY = 4
NEAR_STOP = 3
MIN_STINT = 8


def choose_compounds(degradation: dict, comps_so_far: list, stops: list, n_laps: int) -> list | None:
    """Compounds for every stint: the ones run so far, then for each later
    stint the compound the tyre model makes quickest over that stint's laps,
    an unused one on the last stint if the two-compound rule isn't met yet."""
    comps, used = list(comps_so_far), set(comps_so_far)
    edges = sorted(stops) + [n_laps]
    for i in range(len(comps_so_far), len(edges)):
        start, end = edges[i - 1] + 1, edges[i]
        options = [c for c in ('SOFT', 'MEDIUM', 'HARD') if c in degradation]
        if len(used) < 2 and i == len(edges) - 1:
            options = [c for c in options if c not in used] or options
        best, best_t = None, np.inf
        for c in options:
            t = 0.0
            for age, lap in enumerate(range(start, end + 1), start=1):
                v = rel_at(degradation, c, lap, age)
                if v is None:
                    t = np.inf
                    break
                t += v
            if t < best_t:
                best, best_t = c, t
        if best is None:
            return None
        comps.append(best)
        used.add(best)
    return comps


def candidate_plans(lap: int, done: list, comps_so_far: list, current: list, n_laps: int) -> set:
    """Future stop in-laps to consider at the end of `lap`: the current plan,
    and for each remaining stop count the balanced timing (equal stints over
    the current stint and the rest) with TIMING_SHIFTS on the next stop."""
    stint_start = done[-1] if done else 0
    last_ok = n_laps - 3
    plans = {tuple(sorted(current))}
    for k in range(0, MAX_STOPS + 1):
        if k == 0:
            if len(set(comps_so_far)) >= 2:
                plans.add(())
            continue
        base = stint_start + (n_laps - stint_start) / (k + 1)
        for shift in TIMING_SHIFTS:
            first = int(round(max(lap, base + shift)))
            stops = [int(round(first + i * (n_laps - first) / k)) for i in range(k)]
            stints = np.diff([first] + stops[1:] + [n_laps])
            if first > last_ok or stops[-1] > last_ok or (k > 1 and stints.min() < MIN_STINT):
                continue
            plans.add(tuple(stops))
    return plans


class StrategyDecider:
    """The strategy-level policy for one car: at the end of `lap`, race every
    candidate rest-of-race plan against the forecast field and switch to
    the best one if it beats the current plan by POSITION_MARGIN in the
    objective. Forecasts shared across the race's cars via `cache`."""

    def __init__(self, replay: RaceReplay, driver: str, plan: dict | None, pit_loss: float, neutral: dict | None,
                 pm: dict, rival: str | None, pit_loss_by_driver: dict, cache: dict):
        self.replay, self.driver, self.plan, self.pit_loss, self.neutral = replay, driver, plan, pit_loss, neutral
        self.pm, self.rival, self.pit_loss_by_driver, self.cache = pm, rival, pit_loss_by_driver, cache

    def forecast(self, lap: int) -> RaceReplay:
        if lap not in self.cache:
            self.cache[lap] = self.replay.forecast_from(lap, self.plan, self.pit_loss_by_driver)
        fc = copy.copy(self.cache[lap])
        fc.neutral = truncate_neutral(self.neutral, lap)
        return fc

    def __call__(self, lap: int, done: list, comps_so_far: list, current: list,
                 known_lap: int | None = None) -> list:
        fc = self.forecast(lap if known_lap is None else known_lap)
        if self.driver not in fc.finished:
            return current
        offset = fc.pace_offset(self.driver)
        seed = zlib.crc32(f"{self.replay.race_id}{self.driver}{lap}".encode())
        values = {}
        for future in candidate_plans(lap, done, comps_so_far, current, self.replay.n_laps):
            stops = sorted(done) + list(future)
            comps = choose_compounds(fc.degradation, comps_so_far, stops, self.replay.n_laps)
            if comps is None or len(set(comps)) < 2:
                continue
            sim = fc.simulate(self.driver, stops, comps, self.pit_loss, self.pm, rival=self.rival,
                              n_sims=DECISION_SIMS, seed=seed, offset=offset)
            if sim is not None:
                values[future] = decision_value(sim)
        if not values:
            return list(current)
        cur, best = tuple(sorted(current)), max(values, key=values.get)
        if cur not in values:          # the current plan has become impossible
            return list(best)
        return list(best) if values[best] - values[cur] >= POSITION_MARGIN else list(current)


def truncate_neutral(neutral: dict | None, lap: int) -> dict | None:
    if neutral is None:
        return None
    return {**neutral, 'kinds': {l: k for l, k in neutral['kinds'].items() if l <= lap},
            'period': {l: v for l, v in neutral['period'].items() if l <= lap},
            'compress_after': frozenset(l for l in neutral['compress_after'] if l <= lap)}


class PositionDecider:
    """green_value for 06's follow_default_policy (and 09 live), for one
    car: how much better, in the objective, pitting at the end of `lap` is
    than the best later lap in the planned window, racing the forecast
    field. Forecasts are shared across the race's cars via `cache`."""

    def __init__(self, replay: RaceReplay, driver: str, plan: dict, start_compound: str, pit_loss: float,
                 neutral: dict | None, pm: dict, rival: str | None, pit_loss_by_driver: dict, cache: dict):
        self.replay, self.driver, self.plan, self.start = replay, driver, plan, start_compound
        self.pit_loss, self.neutral, self.pm, self.rival = pit_loss, neutral, pm, rival
        self.pit_loss_by_driver, self.cache = pit_loss_by_driver, cache

    oracle = False   # diagnostic only: race the other cars' REAL future laps instead of a forecast

    def forecast(self, lap: int) -> RaceReplay:
        if lap not in self.cache:
            self.cache[lap] = (self.replay if self.oracle
                               else self.replay.forecast_from(lap, self.plan, self.pit_loss_by_driver))
        fc = copy.copy(self.cache[lap])
        fc.neutral = truncate_neutral(self.neutral, lap)   # this team's SC/VSC prices for its own past stops
        return fc

    def __call__(self, lap: int, done: list, pending: list, known_lap: int | None = None) -> float | None:
        """known_lap: the last lap the forecast may use (live, the decision
        for the end of the next lap is made before it's run); default `lap`."""
        if not pending:
            return None
        fc = self.forecast(lap if known_lap is None else known_lap)
        if self.driver not in fc.finished:
            return None
        offset = fc.pace_offset(self.driver)
        seed = zlib.crc32(f"{self.replay.race_id}{self.driver}{lap}".encode())
        values = {}
        for x in range(lap, max(lap, pending[0][1][1]) + 1):
            f = ds.PlanFollower({'stops': [], 'windows': []}, self.replay.n_laps)
            f.pending, f.done = list(pending), list(done)
            f.record_stop(x)
            stops = sorted(f.done + [st for st, _ in f.pending])
            comps = stint_compounds([self.start], len(stops), self.plan['compounds'])
            sim = fc.simulate(self.driver, stops, comps, self.pit_loss, self.pm, rival=self.rival,
                              n_sims=DECISION_SIMS, seed=seed, offset=offset)
            if sim is not None:
                values[x] = decision_value(sim)
        later = [v for x, v in values.items() if x > lap]
        if lap not in values or not later:
            return None
        return values[lap] - max(later)


def parse_strategy(s) -> tuple | None:
    """'18;41|MEDIUM;HARD;HARD' (06's format) -> ([18, 41], [...])."""
    if not isinstance(s, str) or '|' not in s:
        return None
    laps, comps = s.split('|')
    return [int(x) for x in laps.split(';') if x], comps.split(';')


def bootstrap_mean_ci(per_race: pd.Series, n: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    v = per_race.to_numpy(dtype=float)
    boots = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def position_backtest(events: pd.DataFrame, roll: pd.DataFrame, dp: pd.DataFrame, n_sims: int = 40) -> pd.DataFrame:
    """Real strategy vs default plan vs live policy (06's rolling backtest),
    each re-raced through the position simulator with the same random
    draws, scored in position, points and points vs the championship rival."""
    ml_bundle = dm.load_degradation_model()
    beta = dpl.sc_gap_compression()
    rivals = pd.read_csv('data/championship_rivals.csv').set_index(['race', 'driver'])['rival_driver']
    rows, models = [], {}
    for race_id, g in roll.groupby('race'):
        season = race_id[:4]
        if season not in models:
            models[season] = fit_pass_model(events, before_race=f"{season}_r00")
        dp_race = dp[dp['race'] == race_id].set_index('driver')
        setups = race_setup(race_id, dp_race.loc[dp_race.index.intersection(g['driver'])], ml_bundle, beta)
        for _, r in g.iterrows():
            if r['driver'] not in setups:
                continue
            replay, pit_loss = setups[r['driver']]
            if not replay.ok or r['driver'] not in replay.finished:
                continue
            rival = rivals.get((race_id, r['driver']))
            offset = replay.pace_offset(r['driver'])
            row = {'race': race_id, 'driver': r['driver'], 'rival': rival, 'race_had_sc': r['race_had_sc']}
            for name, strat in [('real', real_strategy(replay, r['driver'])),
                                ('default', parse_strategy(r.get('default_strategy'))),
                                ('policy', parse_strategy(r.get('policy_strategy'))),
                                ('policy_pos', parse_strategy(r.get('policy_pos_strategy'))),
                                ('policy_strat', parse_strategy(r.get('policy_strat_strategy')))]:
                if strat is None:
                    continue
                sim = replay.simulate(r['driver'], strat[0], strat[1], pit_loss, models[season], rival=rival,
                                      n_sims=n_sims, seed=zlib.crc32(f"{race_id}{r['driver']}".encode()), offset=offset)
                if sim is None:
                    continue
                row[f'{name}_position'] = sim['exp_position']
                row[f'{name}_points'] = sim['exp_points']
                row[f'{name}_rival_points'] = sim.get('rival_exp_points', np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    for name in [n for n in ['real', 'default', 'policy', 'policy_pos', 'policy_strat'] if f'{n}_points' in out]:
        out[f'{name}_objective'] = out[f'{name}_points'] + RIVAL_WEIGHT * (
            out[f'{name}_points'] - out[f'{name}_rival_points'].fillna(0))
    return out


def report_position_backtest(out: pd.DataFrame, label: str):
    names = [n for n in ['default', 'policy', 'policy_pos', 'policy_strat'] if f'{n}_points' in out]
    d = out.dropna(subset=['real_points'] + [f'{n}_points' for n in names])
    print(f"\n{label}: {d['race'].nunique()} races, {len(d)} driver-races "
          f"(same random draws for every strategy; mean per driver-race, 95% CI over races)")
    for metric, better in [('position', 'lower'), ('points', 'higher'), ('objective', 'higher')]:
        line = [f"  {metric:9s} real {d[f'real_{metric}'].mean():6.2f}"]
        for name in names:
            diff = (d[f'{name}_{metric}'] - d[f'real_{metric}']).groupby(d['race']).mean()
            m, lo, hi = bootstrap_mean_ci(diff)
            line.append(f"{name} {d[f'{name}_{metric}'].mean():6.2f} (vs real {m:+.2f}, CI {lo:+.2f} to {hi:+.2f})")
        print(' | '.join(line) + f"  [{better} is better]")


def main():
    global FOLLOW_GAP_S
    events = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() else collect_pass_events()
    print(f"{len(events)} close-following green laps (within {CLOSE_GAP_S}s) in {events['race'].nunique()} dry races; "
          f"pass rate {events['passed'].mean():.3f}\n")
    FOLLOW_GAP_S = follow_gap(events)
    print(f"A quicker car (>0.5s/lap) that fails to pass sits a median {FOLLOW_GAP_S:.2f}s behind -> FOLLOW_GAP_S\n")
    validate_pass_model(events)

    dp = pd.read_csv('data/dp_baseline_backtest.csv')
    races = sorted(r for r in dp['race'].unique() if r >= '2022')
    print(f"\nSimulator check: every finisher's real strategy re-raced, {len(races)} races")
    out = validate_simulator(events, dp, races)
    out.to_csv('data/position_model_validation.csv', index=False)

    roll = pd.read_csv('data/backtest_3way_rolling.csv')
    pos = position_backtest(events, roll, dp)
    pos.to_csv('data/position_backtest.csv', index=False)
    print(f"\nObjective = expected points + {RIVAL_WEIGHT:g} x expected points gap to the championship rival")
    report_position_backtest(pos, "Position-scored backtest, all rolling races")
    report_position_backtest(pos[~pos['race_had_sc']], "Races with no SC/VSC")


if __name__ == '__main__':
    main()
