"""
Default pre-race strategy: the plan a car starts the race with, which the
live system then only changes when the evidence is clearer than the
model's own error.

Why it's needed: race data are censored -- teams change tyres before they
fall away -- so the tyre model can't show that an untried strategy (e.g. a
one-stop in Spain 2024, where every team two-stopped) would fail; two
attempts to force that from the data were rejected by the teammate
validation (see 03). That knowledge lives in what teams have actually done
at a circuit and in Pirelli's pre-race guidance, which this encodes.

Sources, in order of precedence:
  1. Pirelli's published pre-race prediction for the race, from
     reference/pirelli_strategies.csv (collected by hand from Pirelli's
     previews / press coverage -- there's no consolidated database).
  2. History: what finishers did at this circuit (02's circuit_of: by
     track, not event name) in earlier races (as of
     the race, so nothing from it or later leaks in), weighted towards
     recent seasons -- the modal stop count, the modal compound sequence
     among those drivers, and the median stop laps as a share of race
     distance, with interquartile windows.
  3. None (a circuit's first visit with no Pirelli entry) -- callers fall
     back to the model's own pre-race optimum.

Run with: python notebooks/14_default_strategy.py [race_id ...]   (prints plans)
"""
import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'degradation_pitloss_lib', Path(__file__).parent / '02_fixed_degradation_pitloss.py')
dpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpl)

BRONZE_DIR = Path('data/bronze')
PIRELLI_FILE = Path(__file__).parent.parent / 'reference' / 'pirelli_strategies.csv'
DRY_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD'}
RECENCY_HALF_LIFE_YEARS = 1.0   # a visit one season older counts half as much
MIN_GREEN_ROWS = 10             # enough green-flag-only strategies -> use only those for timing
_REAL = None


def _strategy(dl: pd.DataFrame) -> dict | None:
    """One driver's real strategy: stop in-laps and compound sequence, with
    the same tyre-set-change rule as 04's replay (red-flag changes are free,
    not stops)."""
    d = dl.sort_values('LapNumber')
    if d['Compound'].isna().any() or d['TyreLife'].isna().any() or not set(d['Compound']) <= DRY_COMPOUNDS:
        return None
    prev = d.shift()
    new_set = ((d['Compound'] != prev['Compound'])
               | (d['TyreLife'] - prev['TyreLife'] != d['LapNumber'] - prev['LapNumber']))
    red = prev['TrackStatus'].astype(str).str.contains('5')
    stops = d[new_set & ~red].iloc[1:]
    in_laps = [int(l) - 1 for l in stops['LapNumber']]
    in_status = d.set_index('LapNumber')['TrackStatus'].astype(str)
    return {'stops': in_laps, 'compounds': [d['Compound'].iloc[0]] + list(stops['Compound']),
            'neutral_stops': sum(dpl._is_sc_or_vsc(in_status.get(l, '')) for l in in_laps)}


def real_strategies() -> pd.DataFrame:
    """Every finisher's real dry strategy in every race (cached)."""
    global _REAL
    if _REAL is not None:
        return _REAL
    rows = []
    for f in sorted(BRONZE_DIR.glob('*_laps.parquet')):
        race = f.stem.replace('_laps', '')
        laps = pd.read_parquet(f, columns=['Driver', 'LapNumber', 'Compound', 'TyreLife', 'TrackStatus', 'EventName'])
        n_laps = int(laps['LapNumber'].max())
        for drv, dl in laps.groupby('Driver'):
            if dl['LapNumber'].max() < n_laps - 2:
                continue   # retired
            s = _strategy(dl)
            if s is None:
                continue
            rows.append({'race': race, 'event': laps['EventName'].iloc[0], 'circuit': dpl.circuit_of(race),
                         'era': dpl.era_of(race), 'year': int(race[:4]), 'driver': drv,
                         'n_laps': n_laps, 'n_stops': len(s['stops']), **s})
    _REAL = pd.DataFrame(rows)
    return _REAL


def _wquantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values, dtype=float)[order], np.asarray(weights, dtype=float)[order]
    c = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(c, q), len(v) - 1)])


def pirelli_plan(race_id: str, n_laps: int) -> dict | None:
    if not PIRELLI_FILE.exists():
        return None
    ref = pd.read_csv(PIRELLI_FILE, comment='#')
    row = ref[ref['race'] == race_id]
    if row.empty:
        return None
    r = row.iloc[0]
    compounds = str(r['compounds']).split('-')
    windows = []
    if pd.isna(r['stop_windows']):
        # Sequence given without laps: spread the stops evenly.
        n_stops = len(compounds) - 1
        for i in range(n_stops):
            mid = round(n_laps * (i + 1) / (n_stops + 1))
            windows.append((mid - 3, mid + 3))
    for w in ([] if pd.isna(r['stop_windows']) else str(r['stop_windows']).split(';')):
        lo, _, hi = w.strip().partition('-')
        windows.append((int(lo), int(hi or lo)))
    return {'stops': [round((lo + hi) / 2) for lo, hi in windows], 'windows': windows,
            'compounds': compounds, 'source': f"Pirelli prediction ({r['source']})"}


def history_plan(race_id: str, n_laps: int) -> dict | None:
    real = real_strategies()
    prior = real[(real['race'] < race_id) & (real['circuit'] == dpl.circuit_of(race_id)) & (real['n_stops'] >= 1)]
    era = prior[prior['era'] == dpl.era_of(race_id)]
    if len(era):
        prior = era
    if prior.empty:
        return None
    w = 0.5 ** ((int(race_id[:4]) - prior['year']) / RECENCY_HALF_LIFE_YEARS)
    n_stops = int(w.groupby(prior['n_stops']).sum().idxmax())
    same = prior[prior['n_stops'] == n_stops]
    ws = w[same.index]
    green = same[same['neutral_stops'] == 0]
    timing, wt = (green, ws[green.index]) if len(green) >= MIN_GREEN_ROWS else (same, ws)
    stops, windows = [], []
    for i in range(n_stops):
        frac = np.array([s[i] / nl for s, nl in zip(timing['stops'], timing['n_laps'])])
        stops.append(int(round(_wquantile(frac, wt, 0.5) * n_laps)))
        windows.append((int(round(_wquantile(frac, wt, 0.25) * n_laps)), int(round(_wquantile(frac, wt, 0.75) * n_laps))))
    seq = same['compounds'].apply(tuple)
    compounds = list(ws.groupby(seq).sum().idxmax())
    return {'stops': stops, 'windows': windows, 'compounds': compounds,
            'source': f"history: {same['driver'].count()} finishers' {n_stops}-stop strategies over "
                      f"{same['race'].nunique()} earlier race(s) here"}


def default_plan(race_id: str, n_laps: int) -> dict | None:
    """Pirelli if we have it, else history, else None."""
    return pirelli_plan(race_id, n_laps) or history_plan(race_id, n_laps)


CLOSE_CALL_S = 1.0   # = 10's CLOSE_CALL_S: margins below the pace model's own error don't count


class PlanFollower:
    """The live policy: follow the default plan, leaving it only when the
    cost model (10's pit-now-vs-wait comparison) is clearly sure -- its
    margin beats CLOSE_CALL_S. Shared by 06's offline replay and 09's live
    consumer so the two can't drift apart.

      - SC/VSC out: pit if pitting now beats staying out by the margin.
      - green, inside the next planned stop's window: pit before the
        planned lap only if pitting now clearly wins; at or after it, stay
        out only if waiting clearly wins, never past the window's end.
      - every close call follows the plan.
    The cost model leans towards late stops and fewer of them (censored
    tyre data, see 03), so green-flag changes are kept inside the plan's
    window -- otherwise the policy drifts back to the model's own optimum."""

    def __init__(self, plan: dict, n_laps: int):
        self.n_laps = n_laps
        self.pending = [(int(st), (int(lo), int(hi))) for st, (lo, hi) in zip(plan['stops'], plan['windows'])
                        if 2 < st < n_laps - 1]
        self.done = []

    def record_stop(self, in_lap: int):
        """A stop at the end of `in_lap` (the policy's, or live, the real
        car's). It replaces the next planned stop if it comes past the
        halfway point of that stint; earlier, it's an extra stop and the
        plan carries on."""
        if self.pending:
            prev = self.done[-1] if self.done else 0
            if in_lap >= (prev + self.pending[0][0]) / 2:
                self._respace(in_lap, self.pending.pop(0)[0])
        self.done.append(int(in_lap))

    def _respace(self, in_lap: int, replaced: int):
        """A planned stop happened early or late: re-space the remaining
        stops so the remaining stints keep the plan's proportions over the
        laps actually left (both remaining sets used as fresh as possible,
        as in the plan), windows moving with them; stops pushed out of the
        race are dropped."""
        if not self.pending:
            return
        edges = [replaced] + [st for st, _ in self.pending] + [self.n_laps]
        stints = [b - a for a, b in zip(edges, edges[1:])]
        scale = (self.n_laps - in_lap) / sum(stints)
        respaced, at = [], in_lap
        for (st, (lo, hi)), stint in zip(self.pending, stints):
            at += stint * scale
            new = int(round(at))
            if 2 < new < self.n_laps - 1:
                respaced.append((new, (lo + new - st, hi + new - st)))
        self.pending = respaced

    def wants_evaluation(self, lap: int) -> bool:
        """Is the cost model's opinion needed for a green-flag call at the
        end of `lap`? Inside the next stop's window before its end, and
        after it (see green_call)."""
        if not self.pending:
            return False
        lo, hi = self.pending[0][1]
        return lo <= lap < hi or lap > hi

    def green_call(self, lap: int, diff_s: float | None, margin: float = CLOSE_CALL_S,
                   unit: str = 's') -> tuple | None:
        """(pit, reason) for the end of `lap` under green, or None if the
        plan has nothing to say yet. diff_s = how much better pitting now is
        than waiting (seconds from the cost model, or -- with margin/unit
        set -- 15's expected-points objective), None if not evaluated."""
        if not self.pending:
            return None
        planned, (lo, hi) = self.pending[0]
        if lap < lo:
            return None
        if lap < planned:
            if diff_s is not None and diff_s >= margin:
                return True, f"pitting now clearly better ({diff_s:.1f}{unit}) -- earlier than the planned lap {planned}"
            return False, f"plan: stop at the end of lap {planned}"
        if lap < hi and diff_s is not None and -diff_s >= margin:
            return False, f"waiting clearly better ({-diff_s:.1f}{unit}) -- window open to lap {hi}"
        if lap == planned:
            return True, "planned stop"
        if lap <= hi:
            return True, "end of the planned window"
        # Live only (the policy itself always stops by the window's end):
        # the real car stayed out past it, so it's on a different plan --
        # the cost model makes the call from here.
        if diff_s is not None and diff_s >= margin:
            return True, f"past the planned window (laps {lo}-{hi}); pitting now clearly better ({diff_s:.1f}{unit})"
        return False, f"past the planned window (laps {lo}-{hi}); no clear case to stop yet"

    @staticmethod
    def neutral_call(diff_s: float | None) -> bool:
        """Pit under the SC/VSC now out?"""
        return diff_s is not None and diff_s >= CLOSE_CALL_S


def main():
    races = sys.argv[1:] or ['2023_r08', '2024_r10']
    for race in races:
        laps = pd.read_parquet(BRONZE_DIR / f"{race}_laps.parquet", columns=['LapNumber', 'EventName'])
        n_laps, event = int(laps['LapNumber'].max()), laps['EventName'].iloc[0]
        plan = default_plan(race, n_laps)
        real = real_strategies()
        actual = real[real['race'] == race]
        print(f"\n{race} {event} ({n_laps} laps)")
        print(f"  default: {plan}")
        print(f"  real finishers: stop counts {actual['n_stops'].value_counts().sort_index().to_dict()}, "
              f"median first stop lap {actual['stops'].apply(lambda s: s[0] if s else None).median()}")


if __name__ == '__main__':
    main()
