"""
Does the pace model price strategies correctly? A teammate natural experiment.

Every strategy comparison in this project (actual vs. classifier vs. DP
hindsight-optimal in 04/06, pit-now-vs-wait in 09/10) is scored by the same
expected-pace model, and the DP optimum is chosen by maximising that model
-- so if the model mis-prices strategies, the "optimal" advantage is
overstated (the optimizer's curse, Smith & Winkler 2006; see
docs/research_notes.md, risk 4a). Comparing predicted race times with real
ones directly is swamped by car pace: the model predicts field-median pace,
not a given car's.

Teammates share a car, a race, and its conditions, so the difference in
their real race times is mostly strategy plus driver skill. For each
teammate pair, compare over the laps both completed:
  predicted difference = model cost of A's real strategy - B's
  real difference      = A's real lap times summed - B's
A regression slope of real on predicted near 1 means the model's strategy
differences are right-sized; well below 1 means it overstates them. Pairs
whose strategies the model prices within NOISE_FLOOR_S of each other show
the noise floor from driver skill and racing incidents alone.

Driver skill can still bias the slope (e.g. if the lead driver tends to get
the better strategy and is also faster), so each pair's skill gap is also
estimated from "matched" laps -- both cars on the same compound with tyre
ages within MATCH_AGE_LAPS, no pit, SC/VSC or opening laps -- where strategy
can't explain any difference, and subtracted over the common laps. The
skill-adjusted slope is the headline.

Stops made at the end of an SC/VSC lap are charged the SC/VSC pit loss by
the replay (04's replay_actual_laps), as in the backtest. Pairs with such a
stop are also reported separately: their slope stays low (~0.4-0.6) whether
SC/VSC stops are priced at the green or the SC/VSC pit loss -- a safety car
bunches the field, which a per-car lap-time model can't represent -- while
pairs without one come out right-sized (~1.0).

Run with: python notebooks/13_strategy_cost_validation.py
"""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

_spec = importlib.util.spec_from_file_location('dp_baseline_lib', Path(__file__).parent / '04_dp_baseline.py')
dpb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dpb)

BRONZE_DIR = Path('data/bronze')
MIN_COMMON_SHARE = 0.95   # common laps must cover this share of each driver's laps
SIGN_TEST_MIN_S = 5.0     # sign agreement only where the model predicts a real difference
NOISE_FLOOR_S = 2.0       # "same strategy" for the noise floor: priced within this of each other
MATCH_AGE_LAPS = 2        # matched laps: same compound, tyre ages within this
MIN_MATCHED_LAPS = 10     # fewer matched laps -> no skill adjustment for that pair
N_BOOTSTRAP = 2000


def strategy_signature(driver_laps: pd.DataFrame) -> tuple:
    """((first lap of each tyre set, compound), ...) -- same tyre-set-change
    rule as 04's replay."""
    d = driver_laps.sort_values('LapNumber')
    prev = d.shift()
    new_set = ((d['Compound'] != prev['Compound'])
               | (d['TyreLife'] - prev['TyreLife'] != d['LapNumber'] - prev['LapNumber']))
    return tuple((int(l), c) for l, c in zip(d.loc[new_set, 'LapNumber'], d.loc[new_set, 'Compound']))


def stopped_under_sc(driver_laps: pd.DataFrame, signature: tuple) -> bool:
    in_laps = {lap - 1 for lap, _ in signature[1:]}
    status = driver_laps.set_index('LapNumber')['TrackStatus'].astype(str)
    return any(any(c in status.get(l, '') for c in '467') for l in in_laps)


def build_pairs() -> pd.DataFrame:
    saved = pd.read_csv('data/dp_baseline_backtest.csv')
    deg_df = pd.read_csv('data/degradation_summary.csv')
    bundle = dpb.dm.load_degradation_model()
    rows = []
    for race, race_rows in saved.groupby('race'):
        laps = dpb.fuel_correct(pd.read_parquet(BRONZE_DIR / f"{race}_laps.parquet"))
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        field_pace = dpb.compute_field_pace(laps)
        beta = dpb.dpl.sc_gap_compression()
        race_deg = deg_df[deg_df['race'] == race].set_index('compound').to_dict('index')
        eligible = [c for c in dpb.DRY_COMPOUNDS if c in race_deg and race_deg[c]['n_laps'] >= 5]
        lookup = dpb.dm.build_ml_pace_lookup(race, bundle, eligible)
        degradation = {c: lookup[c] for c in eligible if c in lookup}

        for team, team_rows in race_rows.groupby('team'):
            if len(team_rows) != 2:
                continue
            cars = []
            for _, r in team_rows.iterrows():
                dl = laps[laps['Driver'] == r['driver']]
                neutral = dpb.dpl.neutralisation(laps, r['team_sc_pit_loss_s'], r['team_vsc_pit_loss_s'], beta)
                rep = dpb.replay_actual_laps(dl, field_pace, degradation, r['team_pit_loss_s'], neutral)
                if rep is None:
                    break
                real = dl.set_index('LapNumber')['LapTimeSeconds'].dropna()
                sig = strategy_signature(dl)
                cars.append({'driver': r['driver'], 'pred': rep['laps'], 'real': real, 'sig': sig,
                             'sc_stop': stopped_under_sc(dl, sig),
                             'tyres': dl.set_index('LapNumber')[['Compound', 'TyreLife', 'TrackStatus']]})
            if len(cars) != 2:
                continue
            a, b = cars
            common = (set(a['pred'].index) & set(b['pred'].index) & set(a['real'].index) & set(b['real'].index))
            # Every in-lap and out-lap must be common, or one car's pit loss
            # would be counted and the other's not.
            pit_laps = {l + k for car in cars for l, _ in car['sig'][1:] for k in (-1, 0)}
            if (not pit_laps <= common or len(common) < MIN_COMMON_SHARE * max(len(a['pred']), len(b['pred']))):
                continue
            common = sorted(common)
            # Matched laps: strategy can't explain a difference there, so the
            # median gap is the pair's skill (and car-setup) difference per lap.
            ta, tb = a['tyres'].reindex(common), b['tyres'].reindex(common)
            status = ta['TrackStatus'].astype(str) + tb['TrackStatus'].astype(str)
            matched = [l for l in common
                       if l > 2 and l not in pit_laps
                       and ta.at[l, 'Compound'] == tb.at[l, 'Compound']
                       and abs(ta.at[l, 'TyreLife'] - tb.at[l, 'TyreLife']) <= MATCH_AGE_LAPS
                       and not any(c in status[l] for c in '4567')]
            gap = ((a['real'][matched] - b['real'][matched]) - (a['pred'][matched] - b['pred'][matched]))
            skill_per_lap = float(gap.median()) if len(matched) >= MIN_MATCHED_LAPS else np.nan
            rows.append({
                'race': race, 'team': team, 'driver_a': a['driver'], 'driver_b': b['driver'],
                'common_laps': len(common),
                'pred_delta_s': a['pred'][common].sum() - b['pred'][common].sum(),
                'real_delta_s': a['real'][common].sum() - b['real'][common].sum(),
                'matched_laps': len(matched), 'skill_gap_per_lap_s': skill_per_lap,
                'same_strategy': a['sig'] == b['sig'],
                'sc_stop': a['sc_stop'] or b['sc_stop'],
            })
    return pd.DataFrame(rows)


def slope(d: pd.DataFrame, real_col: str) -> float:
    return float(np.polyfit(d['pred_delta_s'], d[real_col], 1)[0])


def summarise(d: pd.DataFrame, label: str, real_col: str = 'real_delta_s', seed: int = 0):
    rng = np.random.default_rng(seed)
    races = d['race'].unique()
    by_race = {r: g for r, g in d.groupby('race')}
    boot = [slope(pd.concat([by_race[r] for r in rng.choice(races, len(races))]), real_col) for _ in range(N_BOOTSTRAP)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    big = d[d['pred_delta_s'].abs() >= SIGN_TEST_MIN_S]
    agree = (np.sign(big['pred_delta_s']) == np.sign(big[real_col])).mean() if len(big) else np.nan
    resid = d[real_col] - d['pred_delta_s']
    print(f"\n{label}: {len(d)} pairs over {len(races)} races")
    print(f"  slope of real on predicted difference: {slope(d, real_col):.2f}  (95% CI {lo:.2f} to {hi:.2f}; 1 = right-sized)")
    print(f"  correlation: Pearson {d['pred_delta_s'].corr(d[real_col]):.2f}, "
          f"Spearman {d['pred_delta_s'].corr(d[real_col], method='spearman'):.2f}")
    print(f"  median |real - predicted|: {resid.abs().median():.1f}s")
    print(f"  sign agreement where the model predicts >= {SIGN_TEST_MIN_S:.0f}s: {agree:.0%} of {len(big)} pairs")


def main():
    pairs = build_pairs()
    pairs.to_csv('data/strategy_cost_validation_pairs.csv', index=False)
    pairs['real_delta_skill_adj_s'] = pairs['real_delta_s'] - pairs['skill_gap_per_lap_s'] * pairs['common_laps']
    near = pairs[pairs['pred_delta_s'].abs() < NOISE_FLOOR_S]
    diff = pairs[pairs['pred_delta_s'].abs() >= NOISE_FLOOR_S]
    adj = diff.dropna(subset=['real_delta_skill_adj_s'])
    print(f"{len(pairs)} teammate pairs with comparable laps ({pairs['same_strategy'].sum()} on exactly the same "
          f"strategy); {len(near)} priced within {NOISE_FLOOR_S:.0f}s of each other, {len(diff)} further apart")
    print(f"\nNoise floor (strategies priced within {NOISE_FLOOR_S:.0f}s): real difference SD "
          f"{near['real_delta_s'].std():.1f}s raw, {near['real_delta_skill_adj_s'].std():.1f}s skill-adjusted")
    summarise(diff, "Different strategies, raw real difference")
    summarise(adj, "Different strategies, skill-adjusted (headline)", 'real_delta_skill_adj_s')
    summarise(adj[~adj['sc_stop']], "Skill-adjusted, no stop under SC/VSC", 'real_delta_skill_adj_s')
    print(f"\nMedian skill gap between teammates: {pairs['skill_gap_per_lap_s'].abs().median():.2f}s/lap "
          f"(from a median {pairs['matched_laps'].median():.0f} matched laps per pair)")
    print("\nSaved data/strategy_cost_validation_pairs.csv")


if __name__ == '__main__':
    main()
