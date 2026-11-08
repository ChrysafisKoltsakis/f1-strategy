"""
Phase 17 -- figures. Reads the CSVs the other notebooks write and saves PNGs
to docs/figures/ (a figure whose input is missing is skipped, so this can run
at any stage). Nothing is recomputed from the raw data except bootstrap
intervals, which resample whole races like 06 and 15 do.

Run with: python notebooks/17_plots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path('data')
CMP = DATA / 'model_comparison'
FIG_DIR = Path('docs/figures')
N_BOOTSTRAP = 2000

# Categorical slots in fixed order (the dataviz skill's reference palette); grey is the teams' baseline.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
GREY, INK, MUTED = '#8a8985', '#0b0b0b', '#52514e'
COMPOUND_COLOR = {'SOFT': ORANGE, 'MEDIUM': YELLOW, 'HARD': BLUE}

plt.rcParams.update({
    'figure.dpi': 100, 'savefig.dpi': 160, 'savefig.bbox': 'tight', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"  wrote docs/figures/{name}")


def read(path):
    return pd.read_csv(path) if Path(path).exists() else None


def race_bootstrap(df, stat, seed=0):
    """(estimate, lo, hi): stat(df) with a 95% interval from resampling whole races."""
    by_race = {r: g for r, g in df.groupby('race')}
    races = np.array(list(by_race))
    rng = np.random.default_rng(seed)
    boot = [stat(pd.concat([by_race[r] for r in rng.choice(races, len(races))])) for _ in range(N_BOOTSTRAP)]
    return stat(df), *np.percentile(boot, [2.5, 97.5])


def forest(ax, labels, est, lo, hi, colors, zero=True):
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=colors, lw=2)
    ax.scatter(est, y, color=colors, s=36, zorder=3, edgecolor='white', linewidth=1)
    ax.set_yticks(y, labels)
    ax.grid(axis='y', visible=False)
    if zero:
        ax.axvline(0, color=MUTED, lw=0.8)


# --- Tyre model and pit loss -------------------------------------------------

def tyre_degradation():
    d = read(DATA / 'ml_degradation_curves.csv')
    if d is None:
        return
    d = d[d['compound'].isin(COMPOUND_COLOR)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rng = np.random.default_rng(0)
    for i, c in enumerate(COMPOUND_COLOR):
        s = d[d['compound'] == c]['implied_slope_s_per_lap']
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(s)), s, color=COMPOUND_COLOR[c], s=22, alpha=0.8,
                   edgecolor='white', linewidth=0.5)
        ax.hlines(s.median(), i - 0.3, i + 0.3, color=INK, lw=2)
        ax.text(i + 0.33, s.median(), f"{s.median():.3f}", va='center', fontsize=9, color=MUTED)
    ax.set_xticks(range(3), [f"{c.title()}\n(n={int((d['compound'] == c).sum())} circuits)" for c in COMPOUND_COLOR])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel('Learned wear, seconds per lap of tyre age')
    ax.set_title('Tyre wear the model learns, by circuit and compound')
    ax.grid(axis='x', visible=False)
    save(fig, 'tyre_degradation_slopes.png')


def pit_loss():
    d = read(DATA / 'pit_loss_summary.csv')
    if d is None:
        return
    d = d.reset_index(drop=True)
    d['season'] = d['race'].str[:4].astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(d.index, d['median_pit_loss'], s=14, color=BLUE, alpha=0.8, edgecolor='white', linewidth=0.4)
    for s, g in d.groupby('season'):
        ax.text(g.index.min(), ax.get_ylim()[1], str(s), fontsize=8, color=MUTED, va='top')
        if s > d['season'].min():
            ax.axvline(g.index.min() - 0.5, color='#e4e3df', lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel('Races, in order')
    ax.set_ylabel('Median pit loss (s)')
    ax.set_title('Pit-lane time loss per race')
    save(fig, 'pit_loss_by_race.png')


# --- Pit classifier -----------------------------------------------------------

def classifier_diagnostics():
    d = read(DATA / 'trigger_classifier_rolling_predictions.csv')
    if d is None:
        return
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    d['bin'] = pd.qcut(d['pred_proba'].rank(method='first'), 15, labels=False)
    b = d.groupby('bin').agg(p=('pred_proba', 'mean'), o=('label_pit', 'mean'))
    lim = [0.8 * b.min().min(), 1.2 * b.max().max()]
    ax.plot(lim, lim, color=GREY, lw=1, ls='--')
    ax.plot(b['p'], b['o'], color=BLUE, lw=2, marker='o', ms=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Predicted probability (bin mean)'); ax.set_ylabel('Observed pit rate')
    ax.set_title('Calibration (rolling-origin)')

    ax = axes[1]
    p, r, _ = precision_recall_curve(d['label_pit'], d['pred_proba'])
    ax.plot(r, p, color=BLUE, lw=2)
    ax.axhline(d['label_pit'].mean(), color=GREY, lw=1, ls='--')
    ax.text(0.5, d['label_pit'].mean() + 0.03, 'no-skill (pit rate)', ha='center', fontsize=8, color=MUTED)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f"Precision-recall (AP {average_precision_score(d['label_pit'], d['pred_proba']):.3f})")

    ax = axes[2]
    s = d.groupby('season').apply(lambda g: pd.Series({
        'ap': average_precision_score(g['label_pit'], g['pred_proba']), 'rate': g['label_pit'].mean()}),
        include_groups=False)
    ax.bar(s.index.astype(str), s['ap'], color=BLUE, width=0.55)
    ax.scatter(s.index.astype(str), s['rate'], color=INK, s=30, zorder=3, label='no-skill AP (pit rate)')
    ax.set_ylabel('Average precision'); ax.set_title('By season')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.grid(axis='x', visible=False)
    save(fig, 'classifier_diagnostics.png')


# --- Backtest -------------------------------------------------------------------

def backtest_time():
    d = read(DATA / 'backtest_3way_rolling.csv')
    if d is None:
        return
    strategies = {'Default plan (pre-race)': 'default_s', 'Live policy (default + SC/VSC)': 'policy_sc_s',
                  'Classifier, multi-stop ranking': 'ranked_multi_s', 'Classifier, threshold': 'classifier_s'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True, sharey=True)
    for ax, (title, sub) in zip(axes, [(f"All races ({d['race'].nunique()})", d),
                                       (f"No SC/VSC ({d[~d['race_had_sc']]['race'].nunique()})", d[~d['race_had_sc']])]):
        est, lo, hi, cols = [], [], [], []
        for (name, col), color in zip(strategies.items(), [GREY, BLUE, ORANGE, AQUA]):
            v = sub.dropna(subset=[col, 'actual_s'])
            e, l, h = race_bootstrap(v.assign(diff=v[col] - v['actual_s']), lambda x: x['diff'].median())
            est.append(e); lo.append(l); hi.append(h); cols.append(color)
        forest(ax, list(strategies), est, lo, hi, cols)
        ax.set_title(title)
        ax.set_xlabel('Median race time vs. the teams (s); negative = faster')
    save(fig, 'backtest_time_vs_teams.png')


def backtest_points():
    d = read(DATA / 'position_backtest.csv')
    if d is None:
        return
    names = {'Time-based policy': 'policy', 'Position-aware (windows)': 'policy_pos',
             'Strategy, guarded (no fewer stops)': 'policy_strat_guarded', 'Strategy, unrestricted': 'policy_strat'}
    names = {k: v for k, v in names.items() if f'{v}_points' in d}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True, sharey=True)
    green = d[~d['race_had_sc']]
    for ax, (title, sub) in zip(axes, [(f"All races ({d['race'].nunique()})", d),
                                       (f"Green-flag races ({green['race'].nunique()})", green)]):
        est, lo, hi = [], [], []
        for col in names.values():
            v = sub.dropna(subset=[f'{col}_points', 'real_points'])
            e, l, h = race_bootstrap(v.assign(diff=v[f'{col}_points'] - v['real_points']), lambda x: x['diff'].mean())
            est.append(e); lo.append(l); hi.append(h)
        forest(ax, list(names), est, lo, hi, [BLUE, ORANGE, AQUA, MAGENTA][:len(names)])
        ax.set_title(title)
        ax.set_xlabel('Expected points per driver-race vs. the teams')
    save(fig, 'backtest_points_vs_teams.png')


def position_validation():
    d = read(DATA / 'position_model_validation.csv')
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(d['real_position'], d['sim_position'], s=8, color=BLUE, alpha=0.25, edgecolor='none')
    ax.plot([1, 20], [1, 20], color=GREY, lw=1, ls='--')
    ax.set_xlabel('Real finishing position'); ax.set_ylabel('Re-raced finishing position (mean)')
    ax.set_title('Simulator reproduces real finishes')
    ax = axes[1]
    err = (d['sim_position'] - d['real_position']).abs()
    ax.hist(err, bins=np.arange(0, 12, 0.5), color=BLUE, edgecolor='white')
    ax.axvline(err.mean(), color=ORANGE, lw=2)
    ax.text(err.mean() + 0.15, ax.get_ylim()[1] * 0.9, f"mean {err.mean():.2f} places", color=INK, fontsize=9)
    ax.set_xlabel('|simulated - real| position'); ax.set_ylabel('Driver-races')
    ax.set_title('Error')
    save(fig, 'position_model_validation.png')


def overtaking():
    d = read(DATA / 'overtaking_events.csv')
    if d is None:
        return
    d = d[(d['gap'] > 0) & (d['gap'] < 2.0)]
    d['pace_group'] = pd.cut(d['pace_adv'], [-99, -0.25, 0.25, 99],
                             labels=['attacker slower by >0.25s', 'similar pace', 'attacker faster by >0.25s'])
    d['gap_bin'] = pd.cut(d['gap'], np.arange(0, 2.01, 0.25))
    t = d.groupby(['pace_group', 'gap_bin'], observed=True).agg(rate=('passed', 'mean'), n=('passed', 'size')).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for g, color in zip(t['pace_group'].cat.categories, [GREY, BLUE, ORANGE]):
        s = t[(t['pace_group'] == g) & (t['n'] >= 30)]
        ax.plot([iv.mid for iv in s['gap_bin']], s['rate'], marker='o', ms=5, lw=2, color=color, label=g)
    ax.set_xlabel('Gap to the car ahead at the start of the lap (s)')
    ax.set_ylabel('Share of laps ending in a pass')
    ax.set_title('Chance of passing within a lap')
    ax.legend(frameon=False, fontsize=8)
    save(fig, 'overtaking_probability.png')


def strategy_cost_validation():
    d = read(DATA / 'strategy_cost_validation_pairs.csv')
    if d is None:
        return
    d = d[~d['same_strategy'] & d['skill_gap_per_lap_s'].notna()].copy()
    d['real_adj'] = d['real_delta_s'] - d['skill_gap_per_lap_s'] * d['common_laps']
    d = d[d['real_adj'].abs() < 60]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(d['pred_delta_s'], d['real_adj'], s=14, color=BLUE, alpha=0.55, edgecolor='none')
    slope = (d['pred_delta_s'] * d['real_adj']).sum() / (d['pred_delta_s'] ** 2).sum()
    lim = max(d['pred_delta_s'].abs().max(), d['real_adj'].abs().max())
    ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1, ls='--', label='slope 1 (priced right)')
    ax.plot([-lim, lim], [-lim * slope, lim * slope], color=ORANGE, lw=2, label=f"fit through origin ({slope:.2f})")
    ax.set_xlabel('Predicted time difference between teammates (s)')
    ax.set_ylabel('Real difference, skill-adjusted (s)')
    ax.set_title(f"Strategy pricing, {len(d)} teammate pairs")
    ax.legend(frameon=False, fontsize=8)
    save(fig, 'strategy_cost_validation.png')


# --- Model comparison (16) -----------------------------------------------------

def _short(names, width=30):
    return [n if len(n) <= width else n[:width - 1] + '…' for n in names]


def comparison_tyre():
    d = read(CMP / 'tyre_summary.csv')
    if d is None:
        return
    d = d.sort_values('mae_s', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [BLUE if m.startswith('LightGBM (dep') else GREY for m in d['model']]
    forest(ax, list(d['model']), d['mae_s'], d['mae_ci_low'], d['mae_ci_high'], colors, zero=False)
    ax.set_xlabel('Leave-one-circuit-out MAE of relative pace (s); 95% CI over circuits')
    ax.set_title('Tyre model: held-out error by model family')
    save(fig, 'comparison_tyre_mae.png')


def comparison_tyre_cv(k=5):
    """Grouped k-fold against leave-one-circuit-out, on the same models and metric."""
    loco, cv = read(CMP / 'tyre_summary.csv'), read(CMP / f'tyre_cv{k}_summary.csv')
    circuits = read(CMP / f'tyre_cv{k}_circuits.csv')
    if loco is None or cv is None or circuits is None:
        return
    loco = loco.sort_values('mae_s').set_index('model')
    cv = cv.set_index('model').reindex(loco.index)
    y = np.arange(len(loco))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)

    for off, d, color, label in [(0.14, loco, GREY, 'Leave-one-circuit-out'), (-0.14, cv, BLUE, f'Grouped {k}-fold')]:
        axes[0].hlines(y + off, d['mae_ci_low'], d['mae_ci_high'], color=color, lw=2)
        axes[0].scatter(d['mae_s'], y + off, color=color, s=36, zorder=3, edgecolor='white', linewidth=1, label=label)
        axes[1].hlines(y + off, d['vs_deployed_ci_low'], d['vs_deployed_ci_high'], color=color, lw=2)
        axes[1].scatter(d['vs_deployed_s'], y + off, color=color, s=36, zorder=3, edgecolor='white', linewidth=1)
    axes[0].set_yticks(y, list(loco.index))
    axes[0].set_xlabel('MAE of relative pace (s); 95% CI over circuits')
    axes[0].set_title('Held-out error')
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.07))
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set_xlabel('MAE vs. deployed (s); negative = better')
    axes[1].set_title('Paired against deployed')
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.01))

    by_fold = circuits.groupby(['model', 'repeat', 'fold']).apply(
        lambda g: np.average(g['mae'], weights=g['n']), include_groups=False).rename('mae').reset_index()
    rng = np.random.default_rng(0)
    for model, yy in zip(loco.index, y):
        s = by_fold[by_fold['model'] == model]['mae']
        axes[2].scatter(s, yy + rng.uniform(-0.16, 0.16, len(s)), color=BLUE, s=20, alpha=0.65, edgecolor='white',
                        linewidth=0.5)
    axes[2].set_xlabel(f'MAE of each fold (s), {by_fold["repeat"].nunique()} repeats of {k} folds')
    axes[2].set_title('Fold-to-fold spread')
    for ax in axes:
        ax.grid(axis='y', visible=False)
    save(fig, f'comparison_tyre_cv{k}.png')


def comparison_classifier():
    d = read(CMP / 'classifier_summary.csv')
    if d is None:
        return
    d = d.sort_values('avg_precision')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    colors = [BLUE if m.startswith('LightGBM (dep') else GREY for m in d['model']]
    forest(axes[0], list(d['model']), d['avg_precision'], d['ap_ci_low'], d['ap_ci_high'], colors, zero=False)
    axes[0].set_xlabel('Average precision, rolling-origin 2022-26 (95% CI over races)')
    axes[0].set_title('Ranking pit laps')
    axes[1].barh(np.arange(len(d))[::-1], d['calibration_error'], color=colors, height=0.5)
    axes[1].set_yticks([])
    axes[1].set_xlabel('Calibration error (lower = probabilities mean what they say)')
    axes[1].set_title('Calibration'); axes[1].grid(axis='y', visible=False)
    save(fig, 'comparison_classifier.png')


def comparison_downstream():
    d = read(CMP / 'downstream_summary.csv')
    if d is None:
        return
    # One figure per thing swapped; panels in a figure share the model rows.
    for kind, strategies, name in [
            ('tyre', ['Default plan (pre-race)', 'Live policy (default + SC/VSC)'], 'comparison_downstream_tyre.png'),
            ('classifier', ['Classifier, threshold', 'Classifier, multi-stop ranking'],
             'comparison_downstream_classifier.png')]:
        k = d[d['varied'] == kind]
        if k.empty:
            continue
        order = list(dict.fromkeys(k['model']))
        fig, axes = plt.subplots(1, len(strategies), figsize=(11, 3.6), sharey=True)
        for ax, strat in zip(axes, strategies):
            s = k[k['strategy'] == strat].set_index('model').reindex(order)
            colors = [BLUE if m.startswith('LightGBM (dep') else GREY for m in order]
            forest(ax, order, s['vs_real_median_diff_s'], s['diff_ci_low'], s['diff_ci_high'], colors)
            ax.set_title(strat)
            ax.set_xlabel('Median seconds vs. the teams; negative = faster')
        fig.suptitle(f"{'Tyre model' if kind == 'tyre' else 'Pit classifier'} swapped, rolling backtest "
                     f"({int(k['races'].iloc[0])} races)", x=0.01, ha='left', fontsize=11, fontweight='bold', y=1.04)
        save(fig, name)


def main():
    print("Writing figures to docs/figures/")
    for f in [tyre_degradation, pit_loss, classifier_diagnostics, backtest_time, backtest_points,
              position_validation, overtaking, strategy_cost_validation,
              comparison_tyre, comparison_tyre_cv, comparison_classifier, comparison_downstream]:
        try:
            f()
        except Exception as e:   # one broken figure shouldn't stop the rest
            print(f"  {f.__name__} failed: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
