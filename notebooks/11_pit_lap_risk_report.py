"""
Phase 6 extension -- per-lap risk/reward report, replacing a single
collapsed "pit here" recommendation with a ranked table of genuinely
viable candidate laps. Prompted directly by the VER case found while
investigating whether the classifier differentiates across drivers on
identical strategies: its top pick (lap 47) beat the runner-up (lap 38)
by a probability margin of just 0.004 -- statistically indistinguishable
-- yet the single-answer decision rule reported lap 47 as *the*
recommendation with no indication the call was that close, and lap 38
would have matched VER's real second stop far better.

Reuses select_candidate_stops() and replay_strategy() from
06_backtest_evaluation.py (loaded dynamically, same pattern used
throughout this project to avoid duplicating tested logic) rather than
inventing new candidate-generation or cost logic -- this is the same
mechanism the multi-stop ranking rule already uses internally, just
surfaced as an output instead of being collapsed down to one winner.

For each candidate lap: the classifier's own confidence (how much
historical precedent supports pitting there) and the actual predicted
time cost of committing to it (via the same expected-pace/pit-loss model
the DP baseline and backtest use) are reported side by side. A
strategist can then see when two laps are a real toss-up (like VER's
47 vs. 38) rather than being handed one number with the closeness of the
call thrown away.

Run with: python notebooks/11_pit_lap_risk_report.py <race_id> <driver>
  e.g.    python notebooks/11_pit_lap_risk_report.py 2026_r09 VER
"""
import sys
import pandas as pd
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'backtest_lib', Path(__file__).parent / '06_backtest_evaluation.py')
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

BRONZE_DIR = Path('data/bronze')
DRY_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
N_CANDIDATES = 5


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    race_id, driver = sys.argv[1], sys.argv[2]

    pred = pd.read_csv('data/trigger_classifier_test_predictions.csv')
    dgrp = pred[(pred['race'] == race_id) & (pred['driver'] == driver)]
    if dgrp.empty:
        print(f"No predictions found for {driver} in {race_id} -- is this a held-out test-set race?")
        sys.exit(1)

    laps = pd.read_parquet(BRONZE_DIR / f"{race_id}_laps.parquet")
    laps = bt.fuel_correct(laps)
    n_laps = int(laps['LapNumber'].max())
    field_pace = bt.compute_field_pace(laps)
    event_name = laps['EventName'].iloc[0]
    team = laps.loc[laps['Driver'] == driver, 'Team'].iloc[0]
    start_compound = laps.loc[(laps['Driver'] == driver) & (laps['LapNumber'] == 1), 'Compound'].iloc[0]

    deg_df = pd.read_csv('data/degradation_summary.csv')
    race_deg = deg_df[deg_df['race'] == race_id].set_index('compound').to_dict('index')
    eligible_compounds = [c for c in DRY_COMPOUNDS if c in race_deg and race_deg[c]['n_laps'] >= 5]
    ml_bundle = bt.dm.load_degradation_model()
    ml_lookup = bt.dm.build_ml_pace_lookup(race_id, ml_bundle, eligible_compounds)
    degradation = {c: ml_lookup[c] for c in eligible_compounds if c in ml_lookup}

    green_pit_loss = float(pd.read_csv('data/pit_loss_summary.csv').set_index('race')['median_pit_loss'][race_id])
    team_adj = pd.read_csv('data/team_pit_loss_adjustment.csv').set_index('team')['adjustment_s']
    pit_loss = green_pit_loss + float(team_adj.get(team, 0.0))

    dp = pd.read_csv('data/dp_baseline_backtest.csv')
    dp_row = dp[(dp['race'] == race_id) & (dp['driver'] == driver)]
    dp_optimal_s = float(dp_row['dp_optimal_s'].iloc[0]) if not dp_row.empty else None

    candidates = bt.select_candidate_stops(dgrp, n_laps, N_CANDIDATES)
    if not candidates:
        print(f"No viable candidate laps found for {driver}.")
        sys.exit(1)

    rows = []
    for lap in candidates:
        prob = dgrp.loc[dgrp['lap'] == lap, 'pred_proba_balanced'].iloc[0]
        cost = bt.replay_strategy([lap], start_compound, n_laps, field_pace, degradation, pit_loss)
        rows.append({'lap': lap, 'probability': prob, 'predicted_total_s': cost})

    report = pd.DataFrame(rows).sort_values('probability', ascending=False).reset_index(drop=True)
    best_prob = report['probability'].max()
    best_cost = report['predicted_total_s'].min()
    report['prob_gap_vs_best'] = best_prob - report['probability']
    report['cost_gap_vs_cheapest_s'] = report['predicted_total_s'] - best_cost
    if dp_optimal_s is not None:
        report['delta_vs_dp_optimal_s'] = report['predicted_total_s'] - dp_optimal_s

    print(f"{race_id} ({event_name}) -- {driver} ({team}), started on {start_compound}, {n_laps} laps\n")
    print(f"{'lap':>4s}  {'P(pit)':>7s}  {'prob gap':>9s}  {'pred. total (s)':>16s}  {'cost gap (s)':>13s}"
          + ("  delta vs DP-opt (s)" if dp_optimal_s is not None else ""))
    for _, r in report.iterrows():
        line = (f"{int(r['lap']):>4d}  {r['probability']:>7.3f}  {r['prob_gap_vs_best']:>9.3f}  "
                f"{r['predicted_total_s']:>16.1f}  {r['cost_gap_vs_cheapest_s']:>13.1f}")
        if dp_optimal_s is not None:
            line += f"  {r['delta_vs_dp_optimal_s']:>+18.1f}"
        print(line)

    top2 = report.head(2)
    if len(top2) == 2:
        prob_margin = top2.iloc[0]['probability'] - top2.iloc[1]['probability']
        cost_diff = top2.iloc[1]['predicted_total_s'] - top2.iloc[0]['predicted_total_s']
        print(f"\nTop two candidates (lap {int(top2.iloc[0]['lap'])} vs lap {int(top2.iloc[1]['lap'])}): "
              f"probability margin {prob_margin:.3f}", end="")
        if prob_margin < 0.02:
            print(f" -- essentially a toss-up by model confidence; lap {int(top2.iloc[1]['lap'])} "
                  f"{'costs ' + f'{abs(cost_diff):.1f}s more' if cost_diff > 0 else 'is actually cheaper by ' + f'{abs(cost_diff):.1f}s'}, "
                  f"worth weighing that over blindly taking the top-ranked lap.")
        else:
            print(".")


if __name__ == '__main__':
    main()
