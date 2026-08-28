"""
Championship-rival identification, feeding the trigger classifier's
rival-state features (05_trigger_classifier.py). Real strategy calls
react not just to the car physically nearby on track, but to whoever a
driver/team is actually fighting for position with in the standings --
those can be far apart on track.

For each (race, driver), computes:
  - nearest drivers'-championship rival: the driver adjacent in points
    (whichever of the driver immediately above/below is closer) as of
    the standings BEFORE this race
  - nearest constructors'-championship rival TEAM: the closest opposing
    constructor in points, same leave-out-this-race discipline

Cumulative points are summed only over races strictly *before* the one
being evaluated, within the same season (championships reset each year)
-- so a race's own result can never leak into its own "who is my rival"
feature, same discipline as the SC-probability feature.

Run with: python notebooks/07_championship_rivals.py
"""
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path('data/bronze')


def load_all_results() -> pd.DataFrame:
    files = sorted(BRONZE_DIR.glob('*_results.parquet'))
    dfs = []
    for f in files:
        race_id = f.stem.replace('_results', '')
        df = pd.read_parquet(f)
        df['race'] = race_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    results = load_all_results()
    results = results.sort_values(['Year', 'RoundNumber'])

    rows = []
    for year, season in results.groupby('Year'):
        rounds = sorted(season['RoundNumber'].unique())
        cum_driver_points = {}   # Driver -> points accumulated so far (before current round)
        cum_team_points = {}     # Team -> points accumulated so far

        for rnd in rounds:
            race_rows = season[season['RoundNumber'] == rnd]
            race_id = race_rows['race'].iloc[0]

            # Standings BEFORE this race -- snapshot prior to adding this
            # round's points, so this race's own result can't leak into
            # its own rival identification.
            driver_standings = sorted(cum_driver_points.items(), key=lambda x: -x[1])
            team_standings = sorted(cum_team_points.items(), key=lambda x: -x[1])

            for _, r in race_rows.iterrows():
                driver, team = r['Driver'], r['Team']

                # Nearest drivers'-championship rival: closer of the driver
                # immediately above or below in points-so-far. Drivers with
                # no prior-season history (rookies, mid-season debuts) get
                # no rival for their first race -- nothing to compare yet.
                names = [d for d, _ in driver_standings]
                pts = dict(driver_standings)
                rival_driver, rival_gap = None, None
                if driver in names:
                    idx = names.index(driver)
                    candidates = []
                    if idx > 0:
                        candidates.append((names[idx - 1], pts[names[idx - 1]] - pts[driver]))
                    if idx < len(names) - 1:
                        candidates.append((names[idx + 1], pts[driver] - pts[names[idx + 1]]))
                    if candidates:
                        rival_driver, rival_gap = min(candidates, key=lambda x: x[1])

                team_names = [t for t, _ in team_standings]
                team_pts = dict(team_standings)
                rival_team, rival_team_gap = None, None
                if team in team_names:
                    idx = team_names.index(team)
                    candidates = []
                    if idx > 0:
                        candidates.append((team_names[idx - 1], team_pts[team_names[idx - 1]] - team_pts[team]))
                    if idx < len(team_names) - 1:
                        candidates.append((team_names[idx + 1], team_pts[team] - team_pts[team_names[idx + 1]]))
                    if candidates:
                        rival_team, rival_team_gap = min(candidates, key=lambda x: x[1])

                rows.append({
                    'race': race_id, 'year': year, 'round': rnd, 'driver': driver, 'team': team,
                    'drivers_champ_points_before': pts.get(driver, 0.0),
                    'rival_driver': rival_driver, 'rival_driver_points_gap': rival_gap,
                    'constructors_champ_points_before': team_pts.get(team, 0.0),
                    'rival_team': rival_team, 'rival_team_points_gap': rival_team_gap,
                })

            # Now fold this round's actual points into the running totals,
            # for use as the "before" snapshot of the *next* round.
            for _, r in race_rows.iterrows():
                cum_driver_points[r['Driver']] = cum_driver_points.get(r['Driver'], 0.0) + r['Points']
                cum_team_points[r['Team']] = cum_team_points.get(r['Team'], 0.0) + r['Points']

    out = pd.DataFrame(rows)
    out.to_csv('data/championship_rivals.csv', index=False)
    print(f"Saved data/championship_rivals.csv ({len(out)} rows)")

    has_rival = out['rival_driver'].notna().mean()
    print(f"\n{has_rival*100:.1f}% of (race, driver) rows have an identified drivers'-championship rival "
          f"(rest are each driver's first race of a season -- no prior standings to compare against)")

    last_year = out['year'].max()
    last_round = out.loc[out['year'] == last_year, 'round'].max()
    sample = out[(out['year'] == last_year) & (out['round'] == last_round)].sort_values(
        'drivers_champ_points_before', ascending=False)
    print(f"\nExample -- {last_year} round {last_round} (most recent race), standings-based rivals:")
    print(sample[['driver', 'team', 'drivers_champ_points_before', 'rival_driver', 'rival_driver_points_gap']]
          .head(10).to_string(index=False))


if __name__ == '__main__':
    main()
