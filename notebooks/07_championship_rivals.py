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

"Standings before this race" = the official Jolpica standings after the
previous round (landed by 01_pull_bronze_data.py), plus this weekend's
sprint points if it was a sprint weekend -- the sprint runs before the
race, so its result is known at race start and isn't leakage. Earlier
versions summed race-session points from the Bronze results instead,
which silently missed every sprint point and any official adjustment.
Ties are broken by the previous round's official position (countback),
not arbitrarily. Standings reset each season, so round 1 has no prior
standings unless it's a sprint weekend.

Also cross-checks that previous standings + sprint + this race's points
reproduce the official standings after this round -- a mismatch means a
post-race penalty/adjustment, or FastF1 and Jolpica round numbers not
lining up, which would silently pair races with the wrong standings.

Run with: python notebooks/07_championship_rivals.py
"""
import pandas as pd
from pathlib import Path

BRONZE_DIR = Path('data/bronze')


def load_bronze(suffix: str) -> pd.DataFrame:
    dfs = []
    for f in sorted(BRONZE_DIR.glob(f'*_{suffix}.parquet')):
        df = pd.read_parquet(f)
        df['race'] = f.stem.replace(f'_{suffix}', '')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def standings_before(official: pd.DataFrame, sprints: pd.DataFrame, year: int, rnd: int,
                     id_col: str) -> pd.DataFrame:
    """[id_col, points] sorted best-first: official standings after round
    rnd-1 plus round rnd's sprint points (if any). Empty if neither exists."""
    prev = official[(official['Year'] == year) & (official['RoundNumber'] == rnd - 1)]
    prev = prev.set_index(id_col)[['points', 'position']]
    if not sprints.empty:
        sprint = sprints[(sprints['Year'] == year) & (sprints['RoundNumber'] == rnd)]
        sprint_pts = sprint.groupby(id_col)['points'].sum()
        prev = prev.reindex(prev.index.union(sprint_pts.index))
        prev['points'] = prev['points'].fillna(0.0) + sprint_pts.reindex(prev.index).fillna(0.0)
    return (prev.sort_values(['points', 'position'], ascending=[False, True], na_position='last')
                .reset_index()[[id_col, 'points']])


def nearest_rival(standings: pd.DataFrame, id_col: str, me):
    """Closer of the entries immediately above/below `me` in points."""
    names = standings[id_col].tolist()
    if me not in names:
        return None, None
    pts = dict(zip(standings[id_col], standings['points']))
    idx = names.index(me)
    candidates = []
    if idx > 0:
        candidates.append((names[idx - 1], pts[names[idx - 1]] - pts[me]))
    if idx < len(names) - 1:
        candidates.append((names[idx + 1], pts[me] - pts[names[idx + 1]]))
    return min(candidates, key=lambda x: x[1]) if candidates else (None, None)


def check_standings_consistency(results, driver_official, sprints) -> pd.DataFrame:
    """Rows where official(r) != official(r-1) + sprint(r) + race(r) points."""
    rows = []
    for (year, rnd), race_rows in results.groupby(['Year', 'RoundNumber']):
        after = driver_official[(driver_official['Year'] == year) & (driver_official['RoundNumber'] == rnd)]
        if after.empty:
            continue
        before = standings_before(driver_official, sprints, year, rnd, 'driverId').set_index('driverId')['points']
        race_pts = race_rows.groupby('DriverId')['Points'].sum()
        ids = before.index.union(race_pts.index).union(after['driverId'])
        expected = before.reindex(ids).fillna(0.0) + race_pts.reindex(ids).fillna(0.0)
        actual = after.set_index('driverId')['points'].reindex(ids).fillna(0.0)
        for d in ids[(expected - actual).abs() > 1e-6]:
            rows.append({'race': race_rows['race'].iloc[0], 'driverId': d,
                         'expected': expected[d], 'official': actual[d]})
    return pd.DataFrame(rows)


def main():
    results = load_bronze('results')
    if not {'DriverId', 'TeamId'} <= set(results.columns):
        raise SystemExit("Bronze results predate the DriverId/TeamId columns -- delete "
                         "data/bronze/*_results.parquet and *_laps.parquet and re-run 01_pull_bronze_data.py")
    driver_official = load_bronze('standings_drivers')
    team_official = load_bronze('standings_constructors')
    sprints = load_bronze('sprint')
    if driver_official.empty or team_official.empty:
        raise SystemExit("No standings in data/bronze -- re-run 01_pull_bronze_data.py")
    results = results.sort_values(['Year', 'RoundNumber'])

    rows = []
    missing_prev = []
    for year, season in results.groupby('Year'):
        # id -> the name the rest of the pipeline uses (FastF1 abbreviation /
        # team name), latest seen in the season; this race's own team names
        # take precedence below, since teams can be renamed between seasons.
        # Falls back to Jolpica's driver code for a rival who has no Bronze
        # race this season (e.g. sat out with injury).
        season_driver_name = season.drop_duplicates('DriverId', keep='last').set_index('DriverId')['Driver'] \
            .combine_first(driver_official[driver_official['Year'] == year]
                           .drop_duplicates('driverId', keep='last').set_index('driverId')['driverCode'])
        season_team_name = season.drop_duplicates('TeamId', keep='last').set_index('TeamId')['Team']

        for rnd, race_rows in season.groupby('RoundNumber'):
            race_id = race_rows['race'].iloc[0]
            if rnd > 1 and not ((driver_official['Year'] == year) & (driver_official['RoundNumber'] == rnd - 1)).any():
                missing_prev.append(race_id)
            drv = standings_before(driver_official, sprints, year, rnd, 'driverId')
            team = standings_before(team_official, sprints, year, rnd, 'constructorId')
            drv_pts = dict(zip(drv['driverId'], drv['points']))
            team_pts = dict(zip(team['constructorId'], team['points']))
            team_name = race_rows.drop_duplicates('TeamId').set_index('TeamId')['Team'].combine_first(season_team_name)

            for _, r in race_rows.iterrows():
                # Drivers with no standings entry yet (season debut on a
                # non-sprint weekend) get no rival -- nothing to compare yet.
                rival_driver, rival_gap = nearest_rival(drv, 'driverId', r['DriverId'])
                rival_team, rival_team_gap = nearest_rival(team, 'constructorId', r['TeamId'])
                rows.append({
                    'race': race_id, 'year': year, 'round': rnd, 'driver': r['Driver'], 'team': r['Team'],
                    'drivers_champ_points_before': drv_pts.get(r['DriverId'], 0.0),
                    'rival_driver': season_driver_name.get(rival_driver, rival_driver),
                    'rival_driver_points_gap': rival_gap,
                    'constructors_champ_points_before': team_pts.get(r['TeamId'], 0.0),
                    'rival_team': team_name.get(rival_team, rival_team),
                    'rival_team_points_gap': rival_team_gap,
                })

    out = pd.DataFrame(rows)
    out.to_csv('data/championship_rivals.csv', index=False)
    print(f"Saved data/championship_rivals.csv ({len(out)} rows)")

    if missing_prev:
        print(f"\nWARNING: no official standings for the previous round of {len(missing_prev)} race(s) -- "
              f"those races have no rivals: {', '.join(missing_prev)}. Re-run 01_pull_bronze_data.py.")

    mismatches = check_standings_consistency(results, driver_official, sprints)
    if mismatches.empty:
        print("\nStandings cross-check: previous standings + sprint + race points match the official "
              "standings after every round")
    else:
        print(f"\nStandings cross-check: {len(mismatches)} (race, driver) mismatches across "
              f"{mismatches['race'].nunique()} race(s) -- post-race penalties/adjustments, or a "
              f"FastF1/Jolpica round misalignment if a whole race is off:")
        print(mismatches.to_string(index=False))

    has_rival = out['rival_driver'].notna().mean()
    print(f"\n{has_rival*100:.1f}% of (race, driver) rows have an identified drivers'-championship rival "
          f"(rest have no standings yet -- season debut on a non-sprint weekend)")

    last_year = out['year'].max()
    last_round = out.loc[out['year'] == last_year, 'round'].max()
    sample = out[(out['year'] == last_year) & (out['round'] == last_round)].sort_values(
        'drivers_champ_points_before', ascending=False)
    print(f"\nExample -- {last_year} round {last_round} (most recent race), standings-based rivals:")
    print(sample[['driver', 'team', 'drivers_champ_points_before', 'rival_driver', 'rival_driver_points_gap']]
          .head(10).to_string(index=False))


if __name__ == '__main__':
    main()
