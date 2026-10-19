"""
Bronze-layer ingestion: pull race sessions across multiple seasons and dump
the essential lap/weather/race-control columns to Parquet, unmodified, as
the historical source of truth.

This deliberately does NOT apply the fuel-correction / degradation /
pit-loss logic from 00_explore_one_race.py -- that logic lives in
02_fixed_degradation_pitloss.py and 03_ml_degradation_model.py, which
read from this Bronze layer.

Also lands, per round, the official drivers'/constructors' championship
standings after that round and (on sprint weekends) the sprint results,
from Jolpica (the Ergast successor) via FastF1's built-in client. These
feed 07_championship_rivals.py: summing race-session points alone misses
sprint points and any official adjustments.

Also writes data/circuits.csv (race, event, location) from FastF1's
schedule: event names aren't circuits -- the same track runs under several
names (Styrian/Austrian, 70th Anniversary/British, Barcelona/Spanish in
2026) and one name can move (the 2026 Spanish GP is in Madrid) -- so 02's
circuit_of() keys everything per-circuit off the location instead.

Idempotent: already-written races are skipped, so re-running after adding
a new season only pulls what's missing. Races without a completed race
session yet (future rounds) fail gracefully and are reported, not fatal.

Run with: python notebooks/01_pull_bronze_data.py
"""
import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from pathlib import Path

fastf1.Cache.enable_cache('data/fastf1_cache')
ergast = Ergast(result_type='pandas', auto_cast=True, limit=100)

# 2019-2021 run on the previous (13-inch tyre, pre-ground-effect) rules,
# kept for history -- pit loss, SC/VSC rates, and as-of models for early
# 2022 races -- with downstream models told the era apart. FastF1 has full
# timing data (tyres, stints, track status) from 2018; 2018 itself is left
# out because its absolute compound names (ultrasoft, supersoft, ...) map to
# soft/medium/hard differently at every race.
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
BRONZE_DIR = Path('data/bronze')
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

LAP_COLS = [
    'Driver', 'DriverNumber', 'Team', 'LapNumber', 'LapTime', 'Time', 'Sector1Time',
    'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'FreshTyre',
    'Stint', 'PitInTime', 'PitOutTime', 'TrackStatus', 'Position',
    'IsPersonalBest', 'DeletedReason',
]

RESULT_COLS = ['Abbreviation', 'DriverNumber', 'DriverId', 'TeamName', 'TeamId',
               'Position', 'ClassifiedPosition', 'Points', 'Status']

# DriverId/constructorId are the same Ergast-style ids as FastF1's
# DriverId/TeamId result columns above, which is how 07 joins them.
DRIVER_STANDING_COLS = ['position', 'points', 'driverId', 'driverCode']
CONSTRUCTOR_STANDING_COLS = ['position', 'points', 'constructorId', 'constructorName']
SPRINT_COLS = ['position', 'points', 'driverId', 'driverCode', 'constructorId']


def pull_standings(year: int, rnd: int, is_sprint_weekend: bool) -> str:
    """Official standings after this round (+ sprint results if any).
    Writes nothing for a round Jolpica has no data for yet, so it's
    retried on the next run rather than cached as empty."""
    race_id = f"{year}_r{rnd:02d}"
    drivers_path = BRONZE_DIR / f"{race_id}_standings_drivers.parquet"
    teams_path = BRONZE_DIR / f"{race_id}_standings_constructors.parquet"
    sprint_path = BRONZE_DIR / f"{race_id}_sprint.parquet"  # not *_results: that glob means race results
    if drivers_path.exists() and teams_path.exists() and (sprint_path.exists() or not is_sprint_weekend):
        return 'standings cached'

    drivers = ergast.get_driver_standings(season=year, round=rnd).content
    teams = ergast.get_constructor_standings(season=year, round=rnd).content
    if not drivers or not teams:
        return 'no standings yet'
    if is_sprint_weekend:
        sprint = ergast.get_sprint_results(season=year, round=rnd).content
        if not sprint:
            return 'no sprint results yet'
        sprint[0][SPRINT_COLS].assign(Year=year, RoundNumber=rnd).to_parquet(sprint_path, index=False)
    drivers[0][DRIVER_STANDING_COLS].assign(Year=year, RoundNumber=rnd).to_parquet(drivers_path, index=False)
    teams[0][CONSTRUCTOR_STANDING_COLS].assign(Year=year, RoundNumber=rnd).to_parquet(teams_path, index=False)
    return 'standings written'


results = []
circuit_rows = []
for year in YEARS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    print(f"=== {year}: {len(schedule)} rounds ===")

    for _, event in schedule.iterrows():
        rnd, name = event['RoundNumber'], event['EventName']
        circuit_rows.append({'race': f"{year}_r{rnd:02d}", 'event': name, 'location': event['Location']})
        try:
            standings_status = pull_standings(year, rnd, 'sprint' in str(event['EventFormat']))
        except Exception as e:
            standings_status = f'standings FAILED - {e}'
        print(f"[{year} r{rnd:2d}] {name}: {standings_status}")

        out_path = BRONZE_DIR / f"{year}_r{rnd:02d}_laps.parquet"
        if out_path.exists():
            print(f"[{year} r{rnd:2d}] {name}: already cached, skipping")
            results.append((year, rnd, name, 'cached'))
            continue

        try:
            session = fastf1.get_session(year, rnd, 'R')
            session.load(telemetry=False, weather=True, messages=True)

            laps = session.laps[LAP_COLS].copy()
            laps['Year'] = year
            laps['RoundNumber'] = rnd
            laps['EventName'] = name
            laps.to_parquet(out_path, index=False)

            weather_path = BRONZE_DIR / f"{year}_r{rnd:02d}_weather.parquet"
            session.weather_data.to_parquet(weather_path, index=False)

            rc_path = BRONZE_DIR / f"{year}_r{rnd:02d}_race_control.parquet"
            session.race_control_messages.to_parquet(rc_path, index=False)

            res = session.results[RESULT_COLS].copy().rename(columns={'Abbreviation': 'Driver', 'TeamName': 'Team'})
            res['Year'] = year
            res['RoundNumber'] = rnd
            res['EventName'] = name
            results_path = BRONZE_DIR / f"{year}_r{rnd:02d}_results.parquet"
            res.to_parquet(results_path, index=False)

            print(f"[{year} r{rnd:2d}] {name}: {len(laps)} laps written")
            results.append((year, rnd, name, f'{len(laps)} laps'))
        except Exception as e:
            print(f"[{year} r{rnd:2d}] {name}: FAILED - {e}")
            results.append((year, rnd, name, f'FAILED: {e}'))

pd.DataFrame(circuit_rows).to_csv('data/circuits.csv', index=False)

print("\n--- Summary ---")
for year, rnd, name, status in results:
    print(f"  {year} r{rnd:02d}  {name:30s} {status}")

ok = sum(1 for *_, s in results if s not in ('cached',) and not s.startswith('FAILED'))
cached = sum(1 for *_, s in results if s == 'cached')
failed = sum(1 for *_, s in results if s.startswith('FAILED'))
print(f"\n{ok} newly written, {cached} already cached, {failed} failed")
