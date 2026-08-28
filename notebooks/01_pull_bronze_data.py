"""
Bronze-layer ingestion: pull race sessions across multiple seasons and dump
the essential lap/weather/race-control columns to Parquet, unmodified, as
the historical source of truth.

This deliberately does NOT apply the fuel-correction / degradation /
pit-loss logic from 00_explore_one_race.py -- that logic lives in
02_fixed_degradation_pitloss.py and 03_ml_degradation_model.py, which
read from this Bronze layer.

Idempotent: already-written races are skipped, so re-running after adding
a new season only pulls what's missing. Races without a completed race
session yet (future rounds) fail gracefully and are reported, not fatal.

Run with: python notebooks/01_pull_bronze_data.py
"""
import fastf1
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache('data/fastf1_cache')

YEARS = [2022, 2023, 2024, 2025, 2026]
BRONZE_DIR = Path('data/bronze')
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

LAP_COLS = [
    'Driver', 'DriverNumber', 'Team', 'LapNumber', 'LapTime', 'Time', 'Sector1Time',
    'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'FreshTyre',
    'Stint', 'PitInTime', 'PitOutTime', 'TrackStatus', 'Position',
    'IsPersonalBest', 'DeletedReason',
]

RESULT_COLS = ['Abbreviation', 'DriverNumber', 'TeamName', 'Position', 'ClassifiedPosition', 'Points', 'Status']

results = []
for year in YEARS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    print(f"=== {year}: {len(schedule)} rounds ===")

    for _, event in schedule.iterrows():
        rnd, name = event['RoundNumber'], event['EventName']
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

print("\n--- Summary ---")
for year, rnd, name, status in results:
    print(f"  {year} r{rnd:02d}  {name:30s} {status}")

ok = sum(1 for *_, s in results if s not in ('cached',) and not s.startswith('FAILED'))
cached = sum(1 for *_, s in results if s == 'cached')
failed = sum(1 for *_, s in results if s.startswith('FAILED'))
print(f"\n{ok} newly written, {cached} already cached, {failed} failed")
