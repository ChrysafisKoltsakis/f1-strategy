"""
Bronze-layer ingestion for the first half of a season: pull race sessions
for multiple circuits and dump the essential lap/weather/race-control
columns to Parquet, unmodified, as the historical source of truth.

This deliberately does NOT apply the fuel-correction / degradation /
pit-loss logic from 00_explore_one_race.py -- that logic still has known
issues (SOFT compound negative degradation, outlier laps surviving the
green-flag filter) that need fixing before they're trusted across many
races. This script just gets clean raw data landed so that fix can be
made once and re-run everywhere, instead of per-race.

Run with: python notebooks/01_pull_half_season.py
"""
import fastf1
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache('data/fastf1_cache')

YEAR = 2023
BRONZE_DIR = Path('data/bronze')
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
# "Half a season" -> everything up to and including the summer break.
# 2023 had 22 rounds; round 12 (Belgium) was the last one before the break.
half_season = schedule[schedule['RoundNumber'] <= 12]

print(f"Pulling {len(half_season)} race weekends from {YEAR}:")
print(half_season[['RoundNumber', 'EventName', 'Location']].to_string(index=False))
print()

LAP_COLS = [
    'Driver', 'DriverNumber', 'LapNumber', 'LapTime', 'Sector1Time',
    'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife', 'FreshTyre',
    'Stint', 'PitInTime', 'PitOutTime', 'TrackStatus', 'Position',
    'IsPersonalBest', 'DeletedReason',
]

results = []
for _, event in half_season.iterrows():
    rnd, name = event['RoundNumber'], event['EventName']
    out_path = BRONZE_DIR / f"{YEAR}_r{rnd:02d}_laps.parquet"
    if out_path.exists():
        print(f"[{rnd:2d}] {name}: already cached, skipping")
        results.append((rnd, name, 'cached'))
        continue

    try:
        session = fastf1.get_session(YEAR, rnd, 'R')
        session.load(telemetry=False, weather=True, messages=True)

        laps = session.laps[LAP_COLS].copy()
        laps['Year'] = YEAR
        laps['RoundNumber'] = rnd
        laps['EventName'] = name
        laps.to_parquet(out_path, index=False)

        weather_path = BRONZE_DIR / f"{YEAR}_r{rnd:02d}_weather.parquet"
        session.weather_data.to_parquet(weather_path, index=False)

        rc_path = BRONZE_DIR / f"{YEAR}_r{rnd:02d}_race_control.parquet"
        session.race_control_messages.to_parquet(rc_path, index=False)

        print(f"[{rnd:2d}] {name}: {len(laps)} laps written")
        results.append((rnd, name, f'{len(laps)} laps'))
    except Exception as e:
        print(f"[{rnd:2d}] {name}: FAILED - {e}")
        results.append((rnd, name, f'FAILED: {e}'))

print("\n--- Summary ---")
for rnd, name, status in results:
    print(f"  r{rnd:02d}  {name:30s} {status}")
