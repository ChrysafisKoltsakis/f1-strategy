"""
Phase 0 proof-of-concept: pull one race, compute fuel-corrected lap times,
fit a per-compound degradation curve, derive an empirical pit-loss constant,
and sanity-check that these numbers look sane before building any pipeline.

Run with: python notebooks/00_explore_one_race.py
(or paste cells into a Jupyter notebook / VS Code interactive window)
"""
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cache avoids re-downloading from the F1 timing API / Jolpica-F1 on every run.
fastf1.Cache.enable_cache('data/fastf1_cache')

# Pick a race with a fairly "normal" strategy story to start with.
YEAR, GP, SESSION = 2023, 'Spain', 'R'

session = fastf1.get_session(YEAR, GP, SESSION)
session.load()

laps = session.laps.copy()
print(f"Loaded {len(laps)} laps across {laps['Driver'].nunique()} drivers")
print(laps[['Driver', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'PitInTime', 'PitOutTime']].head(10))

# --- Fuel correction ---------------------------------------------------
# Rough model: car starts ~110kg of fuel, burns down to ~0 by race end,
# ~0.03s/kg lap time penalty. This is a coarse linear approximation —
# good enough to check if the *shape* of degradation curves is sane.
STARTING_FUEL_KG = 110
FUEL_EFFECT_S_PER_KG = 0.03
total_laps = laps['LapNumber'].max()

laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
laps['FuelKgRemaining'] = STARTING_FUEL_KG * (1 - laps['LapNumber'] / total_laps)
laps['FuelCorrectedLapTime'] = laps['LapTimeSeconds'] - laps['FuelKgRemaining'] * FUEL_EFFECT_S_PER_KG

# --- Degradation curve per compound ------------------------------------
# Drop in/out laps and obvious outliers (safety car, red flag, pit laps)
# before fitting — those aren't representative of green-flag pace.
clean = laps[
    laps['PitInTime'].isna() &
    laps['PitOutTime'].isna() &
    laps['LapTimeSeconds'].notna() &
    laps['TrackStatus'].astype(str).isin(['1'])  # '1' = green flag in FastF1 encoding
].copy()

print(f"\n{len(clean)} clean green-flag laps after filtering "
      f"({len(laps) - len(clean)} dropped: pit laps / non-green track status / missing times)")

for compound in clean['Compound'].dropna().unique():
    sub = clean[clean['Compound'] == compound]
    if len(sub) < 10:
        continue
    coeffs = np.polyfit(sub['TyreLife'], sub['FuelCorrectedLapTime'], 1)
    slope, intercept = coeffs
    print(f"{compound:8s}: {len(sub):4d} laps, "
          f"degradation ~{slope:+.3f} s/lap, intercept {intercept:.2f}s")

# --- Empirical pit-loss constant ----------------------------------------
# Pit loss = delta between a lap with a pit stop and the surrounding
# green-flag pace, not a looked-up constant.
pit_laps = laps[laps['PitInTime'].notna() | laps['PitOutTime'].notna()]
if len(pit_laps) > 0:
    avg_green_pace = clean['LapTimeSeconds'].median()
    avg_pit_lap_time = pit_laps['LapTimeSeconds'].dropna().median()
    print(f"\nMedian green-flag lap: {avg_green_pace:.2f}s")
    print(f"Median pit-in/out lap: {avg_pit_lap_time:.2f}s")
    print(f"Rough pit-loss estimate: {avg_pit_lap_time - avg_green_pace:.2f}s "
          f"(crude — real pit loss should sum the in-lap + stationary + out-lap delta "
          f"against a green-flag baseline, this is just a first sanity check)")

# --- Plot for visual sanity check ---------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for compound in clean['Compound'].dropna().unique():
    sub = clean[clean['Compound'] == compound]
    if len(sub) < 10:
        continue
    ax.scatter(sub['TyreLife'], sub['FuelCorrectedLapTime'], label=compound, alpha=0.5, s=15)
ax.set_xlabel('Tyre age (laps)')
ax.set_ylabel('Fuel-corrected lap time (s)')
ax.set_title(f'{YEAR} {GP} — degradation by compound')
ax.legend()
fig.savefig('data/degradation_check.png', dpi=120)
print("\nSaved plot to data/degradation_check.png")
