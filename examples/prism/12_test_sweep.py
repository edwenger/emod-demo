"""
Quick test of parameter sweep functionality
Tests a single parameter combination to verify the pipeline works
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from seasonal_challenge import (
    monthly_eir_challenge,
    surface_area_biting_function,
    ComplexTreatment,
)
from emodlib.malaria import IntrahostComponent, create_config

print("Testing parameter sweep pipeline...")

# Test parameters
monthly_eir = 0.5
treatment_coverage = 0.75
age_at_enrollment = 2.5  # 2.5 years old

# Seasonality
SEASONALITY_AMPLIFICATION = 3.0
BASE_MONTHLY_MULTIPLIER = np.array([
    0.40, 0.30, 0.25, 0.50, 1.50, 2.00,
    1.80, 1.40, 1.00, 0.80, 0.60, 0.45
])
MONTHLY_MULTIPLIER = BASE_MONTHLY_MULTIPLIER * (SEASONALITY_AMPLIFICATION / BASE_MONTHLY_MULTIPLIER.mean())
monthly_eirs = MONTHLY_MULTIPLIER * monthly_eir

print(f"\n1. Testing individual initialization...")
config = create_config()
ic = IntrahostComponent.create(config)
ic.susceptibility.age = 0.0
print(f"   ✓ Created IntrahostComponent, age={ic.susceptibility.age}")

print(f"\n2. Testing burn-in (aging to {age_at_enrollment} years)...")
burnin_days = int(age_at_enrollment * 365.25)
burnin_treatment = ComplexTreatment(prob_treat=0.5)

if burnin_days > 0:
    monthly_eir_challenge(
        duration=burnin_days,
        monthly_eirs=monthly_eirs,
        callback=burnin_treatment,
        ic=ic,
        start_day=0,
    )
print(f"   ✓ Burn-in complete, age={ic.susceptibility.age/365.25:.1f} years")

print(f"\n3. Testing trial period simulation (4 years)...")
trial_days = int(4.0 * 365.25)
trial_treatment = ComplexTreatment(prob_treat=treatment_coverage)

trial_df = monthly_eir_challenge(
    duration=trial_days,
    monthly_eirs=monthly_eirs,
    callback=trial_treatment,
    ic=ic,
    start_day=burnin_days,
)

print(f"   ✓ Simulation complete")
print(f"   DataFrame shape: {trial_df.shape}")
print(f"   Columns: {list(trial_df.columns)}")

print(f"\n4. Testing data processing...")
trial_df = trial_df.reset_index()
if 'days' in trial_df.columns:
    trial_df = trial_df.rename(columns={'days': 'day'})

trial_df['age'] = age_at_enrollment + (trial_df['day'] / 365.25)
trial_df['clinical_episode'] = (
    (trial_df['fever_temperature'] > 38.5) &
    (trial_df['parasite_density'] > 0) &
    (trial_df['treatments'] > 0)
)

n_clinical = trial_df['clinical_episode'].sum()
n_positive = (trial_df['parasite_density'] > 0).sum()
max_density = trial_df['parasite_density'].max()

print(f"   ✓ Data processed")
print(f"   Clinical episodes: {n_clinical}")
print(f"   Positive days: {n_positive}")
print(f"   Max parasite density: {max_density:.0f}")

print(f"\n✓ All tests passed! Parameter sweep should work.")
