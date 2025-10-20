"""
Detailed Analysis of Young Children (<1 and 1-2 Years)

Compares observed vs simulated data for youngest age groups:
1. Clinical episodes (unscheduled visits) - count and seasonality
2. Routine visit prevalence
3. Routine visit parasite density (among positives)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
SITE = 'nagongera'
SIMULATION_OUTPUT_DIR = Path('simulation_output')
FIGURES_OUTPUT_DIR = Path('figures')
FIGURES_OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("YOUNG CHILDREN ANALYSIS (<1 AND 1-2 YEARS)")
print("=" * 80)

# Load observed data
print("\nLoading observed PRISM data...")
obs_df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"  {len(obs_df)} observations, {obs_df['id'].nunique()} participants")

# Load simulated data
print("\nLoading simulated data...")
sim_df = pd.read_csv(SIMULATION_OUTPUT_DIR / 'household_visit_output.csv', parse_dates=['date'])

# Calculate age at visit for simulated data
sim_df['enrollment_date'] = sim_df.groupby('id')['date'].transform('min')
sim_df['days_since_enrollment'] = (sim_df['date'] - sim_df['enrollment_date']).dt.days
sim_df['age'] = sim_df['age_at_enrollment'] + (sim_df['days_since_enrollment'] / 365.25)

print(f"  {len(sim_df)} visits, {sim_df['id'].nunique()} simulated individuals")

# Define age groups
obs_df['age_group'] = pd.cut(obs_df['age'], bins=[0, 1, 2, 100], labels=['<1', '1-2', '2+'])
sim_df['age_group'] = pd.cut(sim_df['age'], bins=[0, 1, 2, 100], labels=['<1', '1-2', '2+'])

# Add month for seasonality analysis
obs_df['month'] = obs_df['date'].dt.month
sim_df['month'] = sim_df['date'].dt.month

# ==============================================================================
# 1. CLINICAL EPISODES (UNSCHEDULED VISITS WITH TREATMENT)
# ==============================================================================
print("\n" + "=" * 80)
print("1. CLINICAL EPISODES (UNSCHEDULED + ANTIMALARIAL TREATMENT)")
print("=" * 80)

# Filter to clinical episodes
obs_clinical = obs_df[
    (obs_df['visittype'] == 'Unscheduled visit') &
    (obs_df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate', case=False, na=False)) &
    (obs_df['parasitedensity'] > 0)
].copy()

sim_clinical = sim_df[
    (sim_df['visittype'] == 'Unscheduled visit') &
    (sim_df['antimalarial'].str.contains('Artemether|Quinine|Artesunate', case=False, na=False)) &
    (sim_df['parasitedensity'] > 0)
].copy()

for age_group in ['<1', '1-2']:
    print(f"\n{age_group} year olds:")

    obs_age = obs_clinical[obs_clinical['age_group'] == age_group]
    sim_age = sim_clinical[sim_clinical['age_group'] == age_group]

    # Count episodes
    obs_n_individuals = obs_df[obs_df['age_group'] == age_group]['id'].nunique()
    sim_n_individuals = sim_df[sim_df['age_group'] == age_group]['id'].nunique()

    obs_n_episodes = len(obs_age)
    sim_n_episodes = len(sim_age)

    # Calculate person-time (approximate from visit data)
    obs_person_years = obs_df[obs_df['age_group'] == age_group].groupby('id')['date'].apply(
        lambda x: (x.max() - x.min()).days / 365.25
    ).sum()
    sim_person_years = sim_df[sim_df['age_group'] == age_group].groupby('id')['date'].apply(
        lambda x: (x.max() - x.min()).days / 365.25
    ).sum()

    print(f"  Observed:")
    print(f"    Episodes: {obs_n_episodes} ({obs_n_individuals} individuals)")
    print(f"    Person-years: {obs_person_years:.1f}")
    print(f"    Incidence: {obs_n_episodes/obs_person_years:.2f} episodes/person-year")

    print(f"  Simulated:")
    print(f"    Episodes: {sim_n_episodes} ({sim_n_individuals} individuals)")
    print(f"    Person-years: {sim_person_years:.1f}")
    print(f"    Incidence: {sim_n_episodes/sim_person_years:.2f} episodes/person-year")

    # Density at clinical episodes
    if len(obs_age) > 0:
        print(f"  Observed density:")
        print(f"    Median: {obs_age['parasitedensity'].median():.0f} /μL")
        print(f"    Geometric mean: {np.exp(np.log(obs_age['parasitedensity']).mean()):.0f} /μL")

    if len(sim_age) > 0:
        print(f"  Simulated density:")
        print(f"    Median: {sim_age['parasitedensity'].median():.0f} /μL")
        print(f"    Geometric mean: {np.exp(np.log(sim_age['parasitedensity']).mean()):.0f} /μL")

# ==============================================================================
# 2. ROUTINE VISIT PREVALENCE
# ==============================================================================
print("\n" + "=" * 80)
print("2. ROUTINE VISIT PREVALENCE")
print("=" * 80)

# Filter to routine visits
obs_routine = obs_df[obs_df['visittype'] == 'Scheduled visit'].copy()
sim_routine = sim_df[sim_df['visittype'] == 'Scheduled visit'].copy()

for age_group in ['<1', '1-2']:
    print(f"\n{age_group} year olds:")

    obs_age = obs_routine[obs_routine['age_group'] == age_group]
    sim_age = sim_routine[sim_routine['age_group'] == age_group]

    obs_n_visits = len(obs_age[obs_age['parasitedensity'].notna()])
    obs_n_positive = (obs_age['parasitedensity'] > 0).sum()
    obs_prev = 100 * obs_n_positive / obs_n_visits if obs_n_visits > 0 else 0

    sim_n_visits = len(sim_age)
    sim_n_positive = (sim_age['parasitedensity'] > 0).sum()
    sim_prev = 100 * sim_n_positive / sim_n_visits if sim_n_visits > 0 else 0

    print(f"  Observed:")
    print(f"    Visits: {obs_n_visits}")
    print(f"    Positive: {obs_n_positive}")
    print(f"    Prevalence: {obs_prev:.1f}%")

    print(f"  Simulated:")
    print(f"    Visits: {sim_n_visits}")
    print(f"    Positive: {sim_n_positive}")
    print(f"    Prevalence: {sim_prev:.1f}%")

# ==============================================================================
# 3. ROUTINE VISIT DENSITY (AMONG POSITIVES)
# ==============================================================================
print("\n" + "=" * 80)
print("3. ROUTINE VISIT PARASITE DENSITY (AMONG POSITIVES)")
print("=" * 80)

for age_group in ['<1', '1-2']:
    print(f"\n{age_group} year olds:")

    obs_age_pos = obs_routine[(obs_routine['age_group'] == age_group) & (obs_routine['parasitedensity'] > 0)]
    sim_age_pos = sim_routine[(sim_routine['age_group'] == age_group) & (sim_routine['parasitedensity'] > 0)]

    if len(obs_age_pos) > 0:
        print(f"  Observed density (n={len(obs_age_pos)}):")
        print(f"    Median: {obs_age_pos['parasitedensity'].median():.0f} /μL")
        print(f"    Geometric mean: {np.exp(np.log(obs_age_pos['parasitedensity']).mean()):.0f} /μL")
        print(f"    Range: {obs_age_pos['parasitedensity'].min():.0f}-{obs_age_pos['parasitedensity'].max():.0f}")

    if len(sim_age_pos) > 0:
        print(f"  Simulated density (n={len(sim_age_pos)}):")
        print(f"    Median: {sim_age_pos['parasitedensity'].median():.0f} /μL")
        print(f"    Geometric mean: {np.exp(np.log(sim_age_pos['parasitedensity']).mean()):.0f} /μL")
        print(f"    Range: {sim_age_pos['parasitedensity'].min():.0f}-{sim_age_pos['parasitedensity'].max():.0f}")

# ==============================================================================
# 4. SEASONALITY OF CLINICAL EPISODES
# ==============================================================================
print("\n" + "=" * 80)
print("4. SEASONALITY OF CLINICAL EPISODES")
print("=" * 80)

for age_group in ['<1', '1-2']:
    print(f"\n{age_group} year olds:")

    obs_age = obs_clinical[obs_clinical['age_group'] == age_group]
    sim_age = sim_clinical[sim_clinical['age_group'] == age_group]

    obs_by_month = obs_age.groupby('month').size()
    sim_by_month = sim_age.groupby('month').size()

    print(f"  Observed episodes by month:")
    for month in range(1, 13):
        count = obs_by_month.get(month, 0)
        print(f"    Month {month:2d}: {count:3d}")

    print(f"  Simulated episodes by month:")
    for month in range(1, 13):
        count = sim_by_month.get(month, 0)
        print(f"    Month {month:2d}: {count:3d}")

# ==============================================================================
# 5. VISUALIZATIONS
# ==============================================================================
print("\n" + "=" * 80)
print("5. CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Panel 1: Clinical episode incidence
ax = axes[0, 0]
age_groups = ['<1', '1-2']
x = np.arange(len(age_groups))
width = 0.35

obs_incidence = []
sim_incidence = []

for age_group in age_groups:
    obs_age = obs_clinical[obs_clinical['age_group'] == age_group]
    sim_age = sim_clinical[sim_clinical['age_group'] == age_group]

    obs_py = obs_df[obs_df['age_group'] == age_group].groupby('id')['date'].apply(
        lambda x: (x.max() - x.min()).days / 365.25
    ).sum()
    sim_py = sim_df[sim_df['age_group'] == age_group].groupby('id')['date'].apply(
        lambda x: (x.max() - x.min()).days / 365.25
    ).sum()

    obs_incidence.append(len(obs_age) / obs_py if obs_py > 0 else 0)
    sim_incidence.append(len(sim_age) / sim_py if sim_py > 0 else 0)

ax.bar(x - width/2, obs_incidence, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, sim_incidence, width, label='Simulated', alpha=0.7)
ax.set_xlabel('Age Group')
ax.set_ylabel('Clinical Episodes per Person-Year')
ax.set_title('Clinical Malaria Incidence')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 2: Routine visit prevalence
ax = axes[0, 1]
obs_prev_vals = []
sim_prev_vals = []

for age_group in age_groups:
    obs_age = obs_routine[obs_routine['age_group'] == age_group]
    sim_age = sim_routine[sim_routine['age_group'] == age_group]

    obs_n = len(obs_age[obs_age['parasitedensity'].notna()])
    obs_pos = (obs_age['parasitedensity'] > 0).sum()
    obs_prev_vals.append(100 * obs_pos / obs_n if obs_n > 0 else 0)

    sim_n = len(sim_age)
    sim_pos = (sim_age['parasitedensity'] > 0).sum()
    sim_prev_vals.append(100 * sim_pos / sim_n if sim_n > 0 else 0)

ax.bar(x - width/2, obs_prev_vals, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, sim_prev_vals, width, label='Simulated', alpha=0.7)
ax.set_xlabel('Age Group')
ax.set_ylabel('Prevalence (%)')
ax.set_title('Routine Visit Prevalence')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 3: Routine visit density (geometric mean)
ax = axes[0, 2]
obs_geomean = []
sim_geomean = []

for age_group in age_groups:
    obs_age_pos = obs_routine[(obs_routine['age_group'] == age_group) & (obs_routine['parasitedensity'] > 0)]
    sim_age_pos = sim_routine[(sim_routine['age_group'] == age_group) & (sim_routine['parasitedensity'] > 0)]

    if len(obs_age_pos) > 0:
        obs_geomean.append(np.exp(np.log(obs_age_pos['parasitedensity']).mean()))
    else:
        obs_geomean.append(0)

    if len(sim_age_pos) > 0:
        sim_geomean.append(np.exp(np.log(sim_age_pos['parasitedensity']).mean()))
    else:
        sim_geomean.append(0)

ax.bar(x - width/2, obs_geomean, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, sim_geomean, width, label='Simulated', alpha=0.7)
ax.set_xlabel('Age Group')
ax.set_ylabel('Geometric Mean Density (/μL)')
ax.set_title('Routine Visit Density (among positives)')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.set_yscale('log')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 4-5: Seasonality of clinical episodes
for idx, age_group in enumerate(age_groups):
    ax = axes[1, idx]

    obs_age = obs_clinical[obs_clinical['age_group'] == age_group]
    sim_age = sim_clinical[sim_clinical['age_group'] == age_group]

    obs_by_month = obs_age.groupby('month').size().reindex(range(1, 13), fill_value=0)
    sim_by_month = sim_age.groupby('month').size().reindex(range(1, 13), fill_value=0)

    months = range(1, 13)
    width_month = 0.35
    x_months = np.arange(12)

    ax.bar(x_months - width_month/2, obs_by_month.values, width_month, label='Observed', alpha=0.7)
    ax.bar(x_months + width_month/2, sim_by_month.values, width_month, label='Simulated', alpha=0.7)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Clinical Episodes')
    ax.set_title(f'Seasonality: {age_group} years')
    ax.set_xticks(x_months)
    ax.set_xticklabels(months)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

# Panel 6: Clinical episode densities (strip plot for both age groups)
ax = axes[1, 2]

age_positions = {'<1': 0, '1-2': 1}
np.random.seed(42)

for age_group in age_groups:
    obs_data = obs_clinical[obs_clinical['age_group'] == age_group]['parasitedensity']
    sim_data = sim_clinical[sim_clinical['age_group'] == age_group]['parasitedensity']

    x_pos = age_positions[age_group]

    if len(obs_data) > 0:
        jitter = np.random.normal(0, 0.06, len(obs_data))
        ax.scatter(x_pos - 0.2 + jitter, obs_data, alpha=0.3, s=20, color='blue',
                  label='Observed' if age_group == '<1' else '')
        ax.plot([x_pos - 0.3, x_pos - 0.1], [obs_data.median(), obs_data.median()],
               'b-', linewidth=3, alpha=0.8)

    if len(sim_data) > 0:
        jitter = np.random.normal(0, 0.06, len(sim_data))
        ax.scatter(x_pos + 0.2 + jitter, sim_data, alpha=0.5, s=30, color='orange',
                  label='Simulated' if age_group == '<1' else '')
        ax.plot([x_pos + 0.1, x_pos + 0.3], [sim_data.median(), sim_data.median()],
               color='orange', linewidth=3, alpha=0.8)

ax.set_yscale('log')
ax.set_xlabel('Age Group')
ax.set_ylabel('Parasite Density (/μL)')
ax.set_title('Clinical Episode Densities\n(individual points, lines = medians)')
ax.set_xticks([0, 1])
ax.set_xticklabels(age_groups)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([10, 1e6])

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'young_children_comparison.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\n  Saved: {fig_file}")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
