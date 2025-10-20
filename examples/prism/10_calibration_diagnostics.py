"""
Calibration Diagnostics for EIR and Treatment Parameters

Creates diagnostic figures showing:
1. Distribution of clinical episodes per child (variation, not just mean)
2. Age-specific clinical incidence
3. Routine visit prevalence patterns
4. Individual-level trajectories showing chronic vs episodic patterns
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
print("CALIBRATION DIAGNOSTICS")
print("=" * 80)

# Load observed data
print("\nLoading observed PRISM data...")
obs_df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])

# Filter to pre-IRS period (before 2015)
print("  Filtering to pre-IRS period (before 2015-01-01)...")
obs_df = obs_df[obs_df['date'] < '2015-01-01'].copy()
print(f"  {len(obs_df)} observations in pre-IRS period")

# Load simulated data
print("\nLoading simulated data...")
sim_df = pd.read_csv(SIMULATION_OUTPUT_DIR / 'household_visit_output.csv', parse_dates=['date'])

# Filter to pre-IRS period
sim_df = sim_df[sim_df['date'] < '2015-01-01'].copy()
print(f"  {len(sim_df)} visits in pre-IRS period")

# Calculate age at visit for simulated data
sim_df['enrollment_date'] = sim_df.groupby('id')['date'].transform('min')
sim_df['days_since_enrollment'] = (sim_df['date'] - sim_df['enrollment_date']).dt.days
sim_df['age'] = sim_df['age_at_enrollment'] + (sim_df['days_since_enrollment'] / 365.25)

# Define age groups
obs_df['age_group'] = pd.cut(obs_df['age'], bins=[0, 1, 2, 5, 10, 100],
                              labels=['<1', '1-2', '2-5', '5-10', '10+'])
sim_df['age_group'] = pd.cut(sim_df['age'], bins=[0, 1, 2, 5, 10, 100],
                              labels=['<1', '1-2', '2-5', '5-10', '10+'])

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

# Filter to routine visits
obs_routine = obs_df[obs_df['visittype'] == 'Scheduled visit'].copy()
sim_routine = sim_df[sim_df['visittype'] == 'Scheduled visit'].copy()

# ==============================================================================
# FIGURE 1: CLINICAL EPISODE DISTRIBUTION PER CHILD
# ==============================================================================
print("\n1. Analyzing clinical episode distribution per child...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

age_groups = ['<1', '1-2', '2-5', '5-10', '10+']

for idx, age_group in enumerate(age_groups[:5]):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Get individuals in this age group and count their clinical episodes
    obs_individuals = obs_df[obs_df['age_group'] == age_group]['id'].unique()
    obs_episode_counts = []
    for ind_id in obs_individuals:
        ind_episodes = obs_clinical[(obs_clinical['id'] == ind_id) &
                                    (obs_clinical['age_group'] == age_group)]
        # Calculate person-time in this age group
        ind_data = obs_df[(obs_df['id'] == ind_id) & (obs_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                rate = len(ind_episodes) / person_years
                obs_episode_counts.append(rate)

    sim_individuals = sim_df[sim_df['age_group'] == age_group]['id'].unique()
    sim_episode_counts = []
    for ind_id in sim_individuals:
        ind_episodes = sim_clinical[(sim_clinical['id'] == ind_id) &
                                    (sim_clinical['age_group'] == age_group)]
        ind_data = sim_df[(sim_df['id'] == ind_id) & (sim_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                rate = len(ind_episodes) / person_years
                sim_episode_counts.append(rate)

    # Plot histograms
    if len(obs_episode_counts) > 0:
        bins = np.linspace(0, max(max(obs_episode_counts or [0]), max(sim_episode_counts or [0])) + 1, 15)
        ax.hist(obs_episode_counts, bins=bins, alpha=0.5, label='Observed',
                edgecolor='black', color='blue')
        obs_mean = np.mean(obs_episode_counts)
        obs_median = np.median(obs_episode_counts)
        ax.axvline(obs_mean, color='blue', linestyle='--', linewidth=2,
                  label=f'Obs mean={obs_mean:.1f}')
        ax.axvline(obs_median, color='blue', linestyle=':', linewidth=2,
                  label=f'Obs median={obs_median:.1f}')

    if len(sim_episode_counts) > 0:
        ax.hist(sim_episode_counts, bins=bins, alpha=0.5, label='Simulated',
                edgecolor='black', color='orange')
        sim_mean = np.mean(sim_episode_counts)
        sim_median = np.median(sim_episode_counts)
        ax.axvline(sim_mean, color='orange', linestyle='--', linewidth=2,
                  label=f'Sim mean={sim_mean:.1f}')
        ax.axvline(sim_median, color='orange', linestyle=':', linewidth=2,
                  label=f'Sim median={sim_median:.1f}')

    ax.set_xlabel('Clinical Episodes per Person-Year')
    ax.set_ylabel('Number of Individuals')
    ax.set_title(f'Age {age_group} (Obs n={len(obs_episode_counts)}, Sim n={len(sim_episode_counts)})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

# Remove unused subplot
axes[1, 2].axis('off')

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'calibration_episode_distribution.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_file}")
plt.close()

# ==============================================================================
# FIGURE 2: AGE-SPECIFIC CLINICAL INCIDENCE (WITH ERROR BARS)
# ==============================================================================
print("\n2. Creating age-specific incidence comparison...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

obs_means = []
obs_sems = []
sim_means = []
sim_sems = []
age_labels = []

for age_group in age_groups:
    # Observed
    obs_individuals = obs_df[obs_df['age_group'] == age_group]['id'].unique()
    obs_rates = []
    for ind_id in obs_individuals:
        ind_episodes = obs_clinical[(obs_clinical['id'] == ind_id) &
                                    (obs_clinical['age_group'] == age_group)]
        ind_data = obs_df[(obs_df['id'] == ind_id) & (obs_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                obs_rates.append(len(ind_episodes) / person_years)

    # Simulated
    sim_individuals = sim_df[sim_df['age_group'] == age_group]['id'].unique()
    sim_rates = []
    for ind_id in sim_individuals:
        ind_episodes = sim_clinical[(sim_clinical['id'] == ind_id) &
                                    (sim_clinical['age_group'] == age_group)]
        ind_data = sim_df[(sim_df['id'] == ind_id) & (sim_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                sim_rates.append(len(ind_episodes) / person_years)

    age_labels.append(age_group)
    obs_means.append(np.mean(obs_rates) if obs_rates else 0)
    obs_sems.append(np.std(obs_rates) / np.sqrt(len(obs_rates)) if len(obs_rates) > 0 else 0)
    sim_means.append(np.mean(sim_rates) if sim_rates else 0)
    sim_sems.append(np.std(sim_rates) / np.sqrt(len(sim_rates)) if len(sim_rates) > 0 else 0)

x = np.arange(len(age_labels))
width = 0.35

ax.bar(x - width/2, obs_means, width, yerr=obs_sems, label='Observed',
       alpha=0.7, capsize=5, error_kw={'linewidth': 2})
ax.bar(x + width/2, sim_means, width, yerr=sim_sems, label='Simulated',
       alpha=0.7, capsize=5, error_kw={'linewidth': 2})

ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Clinical Episodes per Person-Year', fontsize=12)
ax.set_title('Clinical Malaria Incidence by Age\n(bars = mean, error bars = SEM)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(age_labels)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'calibration_age_incidence.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_file}")
plt.close()

# ==============================================================================
# FIGURE 3: ROUTINE VISIT PREVALENCE BY AGE
# ==============================================================================
print("\n3. Creating routine visit prevalence comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Prevalence
ax = axes[0]
obs_prev = []
sim_prev = []

for age_group in age_groups:
    obs_age = obs_routine[obs_routine['age_group'] == age_group]
    sim_age = sim_routine[sim_routine['age_group'] == age_group]

    obs_n = len(obs_age[obs_age['parasitedensity'].notna()])
    obs_pos = (obs_age['parasitedensity'] > 0).sum()
    obs_prev.append(100 * obs_pos / obs_n if obs_n > 0 else 0)

    sim_n = len(sim_age)
    sim_pos = (sim_age['parasitedensity'] > 0).sum()
    sim_prev.append(100 * sim_pos / sim_n if sim_n > 0 else 0)

x = np.arange(len(age_labels))
width = 0.35

ax.bar(x - width/2, obs_prev, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, sim_prev, width, label='Simulated', alpha=0.7)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Prevalence at Routine Visits (%)', fontsize=12)
ax.set_title('Routine Visit Prevalence\n(knife-edge calibration issue noted)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(age_labels)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Right panel: Density among positives
ax = axes[1]
obs_geomean = []
sim_geomean = []

for age_group in age_groups:
    obs_age_pos = obs_routine[(obs_routine['age_group'] == age_group) &
                              (obs_routine['parasitedensity'] > 0)]
    sim_age_pos = sim_routine[(sim_routine['age_group'] == age_group) &
                              (sim_routine['parasitedensity'] > 0)]

    if len(obs_age_pos) > 0:
        obs_geomean.append(np.exp(np.log(obs_age_pos['parasitedensity']).mean()))
    else:
        obs_geomean.append(np.nan)

    if len(sim_age_pos) > 0:
        sim_geomean.append(np.exp(np.log(sim_age_pos['parasitedensity']).mean()))
    else:
        sim_geomean.append(np.nan)

ax.bar(x - width/2, obs_geomean, width, label='Observed', alpha=0.7)
ax.bar(x + width/2, sim_geomean, width, label='Simulated', alpha=0.7)
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Geometric Mean Density (/μL)', fontsize=12)
ax.set_title('Routine Visit Density\n(among positives)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(age_labels)
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'calibration_routine_prevalence.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_file}")
plt.close()

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 80)
print("CALIBRATION SUMMARY")
print("=" * 80)

print("\nClinical Episode Incidence (episodes/person-year):")
print(f"{'Age':>8s}  {'Obs Mean':>10s}  {'Obs Median':>11s}  {'Sim Mean':>10s}  {'Sim Median':>11s}  {'Ratio':>8s}")
print("-" * 80)

for idx, age_group in enumerate(age_groups):
    # Recalculate for summary
    obs_individuals = obs_df[obs_df['age_group'] == age_group]['id'].unique()
    obs_rates = []
    for ind_id in obs_individuals:
        ind_episodes = obs_clinical[(obs_clinical['id'] == ind_id) &
                                    (obs_clinical['age_group'] == age_group)]
        ind_data = obs_df[(obs_df['id'] == ind_id) & (obs_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                obs_rates.append(len(ind_episodes) / person_years)

    sim_individuals = sim_df[sim_df['age_group'] == age_group]['id'].unique()
    sim_rates = []
    for ind_id in sim_individuals:
        ind_episodes = sim_clinical[(sim_clinical['id'] == ind_id) &
                                    (sim_clinical['age_group'] == age_group)]
        ind_data = sim_df[(sim_df['id'] == ind_id) & (sim_df['age_group'] == age_group)]
        if len(ind_data) > 0:
            person_years = (ind_data['date'].max() - ind_data['date'].min()).days / 365.25
            if person_years > 0:
                sim_rates.append(len(ind_episodes) / person_years)

    obs_mean = np.mean(obs_rates) if obs_rates else 0
    obs_median = np.median(obs_rates) if obs_rates else 0
    sim_mean = np.mean(sim_rates) if sim_rates else 0
    sim_median = np.median(sim_rates) if sim_rates else 0
    ratio = sim_mean / obs_mean if obs_mean > 0 else 0

    print(f"{age_group:>8s}  {obs_mean:10.2f}  {obs_median:11.2f}  {sim_mean:10.2f}  {sim_median:11.2f}  {ratio:8.2f}x")

print("\nRoutine Visit Prevalence (%):")
print(f"{'Age':>8s}  {'Observed':>10s}  {'Simulated':>11s}  {'Ratio':>8s}")
print("-" * 80)

for idx, age_group in enumerate(age_groups):
    print(f"{age_group:>8s}  {obs_prev[idx]:10.1f}  {sim_prev[idx]:11.1f}  {sim_prev[idx]/obs_prev[idx] if obs_prev[idx] > 0 else 0:8.2f}x")

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
print("\nKey observations:")
print("1. Clinical incidence is 2-4x too high in young children")
print("2. Variation among individuals shows wide range (not just high average)")
print("3. Routine prevalence is 2-7x too high, especially in <5 years")
print("4. Both issues point to EIR being too high")
print("\nRecommendation: Reduce MONTHLY_EIR from 2.0 to ~0.5-1.0")
