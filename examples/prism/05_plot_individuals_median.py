"""
Individual Trajectory Plotting - Median/Representative Sampling

Creates alternative versions of the individual plots focusing on:
1. Median-infected individuals from each age bin (not extreme high)
2. Evenly-spaced sampling through ranked distribution for overview
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from plotting_utils import plot_individual_detailed, plot_individual_compact

# Load cleaned data
print("Loading cleaned PRISM data...")
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'

df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"Loaded {len(df)} observations for {df['id'].nunique()} participants")

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("EXAMPLE 1: DETAILED - MEDIAN INFECTED FROM EACH AGE BIN")
print("=" * 80)

# Get participant statistics
participant_stats = df.groupby('id').agg({
    'parasitedensity': lambda x: (x > 0).sum(),
    'date': 'count',
    'gametocytes': lambda x: (x == 'Yes').sum(),
    'LAMP': lambda x: (x == 'Positive').sum()
}).rename(columns={
    'parasitedensity': 'n_positive',
    'date': 'n_visits',
    'gametocytes': 'n_gametocytes',
    'LAMP': 'n_lamp_positive'
})

# Filter to individuals with at least 20 visits
frequent = participant_stats[participant_stats['n_visits'] >= 20]

# Get enrollment age for each participant
enrollment_ages = df.groupby('id')['age_at_enrollment'].first()
frequent_with_age = frequent.copy()
frequent_with_age['age_at_enrollment'] = frequent_with_age.index.map(enrollment_ages)

# Define age bins at enrollment
age_bins_enroll = [0, 1, 2, 3, 5, 10, 100]
age_labels_enroll = ['0-1', '1-2', '2-3', '3-5', '5-10', '10+']

# Select MEDIAN infected individual from each age bin
selected_ids = []
for i in range(len(age_bins_enroll) - 1):
    age_min = age_bins_enroll[i]
    age_max = age_bins_enroll[i + 1]

    # Get individuals in this age bin at enrollment
    in_age_bin = frequent_with_age[
        (frequent_with_age['age_at_enrollment'] >= age_min) &
        (frequent_with_age['age_at_enrollment'] < age_max)
    ]

    if len(in_age_bin) > 0:
        # Sort by infection count and get median
        sorted_bin = in_age_bin.sort_values('n_positive')
        median_idx = len(sorted_bin) // 2
        median_infected = sorted_bin.iloc[median_idx].name
        selected_ids.append(median_infected)

print(f"\nPlotting {len(selected_ids)} individuals (median infected from each age bin)...")

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, participant_id in enumerate(selected_ids):
    person_data = df[df['id'] == participant_id].copy()

    n_positive = (person_data['parasitedensity'] > 0).sum()
    n_visits = len(person_data)
    n_gam = (person_data['gametocytes'] == 'Yes').sum()
    n_lamp = (person_data['LAMP'] == 'Positive').sum()

    # Get enrollment age and gender
    enroll_age = person_data['age_at_enrollment'].iloc[0]
    gender = person_data['gender'].iloc[0]
    gender_abbrev = gender[0] if pd.notna(gender) else '?'

    plot_individual_detailed(person_data, ax=axes[idx])

    title = f"ID {participant_id} ({int(enroll_age)}{gender_abbrev}): {n_positive}/{n_visits} positive"
    if n_gam > 0:
        title += f", {n_gam} gametocytes"
    if n_lamp > 0:
        title += f", {n_lamp} LAMP+"
    axes[idx].set_title(title, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '07b_individual_detailed_median.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '07b_individual_detailed_median.png'}")
plt.close()

print("\n" + "=" * 80)
print("EXAMPLE 2: COMPACT - EVENLY-SPACED THROUGH RANKED DISTRIBUTION")
print("=" * 80)

# For compact view, sample evenly through infection count distribution
# within each age group
df['age_group'] = pd.cut(df['age'], bins=[0, 2, 5, 10, 100], labels=['0-2', '2-5', '5-10', '10+'])

sample_ids = []
n_per_group = 15  # Still 15 per age group for total of 60

for age_group in ['0-2', '2-5', '5-10', '10+']:
    age_group_data = df[df['age_group'] == age_group]
    visit_counts = age_group_data.groupby('id').size()
    frequent_ids = visit_counts[visit_counts >= 10].index.tolist()

    if len(frequent_ids) > 0:
        # Get infection counts for this age group
        infection_counts = age_group_data[age_group_data['id'].isin(frequent_ids)].groupby('id').agg({
            'parasitedensity': lambda x: (x > 0).sum()
        })
        infection_counts = infection_counts.sort_values('parasitedensity', ascending=False)

        # Sample evenly through the distribution
        n_available = len(infection_counts)
        if n_available <= n_per_group:
            # Take all if we have fewer than target
            sample_ids.extend(infection_counts.index.tolist())
        else:
            # Sample evenly spaced indices
            indices = np.linspace(0, n_available - 1, n_per_group, dtype=int)
            sampled = [infection_counts.index[i] for i in indices]
            sample_ids.extend(sampled)

print(f"\nPlotting {len(sample_ids)} individuals (evenly-spaced through infection distribution)...")

# Filter to selected individuals
subset = df[df['id'].isin(sample_ids)].copy()

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
plot_individual_compact(subset, ax=ax, fig=fig, detailed=False)
ax.set_title('PRISM Individual Trajectories: Representative Sample\n' +
             '(Evenly-spaced through infection count distribution for each age group: 0-2, 2-5, 5-10, 10+ years)',
             fontsize=14, fontweight='bold')

plt.savefig(output_dir / '08b_individuals_compact_representative.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '08b_individuals_compact_representative.png'}")
plt.close()

print("\n" + "=" * 80)
print("EXAMPLE 3: COMPACT DETAILED - MEDIAN INDIVIDUALS")
print("=" * 80)

# Use the same median individuals selected for detailed plots
subset_median = df[df['id'].isin(selected_ids)].copy()

print(f"\nPlotting {len(selected_ids)} individuals (median infected) with detailed annotations...")

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
plot_individual_compact(subset_median, ax=ax, fig=fig, detailed=True)
ax.set_title('PRISM Individual Trajectories: Median Infected (Detailed Annotations)\n' +
             '(Median infected individual from each enrollment age bin)',
             fontsize=14, fontweight='bold')

plt.savefig(output_dir / '09b_individuals_compact_median_detailed.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '09b_individuals_compact_median_detailed.png'}")
plt.close()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nCreated alternative (median/representative) versions:")
print("  1. Detailed panels: MEDIAN infected from each age bin")
print("  2. Compact overview: EVENLY-SPACED through infection distribution")
print("  3. Compact detailed: MEDIAN infected with annotations")
print(f"\nAll plots saved to: {output_dir}/")
print("\nComparison:")
print("  - Original (04_plot_individuals.py): Most infected (extreme cases)")
print("  - Median (05_plot_individuals_median.py): Typical/representative patterns")
