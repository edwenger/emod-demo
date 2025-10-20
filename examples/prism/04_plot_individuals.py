"""
Individual Trajectory Plotting Examples

Demonstrates both plotting styles:
1. Detailed single-individual plots
2. Compact multi-individual overview
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
print("EXAMPLE 1: DETAILED SINGLE-INDIVIDUAL PLOTS")
print("=" * 80)

# Select a few individuals with interesting patterns
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

# Select top infected individual from each age bin (at enrollment)
# Get enrollment age for each participant
enrollment_ages = df.groupby('id')['age_at_enrollment'].first()
frequent_with_age = frequent.copy()
frequent_with_age['age_at_enrollment'] = frequent_with_age.index.map(enrollment_ages)

# Define age bins at enrollment (including 10+ for 6th facet)
age_bins_enroll = [0, 1, 2, 3, 5, 10, 100]
age_labels_enroll = ['0-1', '1-2', '2-3', '3-5', '5-10', '10+']

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
        # Get most infected individual from this age bin
        top_infected = in_age_bin.nlargest(1, 'n_positive').index[0]
        selected_ids.append(top_infected)

print(f"\nPlotting {len(selected_ids)} individuals with detailed view...")

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, participant_id in enumerate(selected_ids):
    person_data = df[df['id'] == participant_id].copy()

    n_positive = (person_data['parasitedensity'] > 0).sum()
    n_visits = len(person_data)
    n_gam = (person_data['gametocytes'] == 'Yes').sum()
    n_lamp = (person_data['LAMP'] == 'Positive').sum()

    # Get enrollment age and gender (same for all rows of this individual)
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
plt.savefig(output_dir / '07_individual_detailed_examples.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '07_individual_detailed_examples.png'}")
plt.close()

print("\n" + "=" * 80)
print("EXAMPLE 2: COMPACT MULTI-INDIVIDUAL OVERVIEW")
print("=" * 80)

# Select individuals for compact view - aim for variety of ages and patterns
# Sample across different age groups and infection levels
df['age_group'] = pd.cut(df['age'], bins=[0, 2, 5, 10, 100], labels=['0-2', '2-5', '5-10', '10+'])

# Get representative individuals from each age group
sample_ids = []
for age_group in ['0-2', '2-5', '5-10', '10+']:
    age_group_data = df[df['age_group'] == age_group]
    ids_in_group = age_group_data['id'].unique()

    # Filter to those with at least 10 visits
    visit_counts = age_group_data.groupby('id').size()
    frequent_ids = visit_counts[visit_counts >= 10].index.tolist()

    if len(frequent_ids) > 0:
        # Sample up to 15 from this age group, prioritizing those with infections
        infection_counts = age_group_data[age_group_data['id'].isin(frequent_ids)].groupby('id').agg({
            'parasitedensity': lambda x: (x > 0).sum()
        })
        infection_counts = infection_counts.sort_values('parasitedensity', ascending=False)

        sample_ids.extend(infection_counts.head(15).index.tolist())

print(f"\nPlotting {len(sample_ids)} individuals in compact view...")

# Filter to selected individuals
subset = df[df['id'].isin(sample_ids)].copy()

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
plot_individual_compact(subset, ax=ax, fig=fig, detailed=False)
ax.set_title('PRISM Individual Trajectories: Multi-Individual Compact View\n' +
             '(Top 15 most infected individuals from each age group: 0-2, 2-5, 5-10, 10+ years at enrollment)',
             fontsize=14, fontweight='bold')

plt.savefig(output_dir / '08_individuals_compact_overview.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '08_individuals_compact_overview.png'}")
plt.close()

print("\n" + "=" * 80)
print("EXAMPLE 3: COMPACT VIEW WITH ANNOTATIONS (DETAILED)")
print("=" * 80)

# Smaller subset with annotations
annotated_ids = selected_ids[:12] if len(selected_ids) >= 12 else selected_ids
subset_annotated = df[df['id'].isin(annotated_ids)].copy()

print(f"\nPlotting {len(annotated_ids)} individuals with detailed annotations...")

fig, ax = plt.subplots(1, 1, figsize=(20, 8))
plot_individual_compact(subset_annotated, ax=ax, fig=fig, detailed=True)
ax.set_title('PRISM Individual Trajectories: Detailed Annotations', fontsize=14, fontweight='bold')

plt.savefig(output_dir / '09_individuals_compact_detailed.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '09_individuals_compact_detailed.png'}")
plt.close()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nCreated three types of individual trajectory plots:")
print("  1. Detailed single-individual panels (simulation style)")
print("  2. Compact multi-individual overview")
print("  3. Compact with detailed annotations")
print(f"\nAll plots saved to: {output_dir}/")
