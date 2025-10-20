"""
PRISM Data Exploratory Visualizations

This script creates exploratory plots to understand key patterns in the PRISM data:
- Parasite density distributions
- Infection prevalence over time
- Age-stratified infection patterns
- Detection method comparisons (microscopy vs LAMP)
- Visit patterns and fever-based surveillance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Site selection
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'

# Load cleaned data
print(f"Loading cleaned PRISM data for {SITE.upper()} site...")
df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"Loaded {len(df)} observations for {df['id'].nunique()} participants")

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("GENERATING EXPLORATORY PLOTS")
print("=" * 80)

# ============================================================================
# 1. PARASITE DENSITY DISTRIBUTION
# ============================================================================
print("\n1. Parasite density distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# All positive densities (log scale)
positive = df[df['parasitedensity'] > 0]['parasitedensity']
axes[0, 0].hist(np.log10(positive), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Log10 Parasite Density (parasites/µL)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Parasite Density Distribution\n(n={len(positive)} positive samples)')
axes[0, 0].axvline(np.log10(positive.median()), color='red', linestyle='--', label=f'Median: {positive.median():.0f}')
axes[0, 0].legend()

# By visit type
visit_types = ['Scheduled visit', 'Unscheduled visit']
for vtype in visit_types:
    pos_visit = df[(df['parasitedensity'] > 0) & (df['visittype'] == vtype)]['parasitedensity']
    if len(pos_visit) > 0:
        axes[0, 1].hist(np.log10(pos_visit), bins=30, alpha=0.5, label=vtype, edgecolor='black')
axes[0, 1].set_xlabel('Log10 Parasite Density (parasites/µL)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Parasite Density by Visit Type')
axes[0, 1].legend()

# Prevalence by visit type
prev_data = []
for vtype in df['visittype'].dropna().unique():
    subset = df[df['visittype'] == vtype]
    prevalence = 100 * (subset['parasitedensity'] > 0).sum() / len(subset)
    prev_data.append({'Visit Type': vtype, 'Prevalence (%)': prevalence, 'N': len(subset)})
prev_df = pd.DataFrame(prev_data)
axes[1, 0].bar(range(len(prev_df)), prev_df['Prevalence (%)'], tick_label=prev_df['Visit Type'], alpha=0.7)
axes[1, 0].set_ylabel('Prevalence (%)')
axes[1, 0].set_title('Infection Prevalence by Visit Type')
axes[1, 0].set_xticklabels(prev_df['Visit Type'], rotation=45, ha='right')
for i, row in prev_df.iterrows():
    axes[1, 0].text(i, row['Prevalence (%)'] + 1, f"n={row['N']}", ha='center', fontsize=8)

# Fever vs non-fever
fever_data = []
for fever_status in ['Yes', 'No']:
    subset = df[df['fever'] == fever_status]
    prevalence = 100 * (subset['parasitedensity'] > 0).sum() / len(subset)
    fever_data.append({'Fever': fever_status, 'Prevalence (%)': prevalence, 'N': len(subset)})
fever_df = pd.DataFrame(fever_data)
axes[1, 1].bar(fever_df['Fever'], fever_df['Prevalence (%)'], alpha=0.7, color=['red', 'blue'])
axes[1, 1].set_ylabel('Prevalence (%)')
axes[1, 1].set_title('Infection Prevalence by Fever Status')
for i, row in fever_df.iterrows():
    axes[1, 1].text(i, row['Prevalence (%)'] + 1, f"n={row['N']}\n{row['Prevalence (%)']:.1f}%", ha='center')

plt.tight_layout()
plt.savefig(output_dir / '01_parasite_density_overview.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '01_parasite_density_overview.png'}")
plt.close()

# ============================================================================
# 2. AGE-STRATIFIED PREVALENCE
# ============================================================================
print("\n2. Age-stratified prevalence...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create age bins
df['age_bin'] = pd.cut(df['age'], bins=[0, 1, 2, 3, 5, 10, 100], labels=['0-1', '1-2', '2-3', '3-5', '5-10', '10+'])

# Prevalence by age
age_prev = []
for age_bin in df['age_bin'].cat.categories:
    subset = df[df['age_bin'] == age_bin]
    if len(subset) > 0:
        prevalence = 100 * (subset['parasitedensity'] > 0).sum() / len(subset)
        age_prev.append({'Age': age_bin, 'Prevalence (%)': prevalence, 'N': len(subset)})
age_prev_df = pd.DataFrame(age_prev)
axes[0].plot(age_prev_df['Age'], age_prev_df['Prevalence (%)'], marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Age at Event (years)')
axes[0].set_ylabel('Infection Prevalence (%)')
axes[0].set_title('Age-Specific Infection Prevalence')
axes[0].grid(True, alpha=0.3)

# Geometric mean parasite density by age (positive samples only)
age_density = []
for age_bin in df['age_bin'].cat.categories:
    subset = df[(df['age_bin'] == age_bin) & (df['parasitedensity'] > 0)]
    if len(subset) > 0:
        # Geometric mean = 10^(mean of log10 values)
        geom_mean_density = 10 ** np.log10(subset['parasitedensity']).mean()
        age_density.append({'Age': age_bin, 'Geometric Mean Density': geom_mean_density, 'N': len(subset)})
age_density_df = pd.DataFrame(age_density)
axes[1].plot(age_density_df['Age'], age_density_df['Geometric Mean Density'], marker='o', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('Age at Event (years)')
axes[1].set_ylabel('Geometric Mean Parasite Density (parasites/µL)')
axes[1].set_title('Geometric Mean Parasite Density by Age\n(Positive Samples Only)')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '02_age_stratified_patterns.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '02_age_stratified_patterns.png'}")
plt.close()

# ============================================================================
# 3. TEMPORAL PATTERNS
# ============================================================================
print("\n3. Temporal patterns...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Monthly prevalence over time
df['year_month'] = df['date'].dt.to_period('M')
monthly_prev = []
for ym in df['year_month'].dropna().unique():
    subset = df[df['year_month'] == ym]
    prevalence = 100 * (subset['parasitedensity'] > 0).sum() / len(subset)
    monthly_prev.append({'Date': ym.to_timestamp(), 'Prevalence (%)': prevalence, 'N': len(subset)})
monthly_prev_df = pd.DataFrame(monthly_prev).sort_values('Date')

axes[0].plot(monthly_prev_df['Date'], monthly_prev_df['Prevalence (%)'], linewidth=2)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Infection Prevalence (%)')
axes[0].set_title('Monthly Infection Prevalence Over Time')
axes[0].grid(True, alpha=0.3)

# Seasonal pattern (aggregate all years)
df['month'] = df['date'].dt.month
seasonal_prev = []
for month in range(1, 13):
    subset = df[df['month'] == month]
    prevalence = 100 * (subset['parasitedensity'] > 0).sum() / len(subset)
    seasonal_prev.append({'Month': month, 'Prevalence (%)': prevalence, 'N': len(subset)})
seasonal_prev_df = pd.DataFrame(seasonal_prev)

axes[1].plot(seasonal_prev_df['Month'], seasonal_prev_df['Prevalence (%)'], marker='o', linewidth=2, markersize=8)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Infection Prevalence (%)')
axes[1].set_title('Seasonal Pattern (All Years Combined)')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '03_temporal_patterns.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '03_temporal_patterns.png'}")
plt.close()

# ============================================================================
# 4. DETECTION METHOD COMPARISON (Microscopy vs LAMP)
# ============================================================================
print("\n4. Detection method comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Note: LAMP is only performed on microscopy-negative samples (testing strategy)
# Microscopy positive samples don't get LAMP testing

# All samples with microscopy result
micro_tested = df[df['parasitedensity'].notna()].copy()
micro_pos = (micro_tested['parasitedensity'] > 0).sum()
micro_neg = (micro_tested['parasitedensity'] == 0).sum()

# Among microscopy-negative, how many got LAMP?
micro_neg_samples = df[df['parasitedensity'] == 0].copy()
lamp_tested_after_neg = micro_neg_samples[micro_neg_samples['LAMP'].isin(['Positive', 'Negative'])]
lamp_pos_after_neg = (lamp_tested_after_neg['LAMP'] == 'Positive').sum()
lamp_neg_after_neg = (lamp_tested_after_neg['LAMP'] == 'Negative').sum()
lamp_not_tested = micro_neg - len(lamp_tested_after_neg)

print(f"\nTesting strategy summary:")
print(f"  Microscopy positive (no LAMP): {micro_pos}")
print(f"  Microscopy negative, LAMP positive: {lamp_pos_after_neg}")
print(f"  Microscopy negative, LAMP negative: {lamp_neg_after_neg}")
print(f"  Microscopy negative, LAMP not done: {lamp_not_tested}")

# Bar chart showing detection cascade
categories = ['Micro+\n(LAMP not done)', 'Micro-/LAMP+\n(submicroscopic)', 'Micro-/LAMP-', 'Micro-/LAMP\nnot done']
counts = [micro_pos, lamp_pos_after_neg, lamp_neg_after_neg, lamp_not_tested]
colors = ['#FF6B6B', '#FFA07A', '#4ECDC4', '#95E1D3']

axes[0].bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Number of Samples')
axes[0].set_title(f'Testing Strategy: Microscopy → LAMP (if negative)\n(n={len(micro_tested)} samples)')
for i, (cat, count) in enumerate(zip(categories, counts)):
    pct = 100 * count / len(micro_tested)
    axes[0].text(i, count + 500, f'{count}\n({pct:.1f}%)', ha='center', fontsize=9)

# Prevalence by detection method
total_positive = micro_pos + lamp_pos_after_neg
microscopic_prev = 100 * micro_pos / len(micro_tested)
total_prev = 100 * total_positive / len(micro_tested)
submicroscopic_fraction = 100 * lamp_pos_after_neg / total_positive if total_positive > 0 else 0

detection_data = pd.DataFrame({
    'Method': ['Microscopy\nonly', 'Microscopy\n+ LAMP'],
    'Prevalence (%)': [microscopic_prev, total_prev]
})

axes[1].bar(detection_data['Method'], detection_data['Prevalence (%)'],
            color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5, alpha=0.7)
axes[1].set_ylabel('Infection Prevalence (%)')
axes[1].set_title(f'Prevalence by Detection Method\n({submicroscopic_fraction:.1f}% of infections are submicroscopic)')
for i, row in detection_data.iterrows():
    axes[1].text(i, row['Prevalence (%)'] + 0.5, f"{row['Prevalence (%)']:.1f}%", ha='center', fontsize=11)
axes[1].set_ylim([0, detection_data['Prevalence (%)'].max() * 1.15])

plt.tight_layout()
plt.savefig(output_dir / '04_detection_methods.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '04_detection_methods.png'}")
plt.close()

# ============================================================================
# 5. INDIVIDUAL TRAJECTORIES (Sample)
# ============================================================================
print("\n5. Individual trajectories (sample)...")

# Select 9 individuals with interesting patterns
# (mix of infection types: persistent, episodic, negative)
individual_counts = df.groupby('id').agg({
    'parasitedensity': lambda x: (x > 0).sum(),
    'date': 'count'
}).rename(columns={'parasitedensity': 'n_infections', 'date': 'n_visits'})

# Get individuals with at least 20 visits
frequent = individual_counts[individual_counts['n_visits'] >= 20]

# Select some with high, medium, and low infection counts
high_infection = frequent.nlargest(3, 'n_infections').index.tolist()
low_infection = frequent.nsmallest(3, 'n_infections').index.tolist()
medium_infection = frequent.sort_values('n_infections').iloc[len(frequent)//2:len(frequent)//2+3].index.tolist()

sample_ids = high_infection[:3] + medium_infection[:3] + low_infection[:3]

fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
axes = axes.flatten()

for idx, participant_id in enumerate(sample_ids):
    ax = axes[idx]
    person_data = df[df['id'] == participant_id].sort_values('date')

    # Plot parasite density
    dates = person_data['date']
    densities = person_data['parasitedensity'].fillna(0)

    # Use log scale for positive densities
    plot_densities = densities.copy()
    plot_densities[plot_densities == 0] = 0.1  # Small value for log scale

    ax.scatter(dates, plot_densities, c=densities > 0, cmap='RdYlGn_r',
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Mark fever episodes
    fever_dates = person_data[person_data['fever'] == 'Yes']['date']
    for fever_date in fever_dates:
        ax.axvline(fever_date, color='red', alpha=0.2, linewidth=1)

    ax.set_yscale('log')
    ax.set_ylim([0.05, 1e6])
    ax.set_ylabel('Parasite Density\n(parasites/µL)')
    ax.set_title(f'ID {participant_id}\n({(densities > 0).sum()}/{len(densities)} positive)', fontsize=10)
    ax.grid(True, alpha=0.3)

    if idx >= 6:
        ax.set_xlabel('Date')

plt.tight_layout()
plt.savefig(output_dir / '05_individual_trajectories.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {output_dir / '05_individual_trajectories.png'}")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nOverall infection prevalence:")
overall_prev = 100 * (df['parasitedensity'] > 0).sum() / len(df[df['parasitedensity'].notna()])
print(f"  {overall_prev:.2f}% ({(df['parasitedensity'] > 0).sum()}/{len(df[df['parasitedensity'].notna()])})")

print("\nGametocyte prevalence (among positive samples):")
pos_samples = df[df['parasitedensity'] > 0]
gam_prev = 100 * (pos_samples['gametocytes'] == 'Yes').sum() / len(pos_samples[pos_samples['gametocytes'].notna()])
print(f"  {gam_prev:.2f}% ({(pos_samples['gametocytes'] == 'Yes').sum()}/{len(pos_samples[pos_samples['gametocytes'].notna()])})")

print("\nDetection method summary:")
print(f"  Microscopy positive: {micro_pos}")
print(f"  Microscopy negative, LAMP positive (submicroscopic): {lamp_pos_after_neg}")
print(f"  Microscopy negative, LAMP negative: {lamp_neg_after_neg}")
print(f"  Microscopy negative, LAMP not done: {lamp_not_tested}")
print(f"  Total infections detected: {total_positive} ({total_prev:.2f}%)")
print(f"  Submicroscopic fraction: {submicroscopic_fraction:.1f}% of all infections")

print("\n" + "=" * 80)
print("EXPLORATORY PLOTS COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {output_dir}/")
