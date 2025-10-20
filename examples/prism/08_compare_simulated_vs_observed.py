"""
Compare Simulated vs Observed Household Data

Identifies major qualitative differences between simulation and PRISM data
before fine-tuning parameter sweeps. Focus on "wildly different" features:

1. Parasite densities at clinical episodes (fever + treatment)
2. Overall prevalence patterns
3. Age-specific infection rates
4. Gametocyte prevalence
5. Distribution of positive densities
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
print("COMPARING SIMULATED VS OBSERVED DATA")
print("=" * 80)

# Load observed data
print("\nLoading observed PRISM data...")
obs_df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"  {len(obs_df)} observations, {obs_df['id'].nunique()} participants")

# Load simulated data
print("\nLoading simulated data...")
sim_df = pd.read_csv(SIMULATION_OUTPUT_DIR / 'household_visit_output.csv', parse_dates=['date'])

# Calculate age at visit for simulated data (not just age at enrollment)
# Assume enrollment was at first visit for each individual
sim_df['enrollment_date'] = sim_df.groupby('id')['date'].transform('min')
sim_df['days_since_enrollment'] = (sim_df['date'] - sim_df['enrollment_date']).dt.days
sim_df['age'] = sim_df['age_at_enrollment'] + (sim_df['days_since_enrollment'] / 365.25)

print(f"  {len(sim_df)} visits, {sim_df['id'].nunique()} simulated individuals")

# ==============================================================================
# 1. PARASITE DENSITIES AT CLINICAL MALARIA EPISODES (TREATED CASES)
# ==============================================================================
print("\n" + "=" * 80)
print("1. PARASITE DENSITIES AT CLINICAL MALARIA EPISODES (CHILDREN UNDER 5)")
print("=" * 80)
print("\nRestricting to:")
print("  - Children under 5 years (causal malaria fever)")
print("  - Unscheduled visits (symptom-triggered)")
print("  - Treated with antimalarials (clinical malaria diagnosis)")
print("  - Positive microscopy")

# Observed: unscheduled + antimalarial treatment + positive, age <5
obs_clinical = obs_df[
    (obs_df['visittype'] == 'Unscheduled visit') &
    (obs_df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate', case=False, na=False)) &
    (obs_df['parasitedensity'] > 0) &
    (obs_df['age'] < 5)  # age at time of visit, not age at enrollment
].copy()
print(f"\nObserved clinical episodes (age <5): {len(obs_clinical)}")
if len(obs_clinical) > 0:
    print(f"  Median density: {obs_clinical['parasitedensity'].median():.0f} parasites/μL")
    print(f"  Geometric mean density: {np.exp(np.log(obs_clinical['parasitedensity']).mean()):.0f} parasites/μL")
    print(f"  Mean density: {obs_clinical['parasitedensity'].mean():.0f} parasites/μL")
    print(f"  25th-75th percentile: {obs_clinical['parasitedensity'].quantile(0.25):.0f}-{obs_clinical['parasitedensity'].quantile(0.75):.0f}")
    print(f"  Max density: {obs_clinical['parasitedensity'].max():.0f} parasites/μL")
    print(f"  Range: {obs_clinical['parasitedensity'].min():.0f}-{obs_clinical['parasitedensity'].max():.0f}")

# Simulated: unscheduled + antimalarial treatment + positive, age <5
sim_clinical = sim_df[
    (sim_df['visittype'] == 'Unscheduled visit') &
    (sim_df['antimalarial'].str.contains('Artemether|Quinine|Artesunate', case=False, na=False)) &
    (sim_df['parasitedensity'] > 0) &
    (sim_df['age'] < 5)  # age at time of visit, matching observed data
].copy()
print(f"\nSimulated clinical episodes (age <5): {len(sim_clinical)}")
if len(sim_clinical) > 0:
    print(f"  Median density: {sim_clinical['parasitedensity'].median():.0f} parasites/μL")
    print(f"  Geometric mean density: {np.exp(np.log(sim_clinical['parasitedensity']).mean()):.0f} parasites/μL")
    print(f"  Mean density: {sim_clinical['parasitedensity'].mean():.0f} parasites/μL")
    print(f"  25th-75th percentile: {sim_clinical['parasitedensity'].quantile(0.25):.0f}-{sim_clinical['parasitedensity'].quantile(0.75):.0f}")
    print(f"  Max density: {sim_clinical['parasitedensity'].max():.0f} parasites/μL")
    print(f"  Range: {sim_clinical['parasitedensity'].min():.0f}-{sim_clinical['parasitedensity'].max():.0f}")

# Compare for 5-10 and adults too (for reference) - also with antimalarial filter
obs_5_10_clinical = obs_df[
    (obs_df['visittype'] == 'Unscheduled visit') &
    (obs_df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate', case=False, na=False)) &
    (obs_df['parasitedensity'] > 0) &
    (obs_df['age'] >= 5) &
    (obs_df['age'] <= 10)
]
print(f"\n[Reference] Observed clinical episodes (age 5-10): {len(obs_5_10_clinical)}")
if len(obs_5_10_clinical) > 0:
    print(f"  Median density: {obs_5_10_clinical['parasitedensity'].median():.0f} parasites/μL")

obs_adult_clinical = obs_df[
    (obs_df['visittype'] == 'Unscheduled visit') &
    (obs_df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate', case=False, na=False)) &
    (obs_df['parasitedensity'] > 0) &
    (obs_df['age'] > 10)
]
print(f"\n[Reference] Observed clinical episodes (age >10): {len(obs_adult_clinical)}")
if len(obs_adult_clinical) > 0:
    print(f"  Median density: {obs_adult_clinical['parasitedensity'].median():.0f} parasites/μL")
    print(f"  (Note: Adults may have incidental chronic infections)")

# ==============================================================================
# 2. OVERALL PREVALENCE
# ==============================================================================
print("\n" + "=" * 80)
print("2. OVERALL PREVALENCE")
print("=" * 80)

obs_prev = 100 * (obs_df['parasitedensity'] > 0).sum() / len(obs_df[obs_df['parasitedensity'].notna()])
sim_prev = 100 * (sim_df['parasitedensity'] > 0).sum() / len(sim_df)

print(f"\nObserved prevalence: {obs_prev:.1f}%")
print(f"Simulated prevalence: {sim_prev:.1f}%")
print(f"Ratio (sim/obs): {sim_prev/obs_prev:.2f}x")

# ==============================================================================
# 3. AGE-SPECIFIC PATTERNS
# ==============================================================================
print("\n" + "=" * 80)
print("3. AGE-SPECIFIC PREVALENCE")
print("=" * 80)

# Observed - finer age bins for under-5s
obs_df['age_bin'] = pd.cut(obs_df['age'], bins=[0, 1, 2, 5, 100], labels=['<1', '1-2', '2-5', '5+'])
obs_age_prev = obs_df.groupby('age_bin').apply(
    lambda x: 100 * (x['parasitedensity'] > 0).sum() / len(x[x['parasitedensity'].notna()])
)

# Simulated - finer age bins for under-5s (using age at visit, not enrollment)
sim_df['age_bin'] = pd.cut(sim_df['age'], bins=[0, 1, 2, 5, 100], labels=['<1', '1-2', '2-5', '5+'])
sim_age_prev = sim_df.groupby('age_bin').apply(
    lambda x: 100 * (x['parasitedensity'] > 0).sum() / len(x)
)

print("\nAge    Observed    Simulated")
print("-" * 35)
for age_bin in ['<1', '1-2', '2-5', '5+']:
    if age_bin in obs_age_prev.index and age_bin in sim_age_prev.index:
        print(f"{age_bin:6s} {obs_age_prev[age_bin]:6.1f}%    {sim_age_prev[age_bin]:6.1f}%")

# ==============================================================================
# 4. GAMETOCYTE PREVALENCE
# ==============================================================================
print("\n" + "=" * 80)
print("4. GAMETOCYTE PREVALENCE (among positive samples)")
print("=" * 80)

obs_pos = obs_df[obs_df['parasitedensity'] > 0]
obs_gam_prev = 100 * (obs_pos['gametocytes'] == 'Yes').sum() / len(obs_pos[obs_pos['gametocytes'].notna()])

sim_pos = sim_df[sim_df['parasitedensity'] > 0]
sim_gam_prev = 100 * (sim_pos['gametocytes'] == 'Yes').sum() / len(sim_pos)

print(f"\nObserved gametocyte prevalence: {obs_gam_prev:.1f}%")
print(f"Simulated gametocyte prevalence: {sim_gam_prev:.1f}%")

# ==============================================================================
# 5. VISUALIZATION: SIDE-BY-SIDE COMPARISONS
# ==============================================================================
print("\n" + "=" * 80)
print("5. CREATING COMPARISON FIGURES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Clinical episode densities (normalized counts for comparison)
ax = axes[0, 0]
if len(obs_clinical) > 0 and len(sim_clinical) > 0:
    bins = np.logspace(1, 6, 30)

    # Use weights to normalize to percentage instead of density=True
    # This makes the histogram interpretable: height = % of observations in that bin
    obs_weights = np.ones(len(obs_clinical)) / len(obs_clinical) * 100
    sim_weights = np.ones(len(sim_clinical)) / len(sim_clinical) * 100

    ax.hist(obs_clinical['parasitedensity'], bins=bins, alpha=0.5, label='Observed',
            edgecolor='black', weights=obs_weights)
    ax.hist(sim_clinical['parasitedensity'], bins=bins, alpha=0.5, label='Simulated',
            edgecolor='black', weights=sim_weights)
    ax.set_xscale('log')
    ax.set_xlabel('Parasite Density (parasites/μL)')
    ax.set_ylabel('Percentage of Episodes (%)')
    ax.set_title('Parasite Densities at Clinical Malaria Episodes\n(Children <5 years, Unscheduled + Antimalarial Treatment)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add median lines
    ax.axvline(obs_clinical['parasitedensity'].median(), color='blue', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Obs median: {obs_clinical["parasitedensity"].median():.0f}')
    ax.axvline(sim_clinical['parasitedensity'].median(), color='orange', linestyle='--',
               linewidth=2, alpha=0.7, label=f'Sim median: {sim_clinical["parasitedensity"].median():.0f}')
    ax.legend()

# Panel 2: All positive densities (scheduled + unscheduled)
ax = axes[0, 1]
obs_all_pos = obs_df[obs_df['parasitedensity'] > 0]['parasitedensity']
sim_all_pos = sim_df[sim_df['parasitedensity'] > 0]['parasitedensity']

if len(obs_all_pos) > 0 and len(sim_all_pos) > 0:
    bins = np.logspace(1, 6, 30)
    ax.hist(obs_all_pos, bins=bins, alpha=0.5, label='Observed',
            edgecolor='black', density=True)
    ax.hist(sim_all_pos, bins=bins, alpha=0.5, label='Simulated',
            edgecolor='black', density=True)
    ax.set_xscale('log')
    ax.set_xlabel('Parasite Density (parasites/μL)')
    ax.set_ylabel('Density')
    ax.set_title('All Positive Parasite Densities\n(All Visit Types)')
    ax.legend()
    ax.grid(alpha=0.3)

# Panel 3: Individual clinical episode densities by age (strip plot with jitter)
ax = axes[1, 0]

# Prepare age bins for clinical episodes
obs_clinical['age_bin'] = pd.cut(obs_clinical['age'], bins=[0, 1, 2, 5], labels=['<1', '1-2', '2-5'])
sim_clinical['age_bin'] = pd.cut(sim_clinical['age'], bins=[0, 1, 2, 5], labels=['<1', '1-2', '2-5'])

age_bins = ['<1', '1-2', '2-5']
age_positions = {'<1': 0, '1-2': 1, '2-5': 2}

# Plot observed data with jitter
np.random.seed(42)  # For reproducible jitter
for age_bin in age_bins:
    obs_data = obs_clinical[obs_clinical['age_bin'] == age_bin]['parasitedensity']
    if len(obs_data) > 0:
        x_pos = age_positions[age_bin] - 0.2
        jitter = np.random.normal(0, 0.06, len(obs_data))  # Small jitter
        ax.scatter(x_pos + jitter, obs_data, alpha=0.3, s=20, color='blue', label='Observed' if age_bin == '<1' else '')

# Plot simulated data with jitter
for age_bin in age_bins:
    sim_data = sim_clinical[sim_clinical['age_bin'] == age_bin]['parasitedensity']
    if len(sim_data) > 0:
        x_pos = age_positions[age_bin] + 0.2
        jitter = np.random.normal(0, 0.06, len(sim_data))  # Small jitter
        ax.scatter(x_pos + jitter, sim_data, alpha=0.5, s=30, color='orange', label='Simulated' if age_bin == '<1' else '')

# Add median lines for each group
for age_bin in age_bins:
    obs_data = obs_clinical[obs_clinical['age_bin'] == age_bin]['parasitedensity']
    sim_data = sim_clinical[sim_clinical['age_bin'] == age_bin]['parasitedensity']
    x_pos = age_positions[age_bin]

    if len(obs_data) > 0:
        ax.plot([x_pos - 0.3, x_pos - 0.1], [obs_data.median(), obs_data.median()],
                'b-', linewidth=3, alpha=0.8)
    if len(sim_data) > 0:
        ax.plot([x_pos + 0.1, x_pos + 0.3], [sim_data.median(), sim_data.median()],
                color='orange', linewidth=3, alpha=0.8)

ax.set_yscale('log')
ax.set_xlabel('Age Group (years)')
ax.set_ylabel('Parasite Density (parasites/μL)')
ax.set_title('Clinical Episode Densities by Age\n(Individual episodes, lines = medians)')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(age_bins)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([10, 1e6])

# Panel 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
SUMMARY COMPARISON

Overall Prevalence:
  Observed:  {obs_prev:.1f}%
  Simulated: {sim_prev:.1f}%
  Ratio:     {sim_prev/obs_prev:.2f}x

Clinical Episode Densities:
  Observed median:  {obs_clinical['parasitedensity'].median():.0f} /μL
  Simulated median: {sim_clinical['parasitedensity'].median():.0f} /μL
  Ratio:            {sim_clinical['parasitedensity'].median()/obs_clinical['parasitedensity'].median():.2f}x

Gametocyte Prevalence (among +):
  Observed:  {obs_gam_prev:.1f}%
  Simulated: {sim_gam_prev:.1f}%

KEY ISSUES TO ADDRESS:
"""

# Check for major discrepancies
issues = []
if sim_clinical['parasitedensity'].median() < 0.5 * obs_clinical['parasitedensity'].median():
    issues.append("⚠ Clinical episode densities too LOW")
if sim_prev < 0.5 * obs_prev or sim_prev > 2.0 * obs_prev:
    issues.append("⚠ Overall prevalence mismatch")
if abs(sim_gam_prev - obs_gam_prev) > 10:
    issues.append("⚠ Gametocyte prevalence very different")

if len(issues) == 0:
    summary_text += "\n✓ No major discrepancies detected"
else:
    for issue in issues:
        summary_text += f"\n{issue}"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'simulated_vs_observed_comparison.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"\n  Saved: {fig_file}")
plt.close()

# ==============================================================================
# 6. DETAILED CLINICAL EPISODE ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("6. DETAILED CLINICAL EPISODE ANALYSIS")
print("=" * 80)

print("\nObserved clinical episodes by density range:")
density_bins = [0, 100, 1000, 10000, 100000, 1000000]
density_labels = ['<100', '100-1K', '1K-10K', '10K-100K', '>100K']
obs_clinical['density_bin'] = pd.cut(obs_clinical['parasitedensity'], bins=density_bins, labels=density_labels)
print(obs_clinical['density_bin'].value_counts().sort_index())

print("\nSimulated clinical episodes by density range:")
sim_clinical['density_bin'] = pd.cut(sim_clinical['parasitedensity'], bins=density_bins, labels=density_labels)
print(sim_clinical['density_bin'].value_counts().sort_index())

# Calculate what fraction of clinical episodes are in high-density range
obs_high_density = (obs_clinical['parasitedensity'] > 10000).sum()
sim_high_density = (sim_clinical['parasitedensity'] > 10000).sum()

print(f"\nFraction with density >10,000 /μL:")
print(f"  Observed:  {100*obs_high_density/len(obs_clinical):.1f}%")
print(f"  Simulated: {100*sim_high_density/len(sim_clinical):.1f}%")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
print(f"\nFigures saved to: {FIGURES_OUTPUT_DIR}/")
