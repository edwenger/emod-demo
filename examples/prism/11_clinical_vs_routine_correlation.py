"""
Correlation Analysis: Clinical Episodes vs Routine Positives

Examines whether individuals with more clinical episodes also have more
positive routine visits, separately for observed and simulated data by age group.

Key question: Does the max_infections constraint in simulations create an
unrealistic inverse correlation (chronic carriage blocks new incident episodes)?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Configuration
SITE = 'nagongera'
SIMULATION_OUTPUT_DIR = Path('simulation_output')
FIGURES_OUTPUT_DIR = Path('figures')
FIGURES_OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("CLINICAL VS ROUTINE CORRELATION ANALYSIS")
print("=" * 80)

# Load observed data
print("\nLoading observed PRISM data...")
obs_df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])

# Filter to pre-IRS period
print("  Filtering to pre-IRS period (before 2015-01-01)...")
obs_df = obs_df[obs_df['date'] < '2015-01-01'].copy()
print(f"  {len(obs_df)} observations in pre-IRS period")

# Load simulated data
print("\nLoading simulated data...")
sim_df = pd.read_csv(SIMULATION_OUTPUT_DIR / 'household_visit_output.csv', parse_dates=['date'])
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

# ==============================================================================
# Calculate per-individual counts by age group
# ==============================================================================

def calculate_individual_metrics(df, source_label):
    """Calculate clinical episodes and routine positives per individual per age group"""
    results = []

    # Filter to clinical episodes
    clinical = df[
        (df['visittype'] == 'Unscheduled visit') &
        (df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate', case=False, na=False)) &
        (df['parasitedensity'] > 0)
    ]

    # Filter to routine visits
    routine = df[df['visittype'] == 'Scheduled visit']

    # For each individual and age group
    for ind_id in df['id'].unique():
        for age_group in ['<1', '1-2', '2-5', '5-10', '10+']:
            # Get data for this individual in this age group
            ind_age_data = df[(df['id'] == ind_id) & (df['age_group'] == age_group)]

            if len(ind_age_data) > 0:
                # Count clinical episodes
                n_clinical = len(clinical[(clinical['id'] == ind_id) &
                                          (clinical['age_group'] == age_group)])

                # Count routine positives
                ind_routine = routine[(routine['id'] == ind_id) &
                                     (routine['age_group'] == age_group)]
                n_routine_visits = len(ind_routine[ind_routine['parasitedensity'].notna()])
                n_routine_positive = (ind_routine['parasitedensity'] > 0).sum()

                # Calculate person-time
                person_years = (ind_age_data['date'].max() - ind_age_data['date'].min()).days / 365.25

                if person_years > 0.1:  # Require at least ~36 days in age group
                    results.append({
                        'id': ind_id,
                        'age_group': age_group,
                        'source': source_label,
                        'n_clinical': n_clinical,
                        'n_routine_visits': n_routine_visits,
                        'n_routine_positive': n_routine_positive,
                        'person_years': person_years,
                        'clinical_rate': n_clinical / person_years,
                        'routine_positivity': n_routine_positive / n_routine_visits if n_routine_visits > 0 else 0,
                    })

    return pd.DataFrame(results)

print("\nCalculating individual-level metrics...")
obs_metrics = calculate_individual_metrics(obs_df, 'Observed')
sim_metrics = calculate_individual_metrics(sim_df, 'Simulated')

print(f"  Observed: {len(obs_metrics)} individual-age-group observations")
print(f"  Simulated: {len(sim_metrics)} individual-age-group observations")

# Combine for plotting
all_metrics = pd.concat([obs_metrics, sim_metrics], ignore_index=True)

# ==============================================================================
# Create correlation plots by age group
# ==============================================================================
print("\nCreating correlation plots...")

age_groups = ['<1', '1-2', '2-5', '5-10', '10+']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, age_group in enumerate(age_groups):
    ax = axes[idx]

    obs_age = obs_metrics[obs_metrics['age_group'] == age_group]
    sim_age = sim_metrics[sim_metrics['age_group'] == age_group]

    # Scatter plots
    if len(obs_age) > 0:
        ax.scatter(obs_age['n_routine_positive'], obs_age['n_clinical'],
                  alpha=0.5, s=50, color='blue', label='Observed', edgecolors='black')

        # Calculate correlation
        if len(obs_age) > 2 and obs_age['n_routine_positive'].std() > 0 and obs_age['n_clinical'].std() > 0:
            r_obs, p_obs = pearsonr(obs_age['n_routine_positive'], obs_age['n_clinical'])
            rho_obs, p_rho_obs = spearmanr(obs_age['n_routine_positive'], obs_age['n_clinical'])
        else:
            r_obs, p_obs = np.nan, np.nan
            rho_obs, p_rho_obs = np.nan, np.nan
    else:
        r_obs, p_obs = np.nan, np.nan
        rho_obs, p_rho_obs = np.nan, np.nan

    if len(sim_age) > 0:
        ax.scatter(sim_age['n_routine_positive'], sim_age['n_clinical'],
                  alpha=0.5, s=50, color='orange', label='Simulated', edgecolors='black', marker='s')

        # Calculate correlation
        if len(sim_age) > 2 and sim_age['n_routine_positive'].std() > 0 and sim_age['n_clinical'].std() > 0:
            r_sim, p_sim = pearsonr(sim_age['n_routine_positive'], sim_age['n_clinical'])
            rho_sim, p_rho_sim = spearmanr(sim_age['n_routine_positive'], sim_age['n_clinical'])
        else:
            r_sim, p_sim = np.nan, np.nan
            rho_sim, p_rho_sim = np.nan, np.nan
    else:
        r_sim, p_sim = np.nan, np.nan
        rho_sim, p_rho_sim = np.nan, np.nan

    ax.set_xlabel('Number of Routine Positive Visits', fontsize=11)
    ax.set_ylabel('Number of Clinical Episodes', fontsize=11)
    ax.set_title(f'Age {age_group} (Obs n={len(obs_age)}, Sim n={len(sim_age)})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Add correlation text
    text_y = 0.95
    if not np.isnan(r_obs):
        ax.text(0.05, text_y, f'Obs: r={r_obs:.2f} (p={p_obs:.3f})',
                transform=ax.transAxes, fontsize=9, color='blue',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        text_y -= 0.15

    if not np.isnan(r_sim):
        ax.text(0.05, text_y, f'Sim: r={r_sim:.2f} (p={p_sim:.3f})',
                transform=ax.transAxes, fontsize=9, color='orange',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove unused subplot
axes[-1].axis('off')

plt.tight_layout()
fig_file = FIGURES_OUTPUT_DIR / 'clinical_vs_routine_correlation.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {fig_file}")
plt.close()

# ==============================================================================
# Summary statistics
# ==============================================================================
print("\n" + "=" * 80)
print("CORRELATION SUMMARY")
print("=" * 80)

print("\nPearson Correlation (Clinical Episodes vs Routine Positives):")
print(f"{'Age':>8s}  {'Obs r':>8s}  {'Obs p':>8s}  {'Sim r':>8s}  {'Sim p':>8s}  {'Diff':>8s}")
print("-" * 80)

for age_group in age_groups:
    obs_age = obs_metrics[obs_metrics['age_group'] == age_group]
    sim_age = sim_metrics[sim_metrics['age_group'] == age_group]

    if len(obs_age) > 2 and obs_age['n_routine_positive'].std() > 0 and obs_age['n_clinical'].std() > 0:
        r_obs, p_obs = pearsonr(obs_age['n_routine_positive'], obs_age['n_clinical'])
    else:
        r_obs, p_obs = np.nan, np.nan

    if len(sim_age) > 2 and sim_age['n_routine_positive'].std() > 0 and sim_age['n_clinical'].std() > 0:
        r_sim, p_sim = pearsonr(sim_age['n_routine_positive'], sim_age['n_clinical'])
    else:
        r_sim, p_sim = np.nan, np.nan

    diff = r_sim - r_obs if not (np.isnan(r_obs) or np.isnan(r_sim)) else np.nan

    print(f"{age_group:>8s}  {r_obs:8.3f}  {p_obs:8.3f}  {r_sim:8.3f}  {p_sim:8.3f}  {diff:8.3f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("\nKey patterns to look for:")
print("1. Positive correlation (r > 0): More clinical episodes → more routine positives")
print("   - Could indicate higher exposure or lower treatment success")
print("2. Negative correlation (r < 0): More clinical episodes → fewer routine positives")
print("   - Could indicate max_infections constraint (chronic blocks new incidents)")
print("3. No correlation (r ≈ 0): Independent mechanisms")
print("   - Suggests different individuals have different infection patterns")
print("\nIf simulated r << observed r (more negative):")
print("  → Max_infections constraint may be artifactually blocking new infections")
print("  → Chronic carriage prevents incident episodes in model but not in reality")
