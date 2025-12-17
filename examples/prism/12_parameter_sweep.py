"""
Parameter Sweep: EIR × Treatment Coverage Grid Search

Systematically explores the parameter space of monthly EIR and treatment coverage
to identify combinations that best match observed PRISM data for:
1. Clinical episode incidence (by age group)
2. Routine visit prevalence (by age group)
3. Clinical episode density distributions

This addresses the "knife-edge" calibration problem identified in the README where
different parameter combinations produce different trade-offs between matching
clinical incidence vs routine prevalence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from seasonal_challenge import (
    monthly_eir_challenge,
    surface_area_biting_function,
    ComplexTreatment,
    month_index_from_day,
)
from emodlib.malaria import IntrahostComponent, create_config

# Configuration
SITE = 'nagongera'
DATA_DIR = Path('data')
OUTPUT_DIR = Path('parameter_sweep_output')

# Parameter sweep ranges
EIR_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]  # Reduced range for faster testing
TREATMENT_COVERAGE_VALUES = [0.60, 0.70, 0.75, 0.80, 0.85]

# Simulation configuration
N_INDIVIDUALS = 15  # Reduced for faster sweep
SIMULATION_YEARS = 4.0  # Focus on pre-IRS period
BURNIN_YEARS = 10  # Age-appropriate immunity
TREATMENT_COVERAGE_BURNIN = 0.50

# Seasonality pattern (same as main simulation)
SEASONALITY_AMPLIFICATION = 3.0
BASE_MONTHLY_MULTIPLIER = np.array([
    0.40, 0.30, 0.25, 0.50, 1.50, 2.00,  # Jan-Jun (peak May-June)
    1.80, 1.40, 1.00, 0.80, 0.60, 0.45   # Jul-Dec
])
MONTHLY_MULTIPLIER = BASE_MONTHLY_MULTIPLIER * (SEASONALITY_AMPLIFICATION / BASE_MONTHLY_MULTIPLIER.mean())

# Sample representative individuals across age distribution
AGE_BINS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 60)]
N_PER_BIN = [2, 2, 3, 3, 3, 2]  # Total = 15 individuals


def calculate_observed_metrics(df):
    """Calculate key metrics from observed data"""
    df['age_group'] = pd.cut(df['age'], bins=[0, 1, 2, 5, 10, 100],
                              labels=['<1', '1-2', '2-5', '5-10', '10+'])

    results = {}

    # Clinical episodes
    clinical = df[
        (df['visittype'] == 'Unscheduled visit') &
        (df['antimalarial'].str.contains('Artmether|Artemether|Quinine|Artesunate',
                                         case=False, na=False)) &
        (df['parasitedensity'] > 0)
    ]

    # Routine visits
    routine = df[df['visittype'] == 'Scheduled visit']

    for age_group in ['<1', '1-2', '2-5', '5-10', '10+']:
        age_data = df[df['age_group'] == age_group]

        if len(age_data) > 0:
            # Person-time
            person_years = (age_data.groupby('id')['date'].apply(
                lambda x: (x.max() - x.min()).days / 365.25
            )).sum()

            # Clinical incidence
            n_clinical = len(clinical[clinical['age_group'] == age_group])
            clinical_incidence = n_clinical / person_years if person_years > 0 else 0

            # Routine prevalence
            routine_age = routine[routine['age_group'] == age_group]
            routine_prevalence = (routine_age['parasitedensity'] > 0).mean()

            # Clinical densities
            clinical_age = clinical[clinical['age_group'] == age_group]
            clinical_densities = clinical_age['parasitedensity'].values

            results[age_group] = {
                'clinical_incidence': clinical_incidence,
                'routine_prevalence': routine_prevalence,
                'clinical_density_geomean': np.exp(np.log(clinical_densities[clinical_densities > 0]).mean()) if len(clinical_densities) > 0 else 0,
                'n_clinical': n_clinical,
                'person_years': person_years,
            }

    return results


def run_single_simulation(monthly_eir, treatment_coverage, ind_idx, age_min, age_max, config=None):
    """Run a single simulation with given parameters"""
    # Sample age uniformly within bin
    age_at_enrollment = np.random.uniform(age_min, age_max)

    # Monthly EIRs with seasonality
    monthly_eirs = MONTHLY_MULTIPLIER * monthly_eir

    # Initialize individual
    if config is None:
        config = create_config()

    ic = IntrahostComponent.create(config)
    ic.susceptibility.age = 0.0  # Start at birth

    # Burn-in period: age individual to enrollment age with lower treatment coverage
    burnin_days = int(age_at_enrollment * 365.25)
    burnin_treatment = ComplexTreatment(prob_treat=TREATMENT_COVERAGE_BURNIN)

    if burnin_days > 0:
        monthly_eir_challenge(
            duration=burnin_days,
            monthly_eirs=monthly_eirs,
            callback=burnin_treatment,
            ic=ic,
            start_day=0,
        )

    # Trial period: simulate with trial treatment coverage
    trial_days = int(SIMULATION_YEARS * 365.25)
    trial_treatment = ComplexTreatment(prob_treat=treatment_coverage)

    trial_df = monthly_eir_challenge(
        duration=trial_days,
        monthly_eirs=monthly_eirs,
        callback=trial_treatment,
        ic=ic,
        start_day=burnin_days,  # Continue from where burn-in left off for seasonality
    )

    # Add metadata
    trial_df['id'] = ind_idx
    trial_df['age_at_enrollment'] = age_at_enrollment
    trial_df['monthly_eir'] = monthly_eir
    trial_df['treatment_coverage'] = treatment_coverage

    return trial_df


def run_parameter_combination(eir, coverage):
    """Run all individuals for a single parameter combination (sequential)"""
    print(f"\n  Running EIR={eir:.2f}, Coverage={coverage:.2f}...")
    start_time = time.time()

    # Generate parameter sets for all individuals
    all_results = []
    ind_idx = 0

    for (age_min, age_max), n in zip(AGE_BINS, N_PER_BIN):
        for _ in range(n):
            # Run simulation sequentially
            df = run_single_simulation(eir, coverage, ind_idx, age_min, age_max)

            # Reset index to get 'days' as column
            df = df.reset_index()
            if 'days' in df.columns:
                df = df.rename(columns={'days': 'day'})

            all_results.append(df)
            ind_idx += 1

            # Progress indicator
            if ind_idx % 5 == 0:
                print(f"    Completed {ind_idx}/{N_INDIVIDUALS} individuals...")

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f} seconds")

    return combined_df


def analyze_simulation_results(df, monthly_eir, treatment_coverage):
    """Calculate metrics from simulation results"""
    # Add age at each time point (df already has 'day' column from run_parameter_combination)
    df['age'] = df['age_at_enrollment'] + (df['day'] / 365.25)
    df['age_group'] = pd.cut(df['age'], bins=[0, 1, 2, 5, 10, 100],
                              labels=['<1', '1-2', '2-5', '5-10', '10+'])

    results = {
        'monthly_eir': monthly_eir,
        'treatment_coverage': treatment_coverage,
    }

    # Identify clinical episodes (simplified detection)
    # Clinical = fever + parasites + treatment
    df['clinical_episode'] = (
        (df['fever_temperature'] > 38.5) &
        (df['parasite_density'] > 0) &
        (df['treatments'] > 0)  # 'treatments' not 'treated'
    )

    # Routine visits (monthly)
    routine_days = np.arange(0, df['day'].max(), 30)

    for age_group in ['<1', '1-2', '2-5', '5-10', '10+']:
        age_data = df[df['age_group'] == age_group]

        if len(age_data) > 0:
            # Person-time
            person_years = len(age_data['id'].unique()) * (age_data['day'].max() / 365.25)

            # Clinical incidence
            n_clinical = age_data['clinical_episode'].sum()
            clinical_incidence = n_clinical / person_years if person_years > 0 else 0

            # Routine prevalence (sample monthly)
            routine_positives = []
            for day in routine_days:
                day_data = age_data[(age_data['day'] >= day) & (age_data['day'] < day + 1)]
                if len(day_data) > 0:
                    routine_positives.append((day_data['parasite_density'] > 0).any())
            routine_prevalence = np.mean(routine_positives) if len(routine_positives) > 0 else 0

            # Clinical densities
            clinical_densities = age_data[age_data['clinical_episode']]['parasite_density'].values
            clinical_density_geomean = (
                np.exp(np.log(clinical_densities[clinical_densities > 0]).mean())
                if len(clinical_densities) > 0 else 0
            )

            results[f'{age_group}_incidence'] = clinical_incidence
            results[f'{age_group}_prevalence'] = routine_prevalence
            results[f'{age_group}_density'] = clinical_density_geomean

    return results


def calculate_fit_score(sim_results, obs_metrics):
    """Calculate goodness-of-fit score comparing simulated to observed"""
    score = 0.0
    n_metrics = 0

    age_groups = ['<1', '1-2', '2-5', '5-10', '10+']

    for age_group in age_groups:
        if age_group not in obs_metrics:
            continue

        obs = obs_metrics[age_group]

        # Clinical incidence (log-scale ratio)
        sim_inc = sim_results.get(f'{age_group}_incidence', 0)
        if obs['clinical_incidence'] > 0 and sim_inc > 0:
            ratio = sim_inc / obs['clinical_incidence']
            score += (np.log(ratio)) ** 2
            n_metrics += 1

        # Routine prevalence (absolute difference)
        sim_prev = sim_results.get(f'{age_group}_prevalence', 0)
        score += (sim_prev - obs['routine_prevalence']) ** 2
        n_metrics += 1

    return score / n_metrics if n_metrics > 0 else float('inf')


def main():
    """Main execution function"""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("PARAMETER SWEEP: EIR × TREATMENT COVERAGE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Site: {SITE}")
    print(f"  EIR values: {EIR_VALUES}")
    print(f"  Treatment coverage values: {TREATMENT_COVERAGE_VALUES}")
    print(f"  Total combinations: {len(EIR_VALUES) * len(TREATMENT_COVERAGE_VALUES)}")
    print(f"  Individuals per run: {N_INDIVIDUALS}")
    print(f"  Simulation duration: {SIMULATION_YEARS} years")
    print(f"  Burn-in duration: {BURNIN_YEARS} years")

    # Load observed data for comparison
    print(f"\nLoading observed data from {SITE}...")
    obs_df = pd.read_csv(DATA_DIR / f'prism_cleaned_{SITE}.csv', parse_dates=['date'])
    obs_df = obs_df[obs_df['date'] < '2015-01-01'].copy()  # Pre-IRS only
    print(f"  {len(obs_df)} observations in pre-IRS period")

    observed_metrics = calculate_observed_metrics(obs_df)
    print("\nObserved metrics (pre-IRS):")
    print(f"{'Age':<8} {'Incidence':<12} {'Prevalence':<12} {'Geomean Dens':<15}")
    print("-" * 50)
    for age_group in ['<1', '1-2', '2-5', '5-10', '10+']:
        if age_group in observed_metrics:
            m = observed_metrics[age_group]
            print(f"{age_group:<8} {m['clinical_incidence']:>11.2f} {m['routine_prevalence']:>11.1%} {m['clinical_density_geomean']:>14.0f}")

    # Run parameter sweep
    print("\n" + "=" * 80)
    print("RUNNING PARAMETER SWEEP")
    print("=" * 80)

    sweep_results = []
    total_combinations = len(EIR_VALUES) * len(TREATMENT_COVERAGE_VALUES)
    combination_num = 0

    for eir in EIR_VALUES:
        for coverage in TREATMENT_COVERAGE_VALUES:
            combination_num += 1
            print(f"\n[{combination_num}/{total_combinations}]")

            # Run simulation
            sim_df = run_parameter_combination(eir, coverage)

            # Analyze results
            metrics = analyze_simulation_results(sim_df, eir, coverage)

            # Calculate fit score
            fit_score = calculate_fit_score(metrics, observed_metrics)
            metrics['fit_score'] = fit_score

            sweep_results.append(metrics)

            print(f"    Fit score: {fit_score:.4f}")

    # Save results
    results_df = pd.DataFrame(sweep_results)
    output_file = OUTPUT_DIR / f'sweep_results_{SITE}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved sweep results to: {output_file}")

    # Create visualization
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    age_groups = ['<1', '1-2', '2-5', '5-10', '10+']

    # Heatmaps for each age group's clinical incidence
    for idx, age_group in enumerate(age_groups[:5]):
        ax = axes[idx]

        # Pivot data for heatmap
        pivot_data = results_df.pivot(
            index='treatment_coverage',
            columns='monthly_eir',
            values=f'{age_group}_incidence'
        )

        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    ax=ax, cbar_kws={'label': 'Episodes/person-year'})

        ax.set_title(f'Clinical Incidence: Age {age_group}\n(Observed: {observed_metrics[age_group]["clinical_incidence"]:.2f})',
                     fontweight='bold')
        ax.set_xlabel('Monthly EIR')
        ax.set_ylabel('Treatment Coverage')

    # Overall fit score heatmap
    ax = axes[5]
    pivot_fit = results_df.pivot(
        index='treatment_coverage',
        columns='monthly_eir',
        values='fit_score'
    )

    sns.heatmap(pivot_fit, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax, cbar_kws={'label': 'Fit Score (lower=better)'})

    ax.set_title('Overall Fit Score\n(Combined incidence + prevalence)',
                 fontweight='bold')
    ax.set_xlabel('Monthly EIR')
    ax.set_ylabel('Treatment Coverage')

    plt.tight_layout()
    fig_file = OUTPUT_DIR / f'sweep_heatmaps_{SITE}.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap visualization to: {fig_file}")
    plt.close()

    # Find best parameter combination
    best_idx = results_df['fit_score'].idxmin()
    best_params = results_df.loc[best_idx]

    print("\n" + "=" * 80)
    print("BEST PARAMETER COMBINATION")
    print("=" * 80)
    print(f"\nMonthly EIR: {best_params['monthly_eir']:.2f}")
    print(f"Treatment Coverage: {best_params['treatment_coverage']:.2f}")
    print(f"Fit Score: {best_params['fit_score']:.4f}")

    print("\n" + "=" * 80)
    print("COMPARISON TO OBSERVED DATA")
    print("=" * 80)
    print(f"\n{'Age':<8} {'Obs Inc':<10} {'Sim Inc':<10} {'Ratio':<8} {'Obs Prev':<10} {'Sim Prev':<10}")
    print("-" * 70)

    for age_group in age_groups:
        if age_group in observed_metrics:
            obs = observed_metrics[age_group]
            sim_inc = best_params[f'{age_group}_incidence']
            sim_prev = best_params[f'{age_group}_prevalence']
            ratio = sim_inc / obs['clinical_incidence'] if obs['clinical_incidence'] > 0 else 0

            print(f"{age_group:<8} {obs['clinical_incidence']:>9.2f} {sim_inc:>9.2f} {ratio:>7.2f}x "
                  f"{obs['routine_prevalence']:>9.1%} {sim_prev:>9.1%}")

    print("\n" + "=" * 80)
    print("Parameter sweep complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
