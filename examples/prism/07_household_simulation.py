"""
Household Malaria Simulation Matching PRISM Protocol

Simulates a household with multiple individuals following PRISM study design:
- Individual immune initialization based on enrollment age (Barry2021 pattern)
- Quarterly visits (2011-2014), then monthly visits (2014+)
- Fever-triggered unscheduled visits with treatment
- Microscopy at all visits, LAMP for negatives at routine visits
"""

import sys
sys.path.append('../..')
sys.path.append('..')  # Add parent examples directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from emodlib.malaria import IntrahostComponent
from seasonal_challenge import ComplexTreatment

# ==============================================================================
# CONFIGURATION PARAMETERS
# ==============================================================================

# Site selection
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'

# Representative cohort sampled from observed data (8 per age bin: <1, 1-2, 2-5, 5-10, 10-20, 20+)
# Larger sample size for better statistics on individual-level variation
HOUSEHOLD_MEMBERS = [
    # Age <1 (8 individuals)
    {'id': 3381, 'age_years': 0.8, 'gender': 'Female', 'enrollment_date': '2012-07-19'},
    {'id': 3404, 'age_years': 0.6, 'gender': 'Male', 'enrollment_date': '2013-06-02'},
    {'id': 3370, 'age_years': 0.7, 'gender': 'Male', 'enrollment_date': '2012-02-23'},
    {'id': 3395, 'age_years': 0.5, 'gender': 'Male', 'enrollment_date': '2012-12-09'},
    {'id': 3408, 'age_years': 0.8, 'gender': 'Male', 'enrollment_date': '2013-08-25'},
    {'id': 3453, 'age_years': 0.9, 'gender': 'Male', 'enrollment_date': '2014-11-06'},
    {'id': 3374, 'age_years': 0.5, 'gender': 'Female', 'enrollment_date': '2012-03-19'},
    {'id': 3405, 'age_years': 0.7, 'gender': 'Female', 'enrollment_date': '2013-06-25'},
    # Age 1-2 (8 individuals)
    {'id': 3488, 'age_years': 1.8, 'gender': 'Male', 'enrollment_date': '2016-08-23'},
    {'id': 3468, 'age_years': 1.6, 'gender': 'Female', 'enrollment_date': '2015-09-04'},
    {'id': 3169, 'age_years': 1.8, 'gender': 'Female', 'enrollment_date': '2011-09-04'},
    {'id': 3490, 'age_years': 1.4, 'gender': 'Male', 'enrollment_date': '2016-09-13'},
    {'id': 3323, 'age_years': 1.5, 'gender': 'Male', 'enrollment_date': '2011-09-09'},
    {'id': 3282, 'age_years': 1.5, 'gender': 'Male', 'enrollment_date': '2011-09-01'},
    {'id': 3497, 'age_years': 1.3, 'gender': 'Male', 'enrollment_date': '2017-06-30'},
    {'id': 3459, 'age_years': 2.0, 'gender': 'Female', 'enrollment_date': '2015-02-15'},
    # Age 2-5 (8 individuals)
    {'id': 3357, 'age_years': 4.5, 'gender': 'Male', 'enrollment_date': '2011-09-12'},
    {'id': 3378, 'age_years': 3.0, 'gender': 'Male', 'enrollment_date': '2012-05-13'},
    {'id': 3168, 'age_years': 4.2, 'gender': 'Male', 'enrollment_date': '2011-08-26'},
    {'id': 3346, 'age_years': 2.1, 'gender': 'Female', 'enrollment_date': '2011-09-14'},
    {'id': 3434, 'age_years': 3.8, 'gender': 'Male', 'enrollment_date': '2013-12-19'},
    {'id': 3388, 'age_years': 3.2, 'gender': 'Male', 'enrollment_date': '2012-08-04'},
    {'id': 3463, 'age_years': 4.4, 'gender': 'Female', 'enrollment_date': '2015-06-14'},
    {'id': 3049, 'age_years': 3.7, 'gender': 'Female', 'enrollment_date': '2011-08-23'},
    # Age 5-10 (8 individuals)
    {'id': 3198, 'age_years': 9.0, 'gender': 'Female', 'enrollment_date': '2011-09-10'},
    {'id': 3315, 'age_years': 5.2, 'gender': 'Female', 'enrollment_date': '2011-09-03'},
    {'id': 3387, 'age_years': 5.2, 'gender': 'Male', 'enrollment_date': '2012-08-09'},
    {'id': 3242, 'age_years': 8.2, 'gender': 'Male', 'enrollment_date': '2011-08-31'},
    {'id': 3106, 'age_years': 8.7, 'gender': 'Male', 'enrollment_date': '2011-08-21'},
    {'id': 3352, 'age_years': 5.3, 'gender': 'Male', 'enrollment_date': '2011-09-06'},
    {'id': 3241, 'age_years': 9.2, 'gender': 'Female', 'enrollment_date': '2011-09-11'},
    {'id': 3116, 'age_years': 8.3, 'gender': 'Female', 'enrollment_date': '2011-09-03'},
    # Age 10-20 (3 individuals)
    {'id': 3020, 'age_years': 18.2, 'gender': 'Female', 'enrollment_date': '2011-08-25'},
    {'id': 3183, 'age_years': 20.0, 'gender': 'Female', 'enrollment_date': '2011-09-05'},
    {'id': 3185, 'age_years': 18.9, 'gender': 'Female', 'enrollment_date': '2011-09-04'},
    # Age 20+ (8 individuals)
    {'id': 3206, 'age_years': 27.8, 'gender': 'Male', 'enrollment_date': '2011-09-06'},
    {'id': 3115, 'age_years': 45.8, 'gender': 'Female', 'enrollment_date': '2011-09-01'},
    {'id': 3239, 'age_years': 61.9, 'gender': 'Female', 'enrollment_date': '2011-09-01'},
    {'id': 3410, 'age_years': 39.2, 'gender': 'Female', 'enrollment_date': '2013-09-15'},
    {'id': 3344, 'age_years': 23.2, 'gender': 'Female', 'enrollment_date': '2011-09-07'},
    {'id': 3334, 'age_years': 26.9, 'gender': 'Female', 'enrollment_date': '2011-09-11'},
    {'id': 3036, 'age_years': 30.3, 'gender': 'Female', 'enrollment_date': '2011-08-19'},
    {'id': 3023, 'age_years': 27.2, 'gender': 'Female', 'enrollment_date': '2011-08-29'},
]

# Simulation time period
SIMULATION_START = '2011-08-01'  # Start before first enrollment
SIMULATION_END = '2016-12-31'    # Match PRISM study duration
SIMULATION_DAYS = (pd.to_datetime(SIMULATION_END) - pd.to_datetime(SIMULATION_START)).days

# Transmission parameters
MONTHLY_EIR = 0.5  # Infectious bites per person per month (calibrated to pre-IRS clinical incidence)

# Seasonality based on observed clinical malaria cases in Nagongera
# Pattern: Peak in May-June (rainy season), lowest in Feb-Mar (dry season)
SEASONALITY_MONTHS = [1.21, 0.80, 0.71, 0.98, 1.27, 1.28, 0.96, 0.82, 0.88, 0.84, 1.08, 1.17]  # Jan-Dec

# IRS (Indoor Residual Spraying) intervention
IRS_START_DATE = '2015-01-01'  # When IRS began in Nagongera
IRS_EFFICACY = 0.83  # Reduction in transmission (6.05x fold reduction = 1 - 1/6.05 = 83%)

# Age-dependent biting (surface area model)
def biting_risk(age_days):
    """Relative biting risk by age (Barry2021 surface area model)"""
    age_years = age_days / 365.0
    if age_years < 0.25:
        return 0.5
    elif age_years < 20:
        return 0.5 + 0.5 * (age_years / 20.0)
    else:
        return 1.0

# Visit schedule parameters
QUARTERLY_INTERVAL_DAYS = 90   # Quarterly visits in early period
MONTHLY_INTERVAL_DAYS = 28     # Monthly visits in later period
QUARTERLY_TO_MONTHLY_DATE = '2014-12-01'  # Transition date (staggered rollout Dec 2014 - Jan 2015)

# Fever and treatment parameters (using ComplexTreatment from seasonal_challenge.py)
TREATMENT_COVERAGE_BURNIN = 0.50   # Treatment coverage during burn-in period (lower, pre-study)
TREATMENT_COVERAGE_TRIAL = 0.75    # Treatment coverage during trial period (balancing incidence vs prevalence)

# Treatment-seeking delay
# Literature suggests 1-3 days between fever onset and clinic visit in high-access settings
TREATMENT_DELAY_MEAN_DAYS = 3.0  # Mean delay (exponentially distributed)

# Microscopy parameters
# Probabilistic detection based on Poisson sampling model
# Model: Examining 1/20th μL of blood, detection probability P(≥1) = 1 - exp(-density/20)
# At 32 parasites/μL: λ=1.6, P(detect)=80% (most detected)
# At 16 parasites/μL: λ=0.8, P(detect)=55% (about half)
# At 8 parasites/μL: λ=0.4, P(detect)=33% (rarely detected)
# Hard cutoff at <8 parasites/μL matches lowest observed values in PRISM data
MICROSCOPY_SAMPLE_VOLUME_FRACTION = 1.0/20.0  # Fraction of μL examined
MICROSCOPY_HARD_CUTOFF = 8.0  # Below this, assume 0% detection (lowest recorded value)

# Gametocyte detection uses same probabilistic model
# Literature: gametocytes constitute <5% of infections, similar sensitivity to asexual
GAMETOCYTE_SAMPLE_VOLUME_FRACTION = 1.0/20.0  # Same as asexual microscopy
GAMETOCYTE_HARD_CUTOFF = 8.0  # Same hard cutoff

# LAMP parameters (only done on microscopy negatives)
# Literature (PRISM Uganda studies, Malaria Journal 2016): detection limit 1-5 parasites/μL
# LAMP is highly sensitive molecular method for submicroscopic infections
LAMP_THRESHOLD = 2.0  # Midpoint of 1-5 range (deterministic threshold)

# Model parameters (from Barry2021 calibration)
MODEL_PARAMS = {
    'Antigen_Switch_Rate': 2.0e-9,
    'Falciparum_PfEMP1_Variants': 1150,
    'Max_Individual_Infections': 5,
}

# Time stepping
N_UPDATES_PER_TIMESTEP = 2  # Number of updates per day (like seasonal_challenge.py)

# Initialization/burn-in
BURNIN_EIR = 0.5  # EIR during burn-in period (same as MONTHLY_EIR by default)

# Output configuration
OUTPUT_DIR = Path('simulation_output')
SAVE_DAILY_OUTPUT = True
SAVE_VISIT_OUTPUT = True

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def initialize_individual(age_years, burnin_eir=BURNIN_EIR, burnin_treatment_coverage=TREATMENT_COVERAGE_BURNIN):
    """
    Initialize individual with age-appropriate immune state (Barry2021 pattern).

    Runs burn-in simulation from birth to enrollment age under constant EIR
    to build up age-appropriate immunity.

    Parameters
    ----------
    age_years : float
        Age at enrollment in years
    burnin_eir : float
        Monthly EIR during burn-in period
    burnin_treatment_coverage : float
        Treatment coverage during burn-in (typically lower than trial period)
    """
    ic = IntrahostComponent.create()
    ic.susceptibility.age = 0.0  # Start at birth

    # Initialize treatment for burn-in
    treatment = ComplexTreatment(prob_treat=burnin_treatment_coverage)

    # Burn-in: simulate from birth to enrollment age
    burnin_days = int(age_years * 365.0)
    daily_burnin_eir = burnin_eir / 30.0  # Convert monthly to daily

    if burnin_days > 0:
        print(f"    Running {burnin_days} day burn-in for {age_years:.1f} year old " +
              f"(EIR={burnin_eir:.1f}, treatment={burnin_treatment_coverage*100:.0f}%)...")

    for day in range(burnin_days):
        # Age-dependent exposure during burn-in
        age_days = ic.susceptibility.age
        relative_biting = biting_risk(age_days)
        individual_eir = daily_burnin_eir * relative_biting

        # Exposure (Poisson-distributed bites)
        n_bites = np.random.poisson(individual_eir)
        for _ in range(n_bites):
            ic.challenge()

        # Update (using same sub-stepping as main simulation)
        dt = 1.0 / N_UPDATES_PER_TIMESTEP
        for _ in range(N_UPDATES_PER_TIMESTEP):
            ic.update(dt=dt)

        # Apply treatment during burn-in (fever-triggered)
        treatment(ic)

    return ic

def month_from_date(date):
    """Get month index (0-11) from date"""
    return date.month - 1

def get_eir_for_date(date, base_monthly_eir, seasonality, irs_start_date, irs_efficacy):
    """
    Calculate EIR for a given date based on seasonality and IRS intervention.

    Parameters
    ----------
    date : datetime
        Current date
    base_monthly_eir : float
        Base monthly EIR (before IRS)
    seasonality : list
        12-element list of relative seasonal multipliers
    irs_start_date : datetime
        Date when IRS began
    irs_efficacy : float
        Proportional reduction in transmission due to IRS (0-1)

    Returns
    -------
    float
        Daily EIR for the given date
    """
    month_idx = month_from_date(date)
    seasonal_factor = seasonality[month_idx]

    # Normalize so mean = 1.0
    mean_seasonal = np.mean(seasonality)
    daily_eir = base_monthly_eir * (seasonal_factor / mean_seasonal) / 30.0

    # Apply IRS reduction if after start date
    if date >= irs_start_date:
        daily_eir *= (1.0 - irs_efficacy)

    return daily_eir

def should_have_visit(current_date, enrollment_date, last_visit_date, transition_date):
    """
    Determine if individual should have routine scheduled visit.

    Returns: (should_visit, visit_type)
    """
    if current_date < enrollment_date:
        return False, None

    if current_date < transition_date:
        # Quarterly period
        days_since_last = (current_date - last_visit_date).days
        if days_since_last >= QUARTERLY_INTERVAL_DAYS:
            return True, 'Scheduled visit'
    else:
        # Monthly period
        days_since_last = (current_date - last_visit_date).days
        if days_since_last >= MONTHLY_INTERVAL_DAYS:
            return True, 'Scheduled visit'

    return False, None

def perform_microscopy(parasite_density, gametocyte_density):
    """
    Simulate microscopy results using probabilistic Poisson sampling model.

    Detection probability = 1 - exp(-lambda) where lambda = density * sample_volume_fraction
    Hard cutoff below minimum observed density.

    Parameters
    ----------
    parasite_density : float
        True asexual parasite density (parasites/μL)
    gametocyte_density : float
        True gametocyte density (gametocytes/μL)

    Returns
    -------
    asexual_result : bool
        True if asexual parasites detected
    gametocyte_result : bool
        True if gametocytes detected
    detected_density : float
        Reported density (0 if negative, true density if positive)
    """
    # Asexual parasite detection
    if parasite_density < MICROSCOPY_HARD_CUTOFF:
        asexual_positive = False
    else:
        # Poisson sampling: P(detect ≥1) = 1 - P(0) = 1 - exp(-λ)
        lambda_asexual = parasite_density * MICROSCOPY_SAMPLE_VOLUME_FRACTION
        detection_prob = 1.0 - np.exp(-lambda_asexual)
        asexual_positive = np.random.random() < detection_prob

    # Gametocyte detection (same model)
    if gametocyte_density < GAMETOCYTE_HARD_CUTOFF:
        gametocyte_positive = False
    else:
        lambda_gam = gametocyte_density * GAMETOCYTE_SAMPLE_VOLUME_FRACTION
        detection_prob = 1.0 - np.exp(-lambda_gam)
        gametocyte_positive = np.random.random() < detection_prob

    # If positive, report true density; if negative, report 0
    detected_density = parasite_density if asexual_positive else 0.0

    return asexual_positive, gametocyte_positive, detected_density

def perform_lamp(parasite_density):
    """
    Simulate LAMP test (only done on microscopy negatives).

    Returns: lamp_result ('Positive', 'Negative', or 'Na')
    """
    if parasite_density >= LAMP_THRESHOLD:
        return 'Positive'
    else:
        return 'Negative'

# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def simulate_household():
    """Run household simulation"""

    print("=" * 80)
    print("HOUSEHOLD MALARIA SIMULATION")
    print("=" * 80)
    print(f"\nSimulating {len(HOUSEHOLD_MEMBERS)} household members")
    print(f"Period: {SIMULATION_START} to {SIMULATION_END} ({SIMULATION_DAYS} days)")
    print(f"Monthly EIR: {MONTHLY_EIR} (pre-IRS)")
    print(f"IRS: {IRS_EFFICACY*100:.0f}% reduction starting {IRS_START_DATE}")
    print(f"Treatment coverage: {TREATMENT_COVERAGE_BURNIN*100:.0f}% (burn-in) → {TREATMENT_COVERAGE_TRIAL*100:.0f}% (trial)")

    # Set global parameters
    IntrahostComponent.set_params(MODEL_PARAMS)

    # Initialize household members
    individuals = {}
    for member in HOUSEHOLD_MEMBERS:
        ic = initialize_individual(member['age_years'])
        enrollment_date = pd.to_datetime(member['enrollment_date'])

        # Initialize ComplexTreatment for this individual (trial period coverage)
        treatment = ComplexTreatment(prob_treat=TREATMENT_COVERAGE_TRIAL)

        individuals[member['id']] = {
            'component': ic,
            'treatment': treatment,
            'age_years': member['age_years'],
            'gender': member['gender'],
            'enrollment_date': enrollment_date,
            'last_visit_date': enrollment_date,  # First visit is enrollment
            'daily_records': [],
            'visit_records': [],
            'pending_treatment_date': None,  # Date when treatment will occur (None if no pending treatment)
        }

    # Simulation loop
    print("\nRunning simulation...")
    start_date = pd.to_datetime(SIMULATION_START)
    transition_date = pd.to_datetime(QUARTERLY_TO_MONTHLY_DATE)
    irs_start = pd.to_datetime(IRS_START_DATE)

    for day in range(SIMULATION_DAYS):
        current_date = start_date + timedelta(days=day)

        if day % 365 == 0:
            print(f"  Day {day}/{SIMULATION_DAYS} ({current_date.date()})")

        # Calculate daily EIR (includes seasonality and IRS effect)
        daily_eir = get_eir_for_date(current_date, MONTHLY_EIR, SEASONALITY_MONTHS,
                                     irs_start, IRS_EFFICACY)

        # Update each individual
        for person_id, person in individuals.items():
            ic = person['component']
            treatment = person['treatment']

            # Skip if not yet enrolled
            if current_date < person['enrollment_date']:
                continue

            # Age-dependent exposure
            age_days = ic.susceptibility.age
            relative_biting = biting_risk(age_days)
            individual_eir = daily_eir * relative_biting

            # Exposure (Poisson-distributed bites)
            n_bites = np.random.poisson(individual_eir)
            for _ in range(n_bites):
                ic.challenge()

            # Update within-host dynamics (multiple updates per timestep)
            dt = 1.0 / N_UPDATES_PER_TIMESTEP
            for _ in range(N_UPDATES_PER_TIMESTEP):
                ic.update(dt=dt)

            # Get current state
            parasite_density = ic.parasite_density
            gametocyte_density = ic.gametocyte_density
            fever_temp = ic.fever_temperature
            has_fever = fever_temp >= treatment.Clinical_Fever_Threshold_Low

            # Record daily state
            if SAVE_DAILY_OUTPUT:
                person['daily_records'].append({
                    'date': current_date,
                    'day': day,
                    'parasite_density': parasite_density,
                    'gametocyte_density': gametocyte_density,
                    'fever_temperature': fever_temp,
                    'has_fever': has_fever,
                    'n_infections': ic.n_infections,
                })

            # Check if we should schedule a treatment visit (fever onset detection)
            # Use ComplexTreatment to check if this is a new clinical episode
            if person['pending_treatment_date'] is None and has_fever:
                # Check if enough time has passed since last episode (14 days)
                # This mimics ComplexTreatment logic without calling treatment() yet
                days_since_last_visit = (current_date - person['last_visit_date']).days
                if days_since_last_visit >= treatment.Min_Days_Between_Clinical_Incidents:
                    # Schedule treatment with exponential delay
                    if np.random.random() < TREATMENT_COVERAGE_TRIAL:
                        delay_days = np.random.exponential(TREATMENT_DELAY_MEAN_DAYS)
                        person['pending_treatment_date'] = current_date + timedelta(days=delay_days)

            # Check if pending treatment should occur today
            if person['pending_treatment_date'] is not None and current_date >= person['pending_treatment_date']:
                # Treatment visit occurs
                # Record densities BEFORE treatment (this is what microscopy detects)
                pre_treatment_density = parasite_density
                pre_treatment_gametocytes = gametocyte_density

                # Apply treatment using ComplexTreatment
                treatment(ic)

                # Record as unscheduled visit
                # Use densities from just before treatment (current state at visit)
                asexual_pos, gam_pos, detected_density = perform_microscopy(
                    pre_treatment_density, pre_treatment_gametocytes
                )

                person['visit_records'].append({
                    'date': current_date,
                    'visittype': 'Unscheduled visit',
                    'fever': 'Yes',
                    'parasitedensity': detected_density,
                    'gametocytes': 'Yes' if gam_pos else 'No',
                    'LAMP': 'Na',  # Not done at unscheduled visits
                    'malaria_diagnosis': 'Yes' if asexual_pos else 'No',
                    'antimalarial': 'Artemether-lumefantrine for uncomplicated malaria',
                })

                person['last_visit_date'] = current_date
                # Clear pending treatment
                person['pending_treatment_date'] = None

            # Check for scheduled routine visit
            should_visit, visit_type = should_have_visit(
                current_date, person['enrollment_date'],
                person['last_visit_date'], transition_date
            )

            if should_visit:
                # Routine scheduled visit
                asexual_pos, gam_pos, detected_density = perform_microscopy(
                    parasite_density, gametocyte_density
                )

                # LAMP only if microscopy negative
                if not asexual_pos:
                    lamp_result = perform_lamp(parasite_density)
                else:
                    lamp_result = 'Na'

                person['visit_records'].append({
                    'date': current_date,
                    'visittype': visit_type,
                    'fever': 'Yes' if has_fever else 'No',
                    'parasitedensity': detected_density,
                    'gametocytes': 'Yes' if gam_pos else 'No',
                    'LAMP': lamp_result,
                    'malaria_diagnosis': 'No',  # Routine visits don't diagnose
                    'antimalarial': 'No malaria medications given',
                })

                person['last_visit_date'] = current_date

    print("\nSimulation complete!")

    return individuals

# ==============================================================================
# OUTPUT AND VISUALIZATION
# ==============================================================================

def save_results(individuals):
    """Save simulation results to CSV files"""

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save daily output
    if SAVE_DAILY_OUTPUT:
        daily_data = []
        for person_id, person in individuals.items():
            for record in person['daily_records']:
                daily_data.append({
                    'id': person_id,
                    'age_at_enrollment': person['age_years'],
                    'gender': person['gender'],
                    **record
                })

        daily_df = pd.DataFrame(daily_data)
        daily_file = OUTPUT_DIR / 'household_daily_output.csv'
        daily_df.to_csv(daily_file, index=False)
        print(f"\nSaved daily output: {daily_file}")
        print(f"  {len(daily_df)} daily records")

    # Save visit output
    if SAVE_VISIT_OUTPUT:
        visit_data = []
        for person_id, person in individuals.items():
            for record in person['visit_records']:
                visit_data.append({
                    'id': person_id,
                    'age_at_enrollment': person['age_years'],
                    'gender': person['gender'],
                    **record
                })

        visit_df = pd.DataFrame(visit_data)
        visit_file = OUTPUT_DIR / 'household_visit_output.csv'
        visit_df.to_csv(visit_file, index=False)
        print(f"\nSaved visit output: {visit_file}")
        print(f"  {len(visit_df)} visit records")

        # Print summary statistics
        print("\nVisit summary:")
        print(visit_df.groupby('id').agg({
            'visittype': 'count',
            'parasitedensity': lambda x: (x > 0).sum()
        }).rename(columns={'visittype': 'n_visits', 'parasitedensity': 'n_positive'}))

def create_visualizations():
    """Create PRISM-style visualizations of simulation results"""

    import sys
    sys.path.append('.')
    from plotting_utils import plot_individual_compact, plot_individual_detailed

    print("\n" + "=" * 80)
    print(f"CREATING VISUALIZATIONS (Site: {SITE.upper()})")
    print("=" * 80)

    # Load simulation outputs
    visit_df = pd.read_csv(OUTPUT_DIR / 'household_visit_output.csv', parse_dates=['date'])
    daily_df = pd.read_csv(OUTPUT_DIR / 'household_daily_output.csv', parse_dates=['date'])

    # Add age column (for sorting in compact plot)
    visit_df['age'] = visit_df['age_at_enrollment']

    figures_dir = OUTPUT_DIR / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # =========================================================================
    # 1. PRISM-style compact household plot
    # =========================================================================
    print("\n1. Creating PRISM-style compact household plot...")

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plot_individual_compact(visit_df, ax=ax, fig=fig, detailed=True)

    # Add date range buffer
    date_min = visit_df['date'].min() - pd.DateOffset(months=3)
    date_max = visit_df['date'].max() + pd.DateOffset(months=3)
    ax.set_xlim(date_min, date_max)

    # Add label
    ax.text(0.98, 0.98, f'SIMULATED Household',
            transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7,
                     edgecolor='black', linewidth=1.5))

    # Create title
    n_members = visit_df['id'].nunique()
    member_summary = visit_df.groupby('id').agg({
        'age_at_enrollment': 'first',
        'gender': 'first',
        'parasitedensity': lambda x: (x > 0).sum()
    })
    member_list = ', '.join([
        f"ID{idx}({int(row['age_at_enrollment'])}{row['gender'][0]}, {int(row['parasitedensity'])}+)"
        for idx, row in member_summary.iterrows()
    ])

    title = f'Simulated Household: PRISM-style Compact View\n'
    title += f'{n_members} members (EIR={MONTHLY_EIR} pre-IRS, IRS={IRS_EFFICACY*100:.0f}% reduction, Treatment={TREATMENT_COVERAGE_TRIAL*100:.0f}%)\n'
    title += f'Members: {member_list}'

    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    filename = figures_dir / 'simulated_household_compact.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

    # =========================================================================
    # 2. Detailed daily time-series (one facet per individual)
    # =========================================================================
    print("\n2. Creating detailed daily time-series plots...")

    n_individuals = daily_df['id'].nunique()
    # Sort individuals by age (oldest first) to match compact plot order
    age_order = daily_df.groupby('id')['age_at_enrollment'].first().sort_values(ascending=False)
    individuals = age_order.index.tolist()

    fig, axes = plt.subplots(n_individuals, 1, figsize=(16, 3*n_individuals))
    if n_individuals == 1:
        axes = [axes]

    for idx, person_id in enumerate(individuals):
        ax = axes[idx]
        person_daily = daily_df[daily_df['id'] == person_id].copy()
        person_visits = visit_df[visit_df['id'] == person_id].copy()

        # Plot parasite density
        positive_density = person_daily[person_daily['parasite_density'] > 0].copy()
        if len(positive_density) > 0:
            ax.plot(positive_density['date'], positive_density['parasite_density'],
                   'o-', color='navy', label='Parasite density',
                   markersize=2, linewidth=0.5, alpha=0.7)

        # Plot gametocyte density
        positive_gam = person_daily[person_daily['gametocyte_density'] > 0].copy()
        if len(positive_gam) > 0:
            ax.plot(positive_gam['date'], positive_gam['gametocyte_density'],
                   'o', color='darkgreen', label='Gametocytes',
                   markersize=4, markeredgewidth=1, markerfacecolor='none', alpha=0.8)

        # Mark treatment events
        treatment_visits = person_visits[person_visits['antimalarial'].str.contains('Artemether', na=False)]
        for _, row in treatment_visits.iterrows():
            ax.axvline(row['date'], color='red', linestyle='--', alpha=0.5, linewidth=1)

        if len(treatment_visits) > 0:
            ax.plot([], [], '--r', linewidth=2, label=f'Treatment ({len(treatment_visits)} events)')

        # Formatting
        ax.set_yscale('log')
        ax.set_ylim([1, 1e6])
        ax.set_ylabel('Parasites/μL', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

        # Title for each panel
        person_info = person_daily.iloc[0]
        n_infections = (person_visits['parasitedensity'] > 0).sum()
        n_visits = len(person_visits)
        ax.set_title(f"ID {person_id} ({int(person_info['age_at_enrollment'])}{person_info['gender'][0]}): " +
                    f"{n_infections}/{n_visits} visits positive",
                    fontsize=10, fontweight='bold')

        if idx == n_individuals - 1:
            ax.set_xlabel('Date', fontsize=10)

    plt.tight_layout()
    filename = figures_dir / 'simulated_household_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

    print("\nVisualization complete!")

if __name__ == '__main__':
    # Run simulation
    individuals = simulate_household()

    # Save results
    save_results(individuals)

    # Create visualizations
    create_visualizations()

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
