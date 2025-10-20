"""
Incidence of Non-Routine Malaria-Positive Events by Age Group (Broad Age Bins)

Same analysis as 03_incidence_by_age.py but with broader age bins:
- 0-5 years
- 5-10 years
- 10+ years

This provides less noisy temporal signals for easier visualization of trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')

# Load cleaned data
print("Loading cleaned PRISM data...")
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'

df = pd.read_csv(f'data/prism_cleaned_{SITE}.csv', parse_dates=['date'])
print(f"Loaded {len(df)} observations for {df['id'].nunique()} participants")

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("CALCULATING AGE-SPECIFIC INCIDENCE (BROAD AGE BINS)")
print("=" * 80)

# Define broad age bins
age_bins = [0, 5, 10, 100]
age_labels = ['0-5', '5-10', '10+']

# Filter to non-routine, malaria-positive visits (clinical malaria episodes)
clinical_malaria = df[
    (df['visittype'] == 'Unscheduled visit') &
    (df['parasitedensity'] > 0)
].copy()

print(f"\nClinical malaria episodes (unscheduled + parasite+): {len(clinical_malaria)}")

# ============================================================================
# Calculate person-time at risk by age group and time period
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING PERSON-TIME AT RISK")
print("=" * 80)

# Get overall date range
min_date = df['date'].min()
max_date = df['date'].max()
print(f"\nStudy period: {min_date.date()} to {max_date.date()}")

# Create monthly periods
date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
print(f"Number of months: {len(date_range)}")

# For each individual, calculate their contribution to person-time
person_time_data = []

for participant_id in df['id'].unique():
    person_data = df[df['id'] == participant_id].sort_values('date').copy()

    if len(person_data) == 0:
        continue

    first_date = person_data['date'].min()
    last_date = person_data['date'].max()
    first_age = person_data.iloc[0]['age']

    # For each month, calculate age and assign to appropriate age bin
    for i in range(len(date_range) - 1):
        month_start = date_range[i]
        month_end = date_range[i + 1]

        # Check if person is under observation during this month
        if month_start > last_date or month_end < first_date:
            continue

        # Calculate age at this month
        days_since_first = (month_start - first_date).days
        age_at_month = first_age + days_since_first / 365.25

        # Assign to age bin
        age_bin_idx = np.digitize([age_at_month], age_bins)[0] - 1
        if age_bin_idx < 0 or age_bin_idx >= len(age_labels):
            continue

        age_label = age_labels[age_bin_idx]

        # Calculate person-time contribution (in months)
        overlap_start = max(month_start, first_date)
        overlap_end = min(month_end, last_date)

        if overlap_start < overlap_end:
            person_months = (overlap_end - overlap_start).days / 30.44

            person_time_data.append({
                'participant_id': participant_id,
                'year_month': month_start,
                'age_bin': age_label,
                'person_months': person_months,
                'month': month_start.month
            })

person_time_df = pd.DataFrame(person_time_data)
print(f"\nTotal person-months at risk: {person_time_df['person_months'].sum():.1f}")

# ============================================================================
# Count clinical malaria events
# ============================================================================

print("\n" + "=" * 80)
print("COUNTING CLINICAL MALARIA EVENTS")
print("=" * 80)

# Assign age bins to clinical events
clinical_malaria['age_bin'] = pd.cut(clinical_malaria['age'], bins=age_bins, labels=age_labels)
clinical_malaria['year_month'] = clinical_malaria['date'].dt.to_period('M').dt.to_timestamp()
clinical_malaria['month'] = clinical_malaria['date'].dt.month

# Count events by age bin and month
events_by_month_age = clinical_malaria.groupby(['year_month', 'age_bin']).size().reset_index(name='events')

print(f"\nClinical events by age group:")
events_by_age = clinical_malaria.groupby('age_bin').size()
for age_label in age_labels:
    count = events_by_age.get(age_label, 0)
    print(f"  {age_label}: {count}")

# ============================================================================
# Calculate incidence rates
# ============================================================================

print("\n" + "=" * 80)
print("CALCULATING INCIDENCE RATES")
print("=" * 80)

# Aggregate person-time by month and age
person_time_monthly = person_time_df.groupby(['year_month', 'age_bin'])['person_months'].sum().reset_index()

# Merge events with person-time
incidence_df = person_time_monthly.merge(
    events_by_month_age,
    on=['year_month', 'age_bin'],
    how='left'
)
incidence_df['events'] = incidence_df['events'].fillna(0)
incidence_df['incidence'] = incidence_df['events'] / incidence_df['person_months']
incidence_df['month'] = incidence_df['year_month'].dt.month

print(f"\nMean incidence by age group (events per person-month):")
mean_incidence = incidence_df.groupby('age_bin')['incidence'].mean()
for age_label in age_labels:
    rate = mean_incidence.get(age_label, 0)
    print(f"  {age_label}: {rate:.4f} ({rate*12:.2f} per person-year)")

# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Plot incidence over time for each age group (broader bins = less noise)
colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green

for idx, age_label in enumerate(age_labels):
    age_data = incidence_df[incidence_df['age_bin'] == age_label].sort_values('year_month')

    if len(age_data) > 0:
        axes[0].plot(age_data['year_month'], age_data['incidence'],
                    label=f'{age_label} yr', linewidth=2.5, color=colors[idx], alpha=0.8)

axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Incidence (events per person-month)', fontsize=12)
axes[0].set_title('Clinical Malaria Incidence Over Time by Age Group (Broad Bins)\n(Non-routine visits with positive parasitemia)', fontsize=14)
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(True, alpha=0.3)

# ============================================================================
# Seasonal pattern
# ============================================================================

# Aggregate by month of year and age group
seasonal_person_time = person_time_df.groupby(['month', 'age_bin'])['person_months'].sum().reset_index()
seasonal_events = clinical_malaria.groupby(['month', 'age_bin']).size().reset_index(name='events')

seasonal_incidence = seasonal_person_time.merge(
    seasonal_events,
    on=['month', 'age_bin'],
    how='left'
)
seasonal_incidence['events'] = seasonal_incidence['events'].fillna(0)
seasonal_incidence['incidence'] = seasonal_incidence['events'] / seasonal_incidence['person_months']

# Plot seasonal pattern
for idx, age_label in enumerate(age_labels):
    age_seasonal = seasonal_incidence[seasonal_incidence['age_bin'] == age_label].sort_values('month')

    if len(age_seasonal) > 0:
        axes[1].plot(age_seasonal['month'], age_seasonal['incidence'],
                    label=f'{age_label} yr', linewidth=3, marker='o',
                    markersize=10, color=colors[idx], alpha=0.8)

axes[1].set_xlabel('Month', fontsize=12)
axes[1].set_ylabel('Incidence (events per person-month)', fontsize=12)
axes[1].set_title('Seasonal Pattern of Clinical Malaria by Age Group (Broad Bins)\n(All years combined)', fontsize=14)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '06b_clinical_malaria_incidence_broad_bins.png', dpi=300, bbox_inches='tight')
print(f"\n   Saved: {output_dir / '06b_clinical_malaria_incidence_broad_bins.png'}")
plt.close()

# ============================================================================
# Summary statistics
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nAge-specific incidence rates:")
for age_label in age_labels:
    age_person_time = person_time_df[person_time_df['age_bin'] == age_label]['person_months'].sum()
    age_events = (clinical_malaria['age_bin'] == age_label).sum()

    if age_person_time > 0:
        age_incidence = age_events / age_person_time
        print(f"  {age_label}: {age_incidence:.4f} per person-month ({age_incidence*12:.2f} per person-year)")
        print(f"           {age_events} events / {age_person_time:.1f} person-months")

print("\nSeasonal variation:")
seasonal_summary = seasonal_incidence.groupby('month')['incidence'].mean().sort_values(ascending=False)
print(f"  Peak month: {seasonal_summary.index[0]} ({seasonal_summary.iloc[0]:.4f} per person-month)")
print(f"  Lowest month: {seasonal_summary.index[-1]} ({seasonal_summary.iloc[-1]:.4f} per person-month)")
print(f"  Fold change: {seasonal_summary.iloc[0] / seasonal_summary.iloc[-1]:.2f}x")

print("\n" + "=" * 80)
print("INCIDENCE ANALYSIS COMPLETE (BROAD BINS)")
print("=" * 80)
