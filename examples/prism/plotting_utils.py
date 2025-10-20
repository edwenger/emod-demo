"""
Plotting utilities for PRISM individual trajectory visualization

Two main functions:
1. plot_individual_detailed() - Full panel plot for a single individual
2. plot_individual_compact() - Single row for multi-individual facet plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from datetime import datetime
import warnings


def plot_individual_detailed(person_data, ax=None, fig=None):
    """
    Detailed plot of individual trajectory with parasite/gametocyte density on log scale.

    Emulates simulation plotting style with:
    - Parasite density (navy line/markers)
    - Gametocyte density (dark green markers)
    - LAMP positive shown as sub-microscopic placeholder (~5 parasites/µL)
    - Malaria fever episodes (treated) as vertical dashed red lines
    - Visit types indicated

    Parameters
    ----------
    person_data : pd.DataFrame
        Data for single individual with columns: date, parasitedensity, gametocytes,
        fever, LAMP, visittype, malaria_diagnosis, antimalarial
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    fig : matplotlib figure, optional
        Figure to plot on. If None, creates new figure.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    person_data = person_data.sort_values('date').copy()
    dates = person_data['date'].values

    # Parasite density (microscopy)
    parasite_density = person_data['parasitedensity'].fillna(0).values

    # Gametocyte presence (treat as binary, show at parasite density level)
    gametocyte_mask = person_data['gametocytes'] == 'Yes'
    gametocyte_density = parasite_density.copy()
    gametocyte_density[~gametocyte_mask] = np.nan

    # LAMP positive (submicroscopic) - show at ~5 parasites/µL
    lamp_positive = (person_data['LAMP'] == 'Positive')
    lamp_density = np.full(len(person_data), np.nan)
    lamp_density[lamp_positive] = 5  # Placeholder submicroscopic level

    # Plot parasite density
    # Replace zeros with small value for log plotting
    plot_parasite = parasite_density.copy()
    plot_parasite[plot_parasite == 0] = np.nan  # Don't show negatives

    ax.plot(dates, plot_parasite, 'o-', color='navy', label='Parasite density',
            markersize=6, linewidth=1.5, alpha=0.7)

    # Plot gametocyte density (where detected)
    if gametocyte_mask.sum() > 0:
        ax.plot(dates[gametocyte_mask], gametocyte_density[gametocyte_mask],
                'o', color='darkgreen', label='Gametocytes',
                markersize=10, markeredgewidth=2, markerfacecolor='none', alpha=0.8)

    # Plot LAMP positive (submicroscopic)
    if lamp_positive.sum() > 0:
        ax.plot(dates[lamp_positive], lamp_density[lamp_positive],
                'd', color='purple', label='LAMP+ (submicroscopic)',
                markersize=8, alpha=0.7)

    # Mark MALARIA fever episodes with vertical lines (fever + malaria diagnosis)
    # These represent treatment events that break continuity of infection progression
    if 'malaria_diagnosis' in person_data.columns:
        malaria_fever_mask = (person_data['fever'] == 'Yes') & (person_data['malaria_diagnosis'] == 'Yes')
    else:
        # Fallback if malaria_diagnosis not available
        malaria_fever_mask = person_data['fever'] == 'Yes'

    malaria_fever_dates = dates[malaria_fever_mask]
    for fever_date in malaria_fever_dates:
        ax.axvline(fever_date, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add dummy line for legend
    if malaria_fever_mask.sum() > 0:
        ax.plot([], [], ':r', linewidth=2, label='Malaria fever (treated)')

    # Formatting
    ax.set_yscale('log')
    ax.set_ylim([1, 1e6])
    ax.set_ylabel('Parasite Density (parasites/µL)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return fig, ax


def plot_individual_compact(df, ax=None, fig=None, detailed=False):
    """
    Compact multi-individual plot with one row per person, emulating Barry2021 style.

    Each individual gets one horizontal row showing:
    - All visits as small gray dots
    - Fever visits as red dots
    - LAMP positive as purple filled circles (submicroscopic)
    - LAMP negative as open gray circles
    - Microscopy positive as colored circles (size and color by density)
    - Gametocytes as olive edge on microscopy positive

    Parameters
    ----------
    df : pd.DataFrame
        Data for multiple individuals with columns: id, date, age, gender,
        parasitedensity, gametocytes, fever, LAMP, visittype
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.
    fig : matplotlib figure, optional
        Figure to plot on. If None, creates new figure.
    detailed : bool, optional
        If True, add text annotations for densities and LAMP results

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    cmap = plt.cm.YlOrRd
    lamp_color = cmap(0.1)

    # Index individuals sorted by age
    df = df.sort_values(by='age').copy()
    unique_ids = df.drop_duplicates(subset=['id'], keep='first').copy()
    unique_ids['idx'] = range(len(unique_ids))
    df = df.merge(unique_ids[['id', 'idx']], on='id', how='outer')

    # Background: all visits (small gray)
    ax.scatter(df['date'].values, df['idx'].values, c='darkgray', s=5, alpha=1, lw=0)

    # Fever visits (red)
    fever = df[df['fever'] == 'Yes']
    ax.scatter(fever['date'].values, fever['idx'].values, c='firebrick', s=5, alpha=1, lw=0)

    # LAMP results (only done on microscopy negatives)
    lamp = df[df['LAMP'].isin(['Positive', 'Negative'])].copy()

    # LAMP negative: open gray circle (upper bound on parasitemia)
    lamp_neg = lamp[lamp['LAMP'] == 'Negative']
    ax.scatter(lamp_neg['date'].values, lamp_neg['idx'].values,
               s=30, alpha=0.5, c='None', edgecolors='darkgray', lw=0.5)
    if detailed:
        for i in range(len(lamp_neg)):
            ax.annotate('l', (lamp_neg['date'].values[i], lamp_neg['idx'].values[i] - 0.25),
                       fontsize=5, alpha=0.5, ha='center')

    # LAMP positive: filled light color (submicroscopic)
    lamp_positive = lamp[lamp['LAMP'] == 'Positive']
    ax.scatter(lamp_positive['date'].values, lamp_positive['idx'].values,
               s=60, alpha=0.25, color=lamp_color, edgecolors='darkgray', lw=0.5)
    if detailed:
        for i in range(len(lamp_positive)):
            ax.annotate('L', (lamp_positive['date'].values[i], lamp_positive['idx'].values[i] - 0.25),
                       fontsize=5, fontweight='bold', alpha=0.8, ha='center')

    # Microscopy results (no LAMP done on these)
    # Note: LAMP can be NaN (missing) or 'Na' (not applicable/not done)
    micro_only = df[(~df['LAMP'].isin(['Positive', 'Negative', 'No result'])) &
                    (df['parasitedensity'].notna())].copy()

    # Microscopy negative: open gray circle
    micro_neg = micro_only[micro_only['parasitedensity'] == 0]
    ax.scatter(micro_neg['date'].values, micro_neg['idx'].values,
               s=60, alpha=0.5, c='None', edgecolors='darkgray', lw=0.5)
    if detailed:
        for i in range(len(micro_neg)):
            ax.annotate('m', (micro_neg['date'].values[i], micro_neg['idx'].values[i] - 0.25),
                       fontsize=5, alpha=0.5, ha='center')

    # Microscopy positive: size and color by density
    parasite_positive = micro_only[micro_only['parasitedensity'] > 0]
    if len(parasite_positive) > 0:
        marker_size = 50 * np.log10(parasite_positive['parasitedensity'])
        marker_size[marker_size < 60] = 60
        marker_color = parasite_positive['parasitedensity'].values

        ax.scatter(parasite_positive['date'].values, parasite_positive['idx'].values,
                  c=marker_color, s=marker_size, cmap=cmap, norm=LogNorm(10, 3e5),
                  alpha=0.25, lw=0.5, edgecolors='darkgray')

        # Gametocyte positive: olive edge
        parasite_positive_with_gam = parasite_positive[parasite_positive['gametocytes'] == 'Yes']
        if len(parasite_positive_with_gam) > 0:
            marker_size_gam = 50 * np.log10(parasite_positive_with_gam['parasitedensity'])
            marker_size_gam[marker_size_gam < 60] = 60
            ax.scatter(parasite_positive_with_gam['date'].values,
                      parasite_positive_with_gam['idx'].values,
                      s=marker_size_gam, alpha=0.5, c='None', edgecolors='olive', lw=1.5)

        # Text annotations if detailed
        if detailed:
            for i, row in parasite_positive.iterrows():
                density = row['parasitedensity']
                # Format from high to low values
                if density >= 1e6:
                    txt = f'{density/1e6:.1f}M'
                elif density >= 1e5:
                    txt = f'{int(density/1e3)}K'
                elif density >= 1e4:
                    txt = f'{int(density/1e3)}K'
                elif density >= 1e3:
                    txt = f'{density/1e3:.1f}K'
                else:
                    txt = f'{int(density)}'

                is_fever = row['fever'] == 'Yes'
                ax.annotate(txt, (row['date'], row['idx'] + 0.25),
                           fontsize=5, fontweight='bold', alpha=0.8, ha='center',
                           color='firebrick' if is_fever else 'k')

    # Y-axis labels: age and gender (subsampled if too many)
    subsample = max(1, len(unique_ids) // 100)
    ax.set_yticks(unique_ids['idx'].values[::subsample])
    ax.set_yticklabels([f"{int(row['age'])} {row['gender'][0]}"
                        for _, row in unique_ids.iterrows()][::subsample])
    ax.set_ylim(df['idx'].min() - 1, df['idx'].max() + 1)
    ax.set_ylabel('Age (years) & Gender', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)

    # Background horizontal lines
    study_start = df['date'].min()
    study_end = df['date'].max()
    for i in range(len(unique_ids)):
        ax.plot([study_start, study_end], [i, i], color='gray', alpha=0.1, lw=0.25)

    # Format x-axis dates with yearly major ticks and monthly minor ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Show ticks on top axis too (no labels)
    ax.tick_params(top=True, labeltop=False, which='both')

    # Apply tight_layout only if figure doesn't have incompatible elements (like button widgets)
    if fig:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*tight_layout.*')
            try:
                fig.tight_layout()
            except:
                # Silently skip tight_layout if it causes issues (e.g., with button widgets)
                pass

    return fig, ax


def human_format(num):
    """Format large numbers in human-readable format"""
    if num >= 1e5:
        return f'{num/1e5:.1f}e5'
    elif num >= 1e4:
        return f'{num/1e3:.0f}K'
    elif num >= 1e3:
        return f'{num/1e3:.1f}K'
    else:
        return f'{num:.0f}'
