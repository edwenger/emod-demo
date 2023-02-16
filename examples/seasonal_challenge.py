import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from emodlib.malaria import IntrahostComponent

from naive_infection import configure_from_file


def surface_area_biting_function(age_days):
    """
    Piecewise linear rising from birth to age 2
    and then shallower slope to age 20
    as in SusceptibilityVector.h
    """

    newborn_risk = 0.07
    two_year_old_risk = 0.23

    if age_days < 2 * 365:
        return newborn_risk + age_days * (two_year_old_risk - newborn_risk) / (2 * 365.)

    if age_days < 20 * 365:
        return two_year_old_risk + (age_days - 2 * 365.) * (1 - two_year_old_risk) / ((20 - 2) * 365)

    return 1.0


def month_index_from_timestep(t):
    y2k = dt.datetime(2000, 1, 1)
    return (y2k + dt.timedelta(days=t)).month - 1


def monthly_eir_challenge(duration, monthly_eirs):

    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)
    fevers = np.zeros(duration)

    ic = IntrahostComponent.create()

    for t in range(duration):

        daily_eir = monthly_eirs[month_index_from_timestep(t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(t)
        p_infected = 1 - np.exp(-daily_eir)
        
        if np.random.random() < p_infected:
            ic.challenge()

        ic.update(dt=1)
        
        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density
        fevers[t] = ic.fever_temperature

    return pd.DataFrame({'days': range(duration),
                         'parasite_density': asexuals,
                         'gametocyte_density': gametocytes,
                         'fever_temperature': fevers}).set_index('days')


def plot_timeseries(df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    df[['parasite_density', 'gametocyte_density']].plot(ax=ax, color=dict(parasite_density='navy', gametocyte_density='darkgreen'))
    ax.set(yscale='log', ylim=(1e-4, 1e5))
    fig.set_tight_layout(True)
    

def plot_heatmap(df, channel='parasite_density', vmin=1e-4, vmax=1e5):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    densities = np.reshape(df[channel].values, (-1, 365))
    ax.imshow(densities, aspect='auto', origin='lower', norm=LogNorm(vmin=vmin, vmax=vmax), interpolation='none')
    ax.set(
        xlabel='day of year',
        ylabel='age (years)',
        title=channel)
    fig.set_tight_layout(True)


if __name__ == '__main__':

    configure_from_file('config.yaml')

    rafin_marke_monthly_eirs = [1, 1, 0.5, 1, 1, 2, 3.875, 7.75, 15.0, 3.875, 1, 1]

    df = monthly_eir_challenge(
        duration=365*20,
        monthly_eirs=rafin_marke_monthly_eirs)

    # plot_timeseries(df)
    plot_heatmap(df)

    plt.show()
    
