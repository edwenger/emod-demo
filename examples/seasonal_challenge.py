import datetime as dt

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from emodlib.malaria import IntrahostComponent


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


def month_index_from_day(t):
    y2k = dt.datetime(2000, 1, 1)
    return (y2k + dt.timedelta(days=t)).month - 1


def monthly_eir_challenge(duration, monthly_eirs, updates_per_day=2, callback=lambda x: None):

    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)
    fevers = np.zeros(duration)
    infects = np.zeros(duration)
    n_infs = np.zeros(duration)

    ic = IntrahostComponent.create()

    for t in range(duration):

        daily_eir = monthly_eirs[month_index_from_day(t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(t)
        p_infected = 1 - np.exp(-daily_eir)
        
        if np.random.random() < p_infected:
            ic.challenge()

        for _ in range(updates_per_day):
            ic.update(dt=1.0/updates_per_day)
        
        callback(ic)

        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density
        fevers[t] = ic.fever_temperature
        infects[t] = ic.infectiousness
        n_infs[t] = ic.n_infections

    return pd.DataFrame({'days': range(duration),
                         'parasite_density': asexuals,
                         'gametocyte_density': gametocytes,
                         'fever_temperature': fevers,
                         'infectiousness': infects,
                         'n_infections': n_infs}).set_index('days')


def multiple_challenges(n_people, duration, monthly_eirs):

    da = xr.DataArray(dims=('individual', 'time', 'channel'),
                      coords=(range(n_people), pd.date_range('2000-01-01', freq='D', periods=duration), ['parasite_density']))
    
    for individual in range(n_people):
    
        df = monthly_eir_challenge(duration=duration,
                                   monthly_eirs=monthly_eirs)

        da.loc[dict(individual=individual, channel='parasite_density')] = df.parasite_density.values

    return da


def plot_timeseries(df):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    df[['parasite_density', 'gametocyte_density']].plot(ax=axs[0], color=dict(parasite_density='navy', gametocyte_density='darkgreen'))
    axs[0].set(yscale='log', ylim=(1e-4, 1e5))
    df.n_infections.plot(ax=axs[1])
    axs[1].set(ylabel='n_infections')
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

    rafin_marke_monthly_eirs = [1, 1, 0.5, 1, 1, 2, 3.875, 7.75, 15.0, 3.875, 1, 1]

    df = monthly_eir_challenge(
        duration=365*20,
        monthly_eirs=rafin_marke_monthly_eirs)

    # plot_timeseries(df)
    plot_heatmap(df)

    plt.show()
    
