import datetime as dt

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from emodlib.malaria import IntrahostComponent, create_config


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


def initialize_intrahost(duration, monthly_eirs, updates_per_day=2, callback=lambda x: None, config=None):

    if config is None:
        config = create_config()

    ic = IntrahostComponent.create(config)
    ic.susceptibility.age = 0.0  # newborn (TODO: address tracking and initialization of non-disease demographic state variables emodlib#10)

    for t in range(duration):

        daily_eir = monthly_eirs[month_index_from_day(t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(t)
        p_infected = 1 - np.exp(-daily_eir)

        if np.random.random() < p_infected:
            ic.challenge()

        for _ in range(updates_per_day):
            ic.update(dt=1.0/updates_per_day)

        callback(ic)

    return ic


def monthly_eir_challenge(duration, monthly_eirs, updates_per_day=2, callback=lambda x: None, ic=None, start_day=0, config=None):

    if config is None:
        config = create_config()

    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)
    fevers = np.zeros(duration)
    infects = np.zeros(duration)
    n_infs = np.zeros(duration)
    treatments = np.zeros(duration)

    if ic is None:
        ic = IntrahostComponent.create(config)
        ic.susceptibility.age = 0.0  # newborn (TODO: address tracking and initialization of non-disease demographic state variables emodlib#10)

    for t in range(duration):

        daily_eir = monthly_eirs[month_index_from_day(start_day + t)] * 12 / 365.0
        daily_eir *= surface_area_biting_function(ic.susceptibility.age)
        p_infected = 1 - np.exp(-daily_eir)

        if np.random.random() < p_infected:
            ic.challenge()

        for _ in range(updates_per_day):
            ic.update(dt=1.0/updates_per_day)

        treatments[t] = callback(ic)

        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density
        fevers[t] = ic.fever_temperature
        infects[t] = ic.infectiousness
        n_infs[t] = ic.n_infections

    return pd.DataFrame({'days': range(duration),
                         'parasite_density': asexuals,
                         'gametocyte_density': gametocytes,
                         'fever_temperature': fevers,
                         'treatments': treatments,
                         'infectiousness': infects,
                         'n_infections': n_infs}).set_index('days')


def multiple_challenges(n_people, duration, monthly_eirs, config=None):

    if config is None:
        config = create_config()

    da = xr.DataArray(dims=('individual', 'time', 'channel'),
                      coords=(range(n_people), pd.date_range('2000-01-01', freq='D', periods=duration), ['parasite_density']))

    for individual in range(n_people):

        df = monthly_eir_challenge(duration=duration,
                                   monthly_eirs=monthly_eirs,
                                   config=config)

        da.loc[dict(individual=individual, channel='parasite_density')] = df.parasite_density.values

    return da


def plot_timeseries(df, monthly_eirs=None, start_day=0):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    df[['parasite_density', 'gametocyte_density']].plot(ax=axs[0], color=dict(parasite_density='navy', gametocyte_density='darkgreen'))
    axs[0].plot(df.index, df.treatments * 1e5, ':r', label='treatments')
    axs[0].set(yscale='log', ylim=(1e-4, 1e5))
    df.n_infections.plot(ax=axs[1])
    axs[1].set(ylabel='n_infections')
    # plot gray background, darkness of which corresponds to monthly EIRs, on axs[1] looping every 365/12 days until end of xaxis
    def month_index_from_day(t):
        y2k = dt.datetime(2000, 1, 1)
        return (y2k + dt.timedelta(days=t)).month - 1
    if monthly_eirs is not None:
        for t in range(0, len(df), 365//12):
            month = month_index_from_day(t+start_day)
            eir = monthly_eirs[month]
            axs[1].axvspan(t, min(t + 365//12, len(df)), color='gray', alpha=0.1 + 0.5 * np.log10(eir) / max(np.log10(monthly_eirs)))
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


class ComplexTreatment:
    """
    Same behavior as EMOD's accounting of unique clinical episodes
    """

    Clinical_Fever_Threshold_High = 38.5
    Clinical_Fever_Threshold_Low = 37.5
    Min_Days_Between_Clinical_Incidents = 14

    def __init__(self, prob_treat):
        self.symptomatic = False
        self.days_since_incident = 0
        self.prob_treat = prob_treat

    def __call__(self, ic):
        fever = ic.fever_temperature

        if self.symptomatic:
            if fever > self.Clinical_Fever_Threshold_Low:
                self.days_since_incident = 0
            else:
                self.days_since_incident += 1
            if self.days_since_incident > self.Min_Days_Between_Clinical_Incidents:
                self.symptomatic = False

        elif fever > self.Clinical_Fever_Threshold_High:
            self.symptomatic = True
            if np.random.random() < self.prob_treat:
                ic.treat()
                return True

        return False


if __name__ == '__main__':

    rafin_marke_monthly_eirs = [1, 1, 0.5, 1, 1, 2, 3.875, 7.75, 15.0, 3.875, 1, 1]

    df = monthly_eir_challenge(
        duration=365*20,
        monthly_eirs=rafin_marke_monthly_eirs)

    # plot_timeseries(df)
    plot_heatmap(df)

    plt.show()
