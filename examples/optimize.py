import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
import pandas as pd
import xarray as xr

from seasonal_challenge import monthly_eir_challenge  #, multiple_challenges
from emodlib.malaria import IntrahostComponent


def kl(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p*np.log(p/q), 0))


def plot_comparison(ref, sim):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set(ylim=(0, 1), ylabel='prevalence', xlabel='age (years)')
    ages = range(len(ref))
    ax.plot(ages, sim, '-')
    ax.plot(ages, ref, 'o')
    fig.set_tight_layout(True)
    return fig


def eval(ref, sim, plot=False):

    prev_by_season = (sim.loc[dict(channel='parasite_density')] > 16).resample(time='3M').mean()
    avg_prev_by_age = prev_by_season.groupby('time.year').mean().mean(dim='individual')
    n_ages = min(len(ref), len(avg_prev_by_age))

    if plot:
        fig = plot_comparison(ref[:n_ages], avg_prev_by_age[:n_ages])
        mlflow.log_figure(fig, 'prev_compare.png')

    return kl(ref[:n_ages], avg_prev_by_age[:n_ages])


def objective(trial, input_eirs, prev_ref, n_people=10, duration=20*365):

    antigen_switch_rate = trial.suggest_float("Antigen_Switch_Rate", 5e-10, 5e-8, log=True)
    mlflow.log_param("Antigen_Switch_Rate", antigen_switch_rate)

    falciparum_pfemp1_variants = trial.suggest_int("Falciparum_PfEMP1_Variants", 500, 1200)
    mlflow.log_param("Falciparum_PfEMP1_Variants", falciparum_pfemp1_variants)

    max_individual_infections = trial.suggest_int("Max_Individual_Infections", low=3, high=7)
    mlflow.log_param("Max_Individual_Infections", max_individual_infections)

    IntrahostComponent.set_params(dict(infection_params=dict(Antigen_Switch_Rate=antigen_switch_rate),
                                       Falciparum_PfEMP1_Variants=falciparum_pfemp1_variants,
                                       Max_Individual_Infections=max_individual_infections))

    # da = multiple_challenges(n_people=n_people, duration=duration, monthly_eirs=input_eirs)

    ### refactoring to explore optuna.pruners behavior

    da = xr.DataArray(dims=('individual', 'time', 'channel'),
                      coords=(range(n_people), pd.date_range('2000-01-01', freq='D', periods=duration), ['parasite_density']))        

    for individual in range(n_people):
    
        df = monthly_eir_challenge(duration=duration,
                                   monthly_eirs=input_eirs)

        da.loc[dict(individual=individual, channel='parasite_density')] = df.parasite_density.values

        if (individual % 2) == 0:
            intermediate_value = eval(prev_ref, da.where(da.individual <= individual, drop=True))
            # print(trial._trial_id, intermediate_value, individual)
            trial.report(intermediate_value, individual)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    kl_div = eval(prev_ref, da, plot=True)
    mlflow.log_metric("kl_div", kl_div)

    return kl_div


if __name__ == '__main__':
 
    example_EIRs = [
        1, 1, 1,
        1, 1, 2,
        4, 8, 16,
        4, 1, 1]

    example_prev_by_age_ref = [
        0.2, 0.4, 0.7, 0.9, 0.85,
        0.85, 0.8, 0.8, 0.75, 0.75,
        0.7, 0.65, 0.6, 0.5, 0.45,
        0.4, 0.35, 0.3, 0.25, 0.25,
        0.2, 0.2, 0.2, 0.2, 0.2]

    mlflc = MLflowCallback(
        # tracking_uri='./mlruns',
        metric_name="my metric score")

    @mlflc.track_in_mlflow()
    def prevalence_calibration(trial):
        return objective(trial, input_eirs=example_EIRs, prev_ref=example_prev_by_age_ref)

    study_name = "prune10x2_3d_200"  # unique identifier
    storage_name = "sqlite:///optuna_prev_calib.db"
    study = optuna.create_study(study_name=study_name, storage=storage_name,
                                pruner=optuna.pruners.MedianPruner(),
                                direction='minimize')

    study.optimize(prevalence_calibration, n_trials=200, callbacks=[mlflc])
