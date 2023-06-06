import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback

from seasonal_challenge import multiple_challenges
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


def objective(trial, input_eirs, prev_ref, n_people=5, duration=20*365):

    antigen_switch_rate = trial.suggest_float("Antigen_Switch_Rate", 5e-10, 5e-8, log=True)
    mlflow.log_param("Antigen_Switch_Rate", antigen_switch_rate)

    IntrahostComponent.set_params(dict(infection_params=dict(Antigen_Switch_Rate=antigen_switch_rate)))

    da = multiple_challenges(n_people=n_people, duration=duration, monthly_eirs=input_eirs)

    prev_by_season = (da.loc[dict(channel='parasite_density')] > 16).resample(time='3M').mean()
    avg_prev_by_age = prev_by_season.groupby('time.year').mean().mean(dim='individual')

    n_ages = min(len(prev_ref), len(avg_prev_by_age))
    
    fig = plot_comparison(prev_ref[:n_ages], avg_prev_by_age[:n_ages])
    mlflow.log_figure(fig, 'prev_compare.png')

    kl_div = kl(prev_ref[:n_ages], avg_prev_by_age[:n_ages])
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

    study_name = "minimize_KL"  # unique identifier
    storage_name = "sqlite:///optuna_{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction='minimize')

    study.optimize(prevalence_calibration, n_trials=20, callbacks=[mlflc])
