from functools import partial

import numpy as np
import optuna

from seasonal_challenge import multiple_challenges
from emodlib.malaria import IntrahostComponent


def kl(p, q):

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p*np.log(p/q), 0))


def objective(trial, input_eirs, prev_ref, n_people=5, duration=20*365):

    antigen_switch_rate = trial.suggest_float("Antigen_Switch_Rate", 5e-10, 5e-8, log=True)
 
    IntrahostComponent.set_params(dict(infection_params=dict(Antigen_Switch_Rate=antigen_switch_rate)))

    da = multiple_challenges(n_people=n_people, duration=duration, monthly_eirs=input_eirs)

    prev_by_season = (da.loc[dict(channel='parasite_density')] > 16).resample(time='3M').mean()
    avg_prev_by_age = prev_by_season.groupby('time.year').mean().mean(dim='individual')

    n_ages = min(len(prev_ref), len(avg_prev_by_age))
    
    return kl(prev_ref[:n_ages], avg_prev_by_age[:n_ages])


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

    prevalence_calibration = partial(objective, input_eirs=example_EIRs, prev_ref=example_prev_by_age_ref)

    study = optuna.create_study(study_name='minimize_KL', direction='minimize')
    
    study.optimize(prevalence_calibration, n_trials=30)
