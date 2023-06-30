## EMOD Demo

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/edwenger/emod-demo)

This [`emod-demo`](https://github.com/edwenger/emod-demo) repository is intended to demonstrate how the [`emodlib`](https://github.com/edwenger/emodlib) python package can be used for realistic example scenarios.  

The [`emodlib`](https://github.com/edwenger/emodlib) package exports [`EMOD`](https://github.com/InstituteforDiseaseModeling/EMOD)'s malaria within-host model logic from C++ to Python via [`pybind11`](https://github.com/pybind/pybind11). 

Click the badge above to create a codespace for this repository and try running some of the notebooks in the [`examples/`](https://github.com/edwenger/emod-demo/tree/master/examples) directory.

Or fork this template repository to start with the same `requirements.txt` dependencies and `.devcontainer.json` environment settings as you develop your own analysis projects.

### Examples

[`examples/compare_timesteps.ipynb`](https://github.com/edwenger/emod-demo/blob/master/examples/compare_timesteps.ipynb): 
Explore time constants of different within-host processes (parasite density, antibodies, cytokines) and implication for fidelity of model components at different time-step sizes

[`examples/infectious_reservoir.ipynb`](https://github.com/edwenger/emod-demo/blob/master/examples/infectious_reservoir.ipynb):
Investigate parasite densities + infectiousness under repeat exposure with different assumptions of transmission intensity, seasonality, and symptomatic treatment.  Characterize the composition of the human infectious reservoir as a function of age-dependent immunity.

[`examples/interactive_widgets.ipynb`](https://github.com/edwenger/emod-demo/blob/master/examples/interactive_widgets.ipynb):
Construct an interactive dashboard with [`ipywidgets`](https://github.com/jupyter-widgets/ipywidgets) to visualize infection state with controls to step forward in time, to add new infections, and to deliver treatment.

[`examples/calibrate_prevalence.ipynb`](https://github.com/edwenger/emod-demo/blob/master/examples/calibrate_prevalence.ipynb):
Use the [`optuna`](https://github.com/optuna/optuna) hyperparameter optimization library to calibrate a seasonal exposure simulation to dummy age-dependent prevalence data.

[`examples/optimize.py`](https://github.com/edwenger/emod-demo/blob/master/examples/optimize.py):
Extends the model calibration example to include additional [`optuna-dashboard`](https://github.com/optuna/optuna-dashboard) or [`mlflow`](https://github.com/mlflow/mlflow/) hooks for experiment tracking, artifact storage, and visualization.

[`examples/simple_transmission.ipynb`](https://github.com/edwenger/emod-demo/blob/master/examples/simple_transmission.ipynb):
Pass infections between individual hosts using `numpy.array` as a simple + flexible infectiousness container and proxy for mosquito dynamics.
