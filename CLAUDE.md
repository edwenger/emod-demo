# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demonstration repository for the `emodlib` Python package, which exports EMOD's malaria within-host model logic from C++ to Python via pybind11. The repository contains Jupyter notebooks and Python scripts showing realistic usage scenarios for malaria transmission modeling.

## Dependencies

Install dependencies with:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `emodlib>=0.0.3` (from test.pypi.org)
- Scientific computing: numpy, pandas, xarray, matplotlib
- Optimization: optuna, optuna-dashboard, mlflow
- Interactive notebooks: ipywidgets

## Development Environment

The repository includes a `.devcontainer.json` for GitHub Codespaces with pre-configured VS Code extensions for Jupyter, Python, and SQLite viewing.

## Architecture

### Core Simulation Model

The `emodlib.malaria.IntrahostComponent` is the primary interface to the malaria within-host model:
- Create instances with `IntrahostComponent.create()`
- Set global parameters with `IntrahostComponent.set_params(dict(...))`
- Simulate time steps with `ic.update(dt=...)`
- Add new infections with `ic.challenge()`
- Apply treatment with `ic.treat()`

Key state variables accessed from component instances:
- `ic.parasite_density` - asexual parasite density
- `ic.gametocyte_density` - sexual stage parasite density
- `ic.fever_temperature` - host fever temperature
- `ic.infectiousness` - probability of mosquito infection
- `ic.n_infections` - count of concurrent infections
- `ic.susceptibility.age` - host age (days)

### Simulation Patterns

**seasonal_challenge.py** provides reusable simulation functions:
- `monthly_eir_challenge()` - Runs a single individual through a seasonal exposure scenario, returning a pandas DataFrame with daily output
- `multiple_challenges()` - Runs multiple individuals, returning an xarray DataArray
- `surface_area_biting_function()` - Age-dependent exposure model
- `month_index_from_day()` - Maps simulation day to seasonal month (0-11)
- `ComplexTreatment` class - Implements fever-triggered treatment with EMOD's clinical episode logic

The `monthly_eirs` parameter in these functions is a 12-element list representing monthly EIR (entomological inoculation rate) values.

### Optimization Framework

**optimize.py** demonstrates hyperparameter calibration using optuna:
- Objective function searches 3 key parameters: `Antigen_Switch_Rate`, `Falciparum_PfEMP1_Variants`, `Max_Individual_Infections`
- Uses `optuna.pruners.MedianPruner()` for early stopping of unpromising trials
- Integrates with both optuna-dashboard (artifacts, notes) and mlflow (metrics, figures)
- Evaluates fit using KL divergence between simulated and reference age-prevalence curves

Study artifacts stored in:
- `optuna.db` - SQLite database with trial metadata
- `optuna-artifacts/` - Trial figures and artifacts
- `mlruns.db` - MLflow tracking database
- `mlruns/` - MLflow artifacts directory

### Common Commands

Run Python scripts:
```bash
python examples/optimize.py
python examples/seasonal_challenge.py
```

Launch optimization experiment tracking UIs:
```bash
optuna-dashboard sqlite:///optuna.db --artifact-dir optuna-artifacts
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### Jupyter Notebooks

Located in `examples/`:
- `compare_timesteps.ipynb` - Explore time constants and time-step fidelity
- `infectious_reservoir.ipynb` - Age-structured immunity and infectiousness analysis
- `interactive_widgets.ipynb` - ipywidgets dashboard for infection dynamics
- `calibrate_prevalence.ipynb` - Optuna calibration to prevalence data
- `simple_transmission.ipynb` - Multi-host transmission with numpy arrays as mosquito proxy

## Data Output Patterns

Simulations typically output to:
- pandas DataFrames for single-individual time series (from `monthly_eir_challenge`)
- xarray DataArrays for multi-individual cohorts with dimensions: (individual, time, channel)
- Time coordinates use `pd.date_range('2000-01-01', freq='D', periods=duration)`
- Common channels: 'parasite_density', 'gametocyte_density', 'fever_temperature', 'infectiousness'
