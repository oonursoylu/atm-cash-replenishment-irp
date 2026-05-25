# ATM Cash Replenishment via Inventory Routing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CPLEX](https://img.shields.io/badge/solver-CPLEX%2022.1+-green.svg)](https://www.ibm.com/products/ilog-cplex-optimization-studio)
[![XGBoost](https://img.shields.io/badge/forecast-XGBoost%202.0+-orange.svg)](https://xgboost.readthedocs.io/)
![Status: Master's Thesis](https://img.shields.io/badge/status-master's%20thesis-purple.svg)

This repository contains the implementation for a master's thesis on ATM cash
replenishment. The project builds a **predict-then-optimize** pipeline: an
XGBoost quantile forecaster predicts ATM cash demand, and a mixed-integer
inventory routing model turns those forecasts into daily replenishment plans.
The forecast layer is also checked with SHAP-based explainability to inspect
which features drive the point and safety quantile models.

The main goal is not only to reduce stockouts. The project studies the trade-off
between **service quality** and **operational cost** in a realistic cash-in-transit
setting.

## Headline Result

The final 73-day rolling-horizon simulation produced:

| Metric | Value |
|---|---:|
| Service level | 94.65% |
| Steady-state service level, days 14-73 | about 97.0% |
| Stockout events | 121 over 2,263 ATM-days |
| Operational cost | 156,295 TL |
| Reported total cost, including stockout penalty | 519,295 TL |
| Total dispatches | 87 vehicle shifts |
| Total cash loaded | 95,514,012 TL |

The full-window service level is slightly below the 95% lower bound often
reported for deployed ATM operations. Most of the gap comes from an early
catch-up period: days 7-13 create 65 of the 121 stockouts. After day 13, the
system reaches about 97.0% service level.

Detailed results are in `docs/headline_73day_run_results.md`. The raw terminal
output for the final run is kept in `docs/headline_73day_terminal_output.md`.

## What the System Does

The pipeline has three layers.

| Layer | Method | Output |
|---|---|---|
| Forecasting | XGBoost quantile regression | Point demand and safety demand per ATM-day |
| Optimization | Multi-vehicle inventory routing MILP | Daily route and delivery decisions |
| Simulation | 7-day rolling horizon, 1-day execution | Stockouts, dispatches, cash loaded, cost components |
| Explainability | SHAP feature analysis | Forecast feature-importance checks |

The forecasting layer produces two demand values:

- `d_mean`: central demand estimate using `alpha_point = 0.55`
- `d_safety`: safety demand estimate using `alpha_safety = 0.95`

The optimization model uses both. The gap between `d_safety` and `d_mean` becomes
a soft safety floor for residual inventory. This gives the routing model a way
to account for demand uncertainty without turning the problem into a large
stochastic program.

## Study Instance

The evaluated instance has:

- 31 ATMs
- 21 physical locations
- one depot
- a service area on Istanbul's Anatolian side
- 3 vehicles
- 7-day planning horizon
- 73-day test window
- heterogeneous ATM capacity tiers: 250,000 / 400,000 / 500,000 TL
- vehicle capacity: 1,500,000 TL

The final test window covers 8 December 2007 to 25 February 2008. The forecast is
trained once and held fixed during the simulation, which is a deliberate thesis
scope decision rather than a production assumption.

## Method Summary

### Forecasting

The demand model is an aggregated XGBoost quantile forecaster. It uses lag
features, rolling-window features, calendar variables, holiday indicators, and
payday indicators. The model is trained with the pinball loss for two quantiles.

The forecast pipeline includes:

- temporal train/validation/test split
- Optuna hyperparameter search
- conformal calibration of the safety quantile
- per-ATM static bias correction
- rolling adaptive bias correction during test prediction

### Optimization

The replenishment model is a multi-vehicle Inventory Routing Problem solved as a
Mixed-Integer Linear Program with IBM CPLEX.

It includes:

- vehicle capacity constraints
- ATM inventory balance
- location-level routing
- shift-time limits
- minimum load per visit
- MTZ subtour elimination
- soft stockout and safety-floor penalties
- an end-of-horizon inventory target for rolling-horizon stability

### Evaluation

The thesis evaluates systems on two separate axes:

- operational cost: travel, dispatch, drop fees, and holding cost
- stockout count: number of ATM-days where demand exceeds available cash

These are kept separate because the stockout penalty is a modelling parameter,
not a directly measured cash-in-transit cost.

### Explainability

SHAP is used to inspect the trained XGBoost forecast models. The analysis checks
whether the point and safety quantile models rely on sensible demand signals such
as recent lags, calendar effects, payday effects, and volatility features. This
is used as an explainability check for the forecasting layer, not as a separate
decision model.

## Baseline Comparison

The final system is compared with three baselines that remove one component at a
time.

| System | Forecast input | Routing / policy | Stockouts | Service level | Operational cost |
|---|---|---|---:|---:|---:|
| B0 static `(s,S)` | historical mean | greedy routing | 52 | 97.70% | 229,355 TL |
| B1 quantile + greedy | `d_mean + d_safety` | threshold + greedy routing | 25 | 98.90% | 215,726 TL |
| B2 point + IRP | `d_mean` only | IRP MILP | 262 | 88.42% | 136,101 TL |
| Proposed system | `d_mean + d_safety` | IRP MILP | 121 | 94.65% | 156,295 TL |

The proposed system is not the highest-service policy. The greedy quantile
baseline has fewer stockouts, but it also carries much higher operating cost.
The proposed system is the cost-disciplined IRP policy with probabilistic
forecast protection. This is why the result is read as a trade-off, not as a
single ranking.

## How to Run

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install IBM CPLEX

The optimization layer requires IBM CPLEX 22.1+ through the `docplex` Python API.
A valid IBM CPLEX license is required. For this thesis project, CPLEX was used
through academic access available to university students. After installing CPLEX,
check that it is visible to Python:

```bash
python -c "from docplex.mp.model import Model; print('CPLEX ready')"
```

### 3. Train forecast models

```bash
python train_models.py --config configs/forecast.yaml --forecast-version v8
```

This creates trained model artifacts in `models/` and prediction CSVs in
`predictions/`.

### 4. Run the rolling-horizon simulation

```bash
python main.py
```

The simulation reads `configs/optimize.yaml`, solves one 7-day MILP per simulated
day, executes the first-day decision, and rolls forward until the 73-day window
is complete.

## Data and Artifacts

Some files are intentionally not included in version control:

- source data in `data/`
- trained model artifacts in `models/*.joblib`
- prediction CSVs in `predictions/*.csv`
- generated maps and large outputs in `outputs/`

This keeps the public repository focused on code, configuration, reproducible
experiment scripts, and documented results. The full pipeline can be regenerated
locally when the required data and solver are available.

Selected result files are kept under `docs/`:

- `docs/headline_73day_run_results.md`
- `docs/headline_73day_terminal_output.md`
- `docs/baseline_results.json`
- `docs/point_irp_results.json`
- `docs/phase_*_raw.json`

SHAP summary outputs are kept in `outputs/shap/`.

## Repository Structure

```text
ma_2026_project/
|-- main.py                         # Rolling-horizon simulation entry point
|-- train_models.py                 # Forecast training pipeline
|-- configs/                        # Forecast and optimization configs
|-- src/
|   |-- data/                       # Spatial, demand, and travel-time modules
|   |-- forecast/                   # XGBoost forecast implementations
|   |-- optim/                      # MILP model and CPLEX solve logic
|   |-- sim/                        # Rolling-horizon simulation
|   `-- viz/                        # Map generation
|-- scripts/                        # Experiment and ablation scripts
|-- tests/                          # Data-layer and integration checks
|-- docs/                           # Public result summaries and raw grids
|-- models/                         # Trained artifacts, not tracked
|-- predictions/                    # Prediction CSVs, not tracked
`-- outputs/                        # Generated outputs, mostly not tracked
```

## Academic Scope and Limitations

This is a research implementation, not a deployed replenishment product.

Important limitations:

- one service area and one ATM network instance
- static forecast model during the 73-day test window
- sequential calibration of forecast and optimization parameters
- binary stockout count, not stockout magnitude
- limited multi-seed replication
- SHAP analysis at aggregated-model level only

These choices keep the thesis computationally tractable. They also point to
clear future work: periodic retraining, joint forecast-IRP calibration,
magnitude-aware stockout modelling, per-ATM interpretability, and cross-instance
testing.

## Research Transparency

A short transparency note on auxiliary AI-assisted coding and writing support is
included in `docs/ai_usage_disclosure.md`. The experiment design, local runs,
result interpretation, and thesis responsibility remain with the author.

## Key References

- Bertsimas and Kallus (2020), Sadana et al. (2024): contextual optimization and predict-then-optimize
- Arrow et al. (1951), Trapero et al. (2019): critical-fractile logic and quantile-based inventory forecasting
- Bertazzi and Speranza (2013), Coelho et al. (2014): inventory routing formulation and survey literature
- Simutis et al. (2008), Venkatesh et al. (2014), Ekinci et al. (2015): ATM cash-demand forecasting and replenishment context
- van Anholt et al. (2016), Jia et al. (2025): related ATM/online inventory-routing studies
- Lundberg and Lee (2017): SHAP feature attribution

## License

This work is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 oonursoylu.
