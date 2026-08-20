An ML system that predicts whether a steel batch will be defective based on process parameters, and explains *why* using SHAP.

## Problem

In steel manufacturing, defects (cracks, porosity, scale) trace back to process conditions during production. Predicting defect risk early and knowing which factor is driving it helps engineers catch bad batches before shipping and fix the process, not just react after the fact.

## Current state

What actually exists in this repo today, end to end:

- `src/data generations.ipynb` — generates the synthetic `Data/syntheticData.csv` (6,000 batches, 5 process features, U-shaped + interaction risk logic, ~26% defect rate).
- `src/eda.ipynb` — exploratory analysis of the synthetic data.
- `src/train_model.py` — trains a Logistic Regression baseline and an XGBoost classifier, saves both plus the scaler to `models/`.
- `src/tune_xgboost.py` — grid-searches XGBoost hyperparameters (with `scale_pos_weight` for class imbalance) and saves the tuned model.
- `src/cost_model.py` — converts the tuned model's confusion matrix into an estimated dollar cost/savings figure, using one researched industry rate and two explicitly labeled assumptions.
- `src/explain_shap.py` — generates global (summary plot) and single-prediction (force plot) SHAP explanations into `outputs_shap/`.
- `run_pipeline.py` — runs training → tuning → cost model → SHAP end to end in one command.
- `Data/realData.csv`, `Data/Faults.NNA`, `Data/Faults27x7_var` — the UCI "Steel Plates Faults" dataset, present but not yet wired into any script (see below).

There is **no FastAPI service, no Streamlit dashboard, and no inference/serving layer yet** — the pipeline is currently training + explainability only, run locally via `run_pipeline.py`. Any earlier reference to those in project docs described the target end state, not what's implemented.

## Scope of improvement

Rough priority order:

**High priority**
- Build the serving layer the docs promise: a FastAPI `/predict` (and `/predict/batch`) endpoint around the tuned model + scaler, and a Streamlit "what-if" dashboard for engineers to move the sliders and see risk/SHAP update live.
- Add `requirements.txt` (or `pyproject.toml`) pinning `scikit-learn`, `xgboost`, `shap`, `pandas`, `joblib`, `matplotlib` — there's currently no dependency manifest, so the project can't be set up reproducibly from a fresh clone.
- Add a `.gitignore` (`__pycache__/`, `*.pyc`, `.venv/`)  a compiled `utils/__pycache__/model_utils.cpython-313.pyc` is currently committed to the repo.

**Medium priority**
- Wire up the UCI real-data validation study described in `Data/DATA_README.md`: a script that trains XGBoost on `Faults.NNA`/`Faults27x7_var` for multi-class fault-type classification and applies SHAP, to show the methodology generalizes beyond the synthetic set.
- Add a test suite (`pytest`) covering `utils/model_utils.py` (data loading/splitting) and the cost model's arithmetic — there are currently no automated tests.
- Add a CI workflow (GitHub Actions) to run linting/tests on push, and consider a `pre-commit` config to strip notebook outputs before commit (`eda.ipynb` currently carries ~400KB of committed output/plots).
- Remove the unused `Test_size`/`Random_state` module-level constants in `train_model.py` (dead code — the functions already use `RANDOM_STATE`/`TEST_SIZE` imported from `model_utils.py`).
- Track experiments (params, metrics, model artifacts) with something like MLflow instead of only console `print` output from `evaluate()`, so tuning runs are comparable over time.

**Lower priority / polish**
- Add type hints and docstrings across `utils/model_utils.py` and the `src/` scripts.
- Rename `src/data generations.ipynb` to remove the space (`data_generation.ipynb`) for shell/tooling friendliness.
- Add a `LICENSE` file.
- Expand `README.md` with setup instructions (`pip install -r requirements.txt`, `python run_pipeline.py`) and sample output once the dependency manifest exists.
