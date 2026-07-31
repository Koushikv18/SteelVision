# Dataset Documentation — Explainable Steel Defect Prediction System

This project uses **two datasets** for two different purposes. They are not merged — they measure different things and answer different questions.

---

## 1. `steel_defect_data.csv` (PRIMARY — synthetic)

**Purpose:** Main product. Predicts defect risk *before* production, from process-control parameters an engineer can actually adjust. Powers the FastAPI service, SHAP explanations, and the Streamlit what-if dashboard.

**Rows:** 6,000 synthetic steel batches
**Defect rate:** ~25%

| Column | Type | Description |
|---|---|---|
| batch_id | string | Unique batch identifier |
| carbon_pct | float | Carbon content (%), ideal ~0.40% |
| manganese_pct | float | Manganese content (%), protective element |
| furnace_temp_C | float | Furnace temperature (°C), ideal ~1600°C |
| rolling_speed_mps | float | Rolling speed (m/s) |
| cooling_rate_Cps | float | Cooling rate (°C/s), ideal ~12-15 |
| defect | int (0/1) | Target: whether the batch is defective |

**Generation logic:** Defect probability is a smooth function of deviation from metallurgically-ideal values (U-shaped risk for carbon/temp/cooling — both too high AND too low are bad), plus one explicit interaction term (rolling speed × furnace temp mismatch → surface cracking), plus Gaussian noise. This is intentional: linear correlations with `defect` are deliberately weak (~0.02–0.07) because the real relationships are nonlinear and interaction-driven — which is exactly why XGBoost + SHAP (not logistic regression or plain correlation) are the right tools here.

**Honesty note for README/report:** This is a **synthetic dataset with domain-informed relationships**, not observed plant data. State this explicitly — real defect data tied to process parameters is proprietary to steel manufacturers.

---

## 2. `real_steel_plates_faults.csv` (VALIDATION — real, UCI)

**Purpose:** Secondary validation study. Proves the modeling methodology (XGBoost + SHAP) generalizes to real, independently-collected industrial defect data — not just this project's own synthetic assumptions.

**Source:** UCI Machine Learning Repository — "Steel Plates Faults" dataset (Semeion Research Center, commissioned by Centro Sviluppo Materiali, Italy). Provided as raw files `Faults.NNA` (tab-separated data) + `Faults27x7_var` (column names).

**Rows:** 1,941 real inspected steel plates (all defective — this dataset contains no "good" plates)

| Column group | Description |
|---|---|
| X_Minimum … Y_Maximum | Bounding box of the defect on the plate |
| Pixels_Areas, X/Y_Perimeter | Size/shape of the defect region |
| Sum/Min/Max_of_Luminosity | Luminosity (brightness) statistics of the defect |
| Length_of_Conveyer, TypeOfSteel_A300/A400, Steel_Plate_Thickness | Plate/process metadata |
| Edges_Index … SigmoidOfAreas | 13 derived geometric indices |
| Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults | One-hot columns for the 7 defect types |
| fault_class | Single readable label = whichever of the 7 one-hot columns is 1 |

**Key difference from the synthetic dataset:** This is **image-derived, post-production** defect classification (what type of defect is on this already-faulty plate?), not **pre-production** process-parameter prediction (will this batch fail?). The task here is naturally **multi-class classification** (7 fault types), not binary pass/fail.

**How to use it in your project:** Train XGBoost to classify `fault_class` from the 27 geometric/luminosity features, apply SHAP the same way, and present it as: *"To validate that the explainability methodology holds on real, independently-collected industrial data, the same XGBoost + SHAP pipeline was applied to the UCI Steel Plates Faults dataset."* This is a genuinely strong addition — real data, honestly scoped, clearly distinguished from the synthetic core product.

---

## Quick reference — which dataset for which project step

| Project step | Dataset |
|---|---|
| EDA (Day 3) | Both — separately |
| Primary model (XGBoost pass/fail) | Synthetic |
| Validation model (XGBoost multi-class) | Real (UCI) |
| SHAP explanations | Both |
| FastAPI `/predict`, `/predict/batch` | Synthetic only (this is the live product) |
| Streamlit what-if sliders | Synthetic only (needs process parameters engineer controls) |
| Streaming simulator | Synthetic only |
