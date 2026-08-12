"""
cost_model.py
----------------
Translates the tuned XGBoost model's confusion matrix into an estimated
business cost impact. This is the "so what?" step -- turning classifier
metrics into a number a non-technical stakeholder actually cares about.

Cost basis:
- $147/ton rework-cost reduction from early defect detection, per the
  AISC 2022 report on dimensional-check inspection programs (REAL,
  citable industry figure).
- Typical batch tonnage: ASSUMED at 150 tons (a realistic EAF/BOF batch
  size) since no public source gives an exact per-batch figure for this
  hypothetical scenario. Clearly labeled as an assumption below.
- False-alarm (false positive) inspection cost: ASSUMED at $500/batch
  (a modest manual inspection labor cost) -- no public source found for
  this specific figure either.

Both assumed figures are ISOLATED into named constants at the top of
this file specifically so they can be swapped for better numbers later
without touching any of the calculation logic.
"""

import os
import sys
import joblib
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "utils")))

from model_utils import load_data, split_data, MODEL_DIR

# ---------------------------------------------------------------------
# Cost assumptions -- clearly separated: researched vs. assumed
# ---------------------------------------------------------------------
REWORK_COST_PER_TON = 147          # RESEARCHED: AISC 2022 report
ASSUMED_BATCH_TONNAGE = 150         # ASSUMED: typical EAF/BOF batch size
FALSE_ALARM_COST_PER_BATCH = 500     # ASSUMED: manual inspection labor cost

MISSED_DEFECT_COST = REWORK_COST_PER_TON * ASSUMED_BATCH_TONNAGE  # $22,050/batch


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = joblib.load(f"{MODEL_DIR}/xgb_model_tuned.pkl")
    y_pred = model.predict(X_test)

    # confusion_matrix returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"True positives (caught defects)  : {tp}")
    print(f"False negatives (missed defects) : {fn}")
    print(f"False positives (false alarms)   : {fp}")
    print(f"True negatives (correctly safe)  : {tn}")

    # ---- Cost WITH the model ----
    missed_defect_cost = fn * MISSED_DEFECT_COST
    false_alarm_cost = fp * FALSE_ALARM_COST_PER_BATCH
    total_cost_with_model = missed_defect_cost + false_alarm_cost

    # ---- Cost WITHOUT any model (ship everything blindly) ----
    # every real defect (tp + fn) would go undetected and cost full
    # rework price, with zero false-alarm cost since no inspections happen
    total_real_defects = tp + fn
    total_cost_without_model = total_real_defects * MISSED_DEFECT_COST

    savings = total_cost_without_model - total_cost_with_model
    pct_reduction = (savings / total_cost_without_model) * 100

    print(f"\n--- Cost impact (assumed {ASSUMED_BATCH_TONNAGE}-ton batches) ---")
    print(f"Missed-defect cost (model)   : ${missed_defect_cost:,.0f}")
    print(f"False-alarm cost (model)     : ${false_alarm_cost:,.0f}")
    print(f"Total cost WITH model        : ${total_cost_with_model:,.0f}")
    print(f"Total cost WITHOUT model     : ${total_cost_without_model:,.0f}")
    print(f"Estimated savings            : ${savings:,.0f}")
    print(f"Percent cost reduction       : {pct_reduction:.1f}%")

    print("\nNOTE: figures based on a real AISC industry rework-cost rate")
    print("($147/ton) combined with an ASSUMED batch tonnage (150 tons)")
    print("and an ASSUMED false-alarm inspection cost ($500/batch).")
    print("Label these assumptions clearly in any report using this output.")


if __name__ == "__main__":
    main()