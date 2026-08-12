import sys
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

sys.path.append("../utils")
from model_utils import load_data, split_data, MODEL_DIR

OUT_DIR = "../../outputs_shap"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = joblib.load(f"{MODEL_DIR}/xgb_model_tuned.pkl")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_global_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/1_global_feature_importance.png")


    probs = model.predict_proba(X_test)[:, 1]
    highest_risk_idx = probs.argmax()

    print(f"\nExplaining batch at index {highest_risk_idx}")
    print("Actual process values:")
    print(X_test.iloc[highest_risk_idx])
    print(f"Predicted defect probability: {probs[highest_risk_idx]:.3f}")
    print(f"Actual label: {y_test.iloc[highest_risk_idx]}")

    print("\nSHAP contributions (positive = pushes toward defective):")
    contributions = pd.Series(
        shap_values[highest_risk_idx], index=X_test.columns
    ).sort_values(key=abs, ascending=False)
    print(contributions.round(3))

    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[highest_risk_idx],
        X_test.iloc[highest_risk_idx], matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/2_single_prediction_explanation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {OUT_DIR}/2_single_prediction_explanation.png")


if __name__ == "__main__":
    main()