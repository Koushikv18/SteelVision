
import sys
import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

sys.path.append("../utils")
from model_utils import load_data, split_data, evaluate, MODEL_DIR, RANDOM_STATE


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    param_grid = {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [100, 300],
        "min_child_weight": [1, 5],
    }

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        base_model, param_grid, scoring="f1", cv=5, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print("\nBest params:", search.best_params_)
    print("Best CV F1 score:", round(search.best_score_, 3))

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    evaluate("Tuned XGBoost (test set)", y_test, y_pred, y_proba)

    joblib.dump(best_model, f"{MODEL_DIR}/xgb_model_tuned.pkl")
    print(f"\nSaved tuned model to {MODEL_DIR}/xgb_model_tuned.pkl")


if __name__ == "__main__":
    main()