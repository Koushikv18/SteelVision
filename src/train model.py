import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
from xgboost import XGBClassifier

sys.path.append("../utils")
from model_utils import load_data, split_data, evaluate, RANDOM_STATE

Test_size = 0.2
Random_state = 42

def train_logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
 
    model = LogisticRegression(random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
 
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
 
    evaluate("Logistic Regression (baseline)", y_test, y_pred, y_proba)
    return model, scaler

def train_xgboost(X_train, X_test, y_train, y_test):
    """PRIMARY model. Trees naturally handle U-shapes and interactions
    without needing them hand-engineered."""
 
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
 
    evaluate("XGBoost", y_test, y_pred, y_proba)
    return model
 
 
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    logreg_model, scaler = train_logistic_regression(X_train, X_test, y_train, y_test)
    xgb_model = train_xgboost(X_train, X_test, y_train, y_test)
 
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(logreg_model, f"{MODEL_DIR}/logreg_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgb_model.pkl")
    print(f"\nModels saved to {MODEL_DIR}/")
 
 
if __name__ == "__main__":
    main()




