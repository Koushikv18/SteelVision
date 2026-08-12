import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

import os

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(UTILS_DIR, "..", "Data", "syntheticData.csv"))
MODEL_DIR = os.path.abspath(os.path.join(UTILS_DIR, "..", "models"))
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = ["carbon_pct", "manganese_pct", "furnace_temp_C",
                 "rolling_speed_mps", "cooling_rate_Cps"]
TARGET_COL = "defect"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Train defect rate: {y_train.mean():.3f}  |  Test defect rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


def evaluate(model_name, y_true, y_pred, y_proba):
    print(f"\n--- {model_name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_proba):.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))