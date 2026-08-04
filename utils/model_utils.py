import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

DATA_PATH = "../../Data/syntheticData.csv"
MODEL_DIR = "../../models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = ["carbon_pct", "manganese_pct", "furnace_temp_C",
                 "rolling_speed_mps", "cooling_rate_Cps"]
TARGET_COL = "defect"


