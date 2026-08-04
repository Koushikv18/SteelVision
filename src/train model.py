from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
from xgboost import XGBClassifier

df = pd.read_csv('../../Data/syntheticData.csv', encoding = 'latin-1')

x = df.drop(df.columns['defect'])
y = df['defect']

Test_size = 0.2
Random_state = 42

FEATURE_COLS = ["carbon_pct", "manganese_pct", "furnace_temp_C",
                 "rolling_speed_mps", "cooling_rate_Cps"]
TARGET_COL = "defect"
 
 
def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y






