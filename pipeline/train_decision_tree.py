# train_decision_tree.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ===== input file =====
CSV_PATH = r"C:\Users\yarib_e700ups\OneDrive\Documents\Python\decision_tree_input.csv"

# ===== load =====
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# ===== features (X) =====
FEATURE_COLS = ["H_norm","D_norm","Z_norm","PRES_normalized","STEC"]

X = df[FEATURE_COLS]
y = df["is_anom"]        # 0 or 1

# ===== split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25,
    random_state = 42,
    stratify = y
)

# ===== model =====
dt = DecisionTreeClassifier(
    max_depth = 4,        # not too complex
    random_state = 42
)

dt.fit(X_train, y_train)

# ===== eval =====
y_pred = dt.predict(X_test)
print("=== DECISION TREE REPORT ===")
print(classification_report(y_test, y_pred))

# ===== save =====
joblib.dump(dt, "decision_tree_model.pkl")
print("\nSaved decision_tree_model.pkl")
