# make_labels.py
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = r"C:\Users\yarib_e700ups\OneDrive\Documents\Python\autoencoder_model.keras"   # your AE path
CSV_PATH   = r"combo_anomalies.csv"              # merged CSV
SCALER_PATH = r"scaler.save"                     # optional scaler (if you have it)
OUT_CSV = "decision_tree_input.csv"

FEATURE_COLS = ["H_norm","D_norm","Z_norm","PRES_normalized","STEC"]
SIGMA = 2.2   # <-- use 2.0 sigma as you requested

# ---------- load dataset ----------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
df = df.copy().reset_index(drop=True)

# Fill NaNs sensibly (STEC had intentionally blank rows)
df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

# ---------- prepare model input (use saved scaler if available) ----------
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("Loaded scaler:", SCALER_PATH)
    X = scaler.transform(df[FEATURE_COLS])
else:
    scaler = None
    # assume data already scaled / normalized; ensure numeric
    X = df[FEATURE_COLS].astype(float).values

# ---------- load autoencoder ----------
print("Loading autoencoder model:", MODEL_PATH)
ae = load_model(MODEL_PATH)

# ---------- inference ----------
print("Running AE inference...")
X_pred = ae.predict(X, verbose=0)

# ---------- per-feature squared error and total recon_error ----------
feat_err = (X - X_pred) ** 2                       # shape = (n_samples, n_features)
feat_err_df = pd.DataFrame(feat_err, columns=[c + "_err" for c in FEATURE_COLS])
feat_err_df["timestamp"] = df["timestamp"]

# scalar reconstruction error per sample (mean of feature errors)
recon_error = feat_err.mean(axis=1)                # numpy array
df["recon_error"] = recon_error

# ---------- scalar threshold using sigma ----------
threshold = recon_error.mean() + SIGMA * recon_error.std()
df["is_anom"] = df["recon_error"] > threshold

print(f"Using sigma = {SIGMA:.2f} -> threshold = {threshold:.6f}")
print("Total anomalies labeled (is_anom==1):", int(df["is_anom"].sum()))

# ---------- per-row top contributing feature (for flagged rows) ----------
# Build DataFrame for original features and per-feature errors
out_df = df[["timestamp"] + FEATURE_COLS].copy()
for col in feat_err_df.columns:
    out_df[col] = feat_err_df[col]

# Add max contribution columns
out_df["max_feature"] = feat_err_df[[c + "_err" for c in FEATURE_COLS]].idxmax(axis=1)
out_df["max_feature_error"] = feat_err_df[[c + "_err" for c in FEATURE_COLS]].max(axis=1)
out_df["recon_error"] = df["recon_error"]
out_df["is_anom"] = df["is_anom"].astype(int)

# ---------- save final csv for decision tree -->
out_df.to_csv(OUT_CSV, index=False)
print("Saved labeled file ->", OUT_CSV)

# ---------- print the anomaly timestamps and their top feature contributions ----------
anom_rows = out_df[out_df["is_anom"] == 1].copy()
if len(anom_rows) == 0:
    print("No anomalies detected with this threshold.")
else:
    print("\nAnomalies (timestamp, max_feature, max_feature_error):")
    for _, r in anom_rows.iterrows():
        print(f"{r['timestamp']} | {r['max_feature']} | {r['max_feature_error']:.6f}")

# ---------- optional: quick summary counts per feature ----------
counts = {}
for f in FEATURE_COLS:
    counts[f] = int((out_df["max_feature"] == f + "_err").sum())
print("\nCount of rows where each feature was top contributor:")
for k,v in counts.items():
    print(f"  {k}: {v}")

print("\nDone.")
