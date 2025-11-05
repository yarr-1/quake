import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ==== config ====
MODEL_PATH = r"C:\Users\yarib_e700ups\OneDrive\Documents\Python\autoencoder_model.keras"
CSV_PATH   = r"C:\Users\yarib_e700ups\OneDrive\Documents\Python\combo_anomalies.csv"

# === load ===
print("loading model...")
model = load_model(MODEL_PATH)

print("loading csv...")
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

# fill missing STEC (TEC gaps from 00–03)
df["STEC"] = df["STEC"].fillna(method="ffill")

# drop timestamp for model input
X = df.drop(columns=["timestamp"]).values

# === forward ===
print("running inference...")
recon = model.predict(X)

# per-feature reconstruction error
per_feature_err = np.abs(X - recon)

# make dataframe for inspection
err_df = pd.DataFrame(per_feature_err, columns=[
    "H_norm_err",
    "D_norm_err",
    "Z_norm_err",
    "PRES_normalized_err",
    "STEC_err"
])
 
# total reconstruction (for threshold)
total_err = np.mean(per_feature_err, axis=1)
recon_error = np.mean(np.square(X - recon), axis=1)
feature_error = (X - recon)**2   # shape = (samples, features)

# store main scalar reconstruction error in df
df["recon_error"] = total_err

feature_cols = ["H_norm", "D_norm", "Z_norm", "PRES_normalized", "STEC"]

# ---- per-sample per-feature reconstruction error ----
recon = pd.DataFrame(recon, columns=feature_cols)
orig  = pd.DataFrame(X, columns=feature_cols)

diff = (orig - recon).abs()

# total / mean error per sample
df["recon_error"] = diff.mean(axis=1)

# anomaly threshold
threshold = df["recon_error"].mean() + 2.0 * df["recon_error"].std()
feature_err_df = pd.DataFrame(feature_error, columns=feature_cols)

# compute 97.5 percentile per feature
feature_thresholds = feature_err_df.quantile(0.975)

# flag per feature anomalies
feature_anom = feature_err_df.gt(feature_thresholds)

df[ [f"anom_{c}" for c in feature_cols] ] = feature_anom.values

# row-level anomaly if ANY feature is anomalous
df["is_anom_any"] = feature_anom.any(axis=1)

df["is_anom"] = df["recon_error"] > threshold

# the feature that contributed the most to the anomaly
df["max_feature"] = diff.idxmax(axis=1)
df["max_feature_error"] = diff.max(axis=1)

# ==== print summary =====
print("anomalies detected = ", df["is_anom"].sum())
print(df[df["is_anom"]].head())

# === graph ===
features = ["H_norm","D_norm","Z_norm","PRES_normalized","STEC"]

for feat in features:
    plt.figure()
    plt.plot(df["timestamp"], df[feat], label=feat)
    # mark anomalies in red
    plt.scatter(df.loc[df["is_anom"],"timestamp"], df.loc[df["is_anom"],feat], marker="o", s=30)
    plt.xlabel("time")
    plt.ylabel(feat)
    plt.title(f"{feat} with anomalies")
    plt.legend()
    plt.show()

# before exporting

anom_df = df[df["is_anom"]][["timestamp","max_feature","max_feature_error"]]

if len(anom_df) > 0:
    anom_df.to_csv("feature_error_output.csv", index=False)
    print("saved anomalies → feature_error_output.csv")
else:
    print("no anomalies detected — try lowering threshold")


# === save output for later chain (decision tree use) ===
err_df.to_csv("feature_error_output.csv", index=False)
print("Saved per-feature error → feature_error_output.csv")

print(df[df["is_anom"]][["timestamp","max_feature","max_feature_error"]])


