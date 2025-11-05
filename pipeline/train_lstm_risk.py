# train_lstm_risk.py
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =========== CONFIG ===========
SENSOR_CSV = "combo_anomalies.csv"   # your merged sensor file
EQ_CSV     = "eq_catalog.csv"        # your earthquake list (timestamp, mag)
OUT_MODEL  = "lstm_risk.keras"
SCALER_OUT = "lstm_scaler.save"

MAG_THRESHOLD = 4.0      # significant earthquake threshold
PRED_DAYS = 5            # predict if EQ occurs within next 5 days
SEQ_HOURS = 72           # input sequence length (past 72 hours)
TEST_SPLIT = 0.25
BATCH_SIZE = 32
EPOCHS = 60
RANDOM_SEED = 42
# =============================

np.random.seed(RANDOM_SEED)

# ---------- 1) load data ----------
print("Loading sensor data:", SENSOR_CSV)
df = pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"]).reset_index(drop=True)

print("Loading EQ catalog:", EQ_CSV)
eq = pd.read_csv(EQ_CSV, parse_dates=["timestamp"]).reset_index(drop=True)

# sanitize column names (user reported STEC and PRES names)
FEATURE_COLS = ["H_norm","D_norm","Z_norm","PRES_normalized","STEC"]
for c in FEATURE_COLS:
    if c not in df.columns:
        raise Exception(f"Missing feature column in sensor CSV: {c}")

# ---------- 2) build 5-day ahead label ----------
print("Building 5-day ahead labels (mag >= %.2f) ..." % MAG_THRESHOLD)

# sort eq by time
eq = eq.sort_values("timestamp").reset_index(drop=True)

# for speed: build numpy arrays of eq timestamps and mags
eq_times = eq["timestamp"].values
eq_mags = eq["mag"].values

labels = []
for t in df["timestamp"]:
    # window end
    end = t + pd.Timedelta(days=PRED_DAYS)
    # check if any eq in (t, end] with mag >= threshold
    mask = (eq["timestamp"] > t) & (eq["timestamp"] <= end) & (eq["mag"] >= MAG_THRESHOLD)
    labels.append(1 if mask.any() else 0)

df["risk_label"] = labels
print("Total positive risk labels:", int(df["risk_label"].sum()), " / ", len(df))

# ---------- 3) prepare sequences for LSTM ----------
print("Preparing sequences (SEQ_HOURS =", SEQ_HOURS, ") ...")

# make sure timestamps are evenly spaced hourly; if not, you may want to reindex/resample
# we'll assume df has hourly index aligned (as you set earlier)

values = df[FEATURE_COLS].astype(float).values
y = df["risk_label"].astype(int).values

# scale features with StandardScaler (fit on entire dataset then save)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(values)
joblib.dump(scaler, SCALER_OUT)
print("Saved scaler to", SCALER_OUT)

# build sliding windows
seqs = []
seq_labels = []
for i in range(len(X_scaled) - SEQ_HOURS):
    seq = X_scaled[i : i + SEQ_HOURS]        # shape (SEQ_HOURS, n_features)
    target = y[i + SEQ_HOURS]               # label at time after the window
    seqs.append(seq)
    seq_labels.append(target)

X = np.stack(seqs)          # (samples, seq_len, features)
Y = np.array(seq_labels)

print("X shape:", X.shape, "Y shape:", Y.shape)

# ---------- 4) train/test split (time-based: first 75% train, last 25% test) ----------
n = len(X)
split = int((1 - TEST_SPLIT) * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

print("Train samples:", len(X_train), "Test samples:", len(X_test))

# ---------- 5) build LSTM model ----------
n_features = X.shape[2]
model = Sequential([
    Masking(mask_value=0., input_shape=(SEQ_HOURS, n_features)),
    LSTM(64, return_sequences=False),
    Dropout(0.25),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
model.summary()

# ---------- 6) training with class weights ----------
# compute class weights to account for imbalance
from sklearn.utils import class_weight
classes = np.unique(y_train)
cw = class_weight.compute_class_weight("balanced", classes=classes, y=y_train)
class_weights = {int(classes[i]): cw[i] for i in range(len(classes))}
print("class_weights:", class_weights)

chk = ModelCheckpoint("lstm_best.keras", monitor="val_loss", save_best_only=True, verbose=1)
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[chk, es],
    verbose=2
)

# save final model
model.save(OUT_MODEL)
print("Saved LSTM model to", OUT_MODEL)

# ---------- 7) evaluation ----------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=4))

try:
    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC:", auc)
except Exception as e:
    print("ROC AUC could not be computed:", e)

# plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={auc:.3f}" if 'auc' in locals() else "ROC")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("lstm_roc.png")
print("Saved ROC plot -> lstm_roc.png")

# save training history plot
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Training Loss")
plt.savefig("lstm_history.png")
print("Saved training history -> lstm_history.png")

# also save a small CSV for later inspection (timestamps aligned to window end)
test_idx_start = split
test_timestamps = df["timestamp"].iloc[test_idx_start + SEQ_HOURS : test_idx_start + SEQ_HOURS + len(X_test)].reset_index(drop=True)
pd.DataFrame({"timestamp": test_timestamps, "y_true": y_test, "y_prob": y_prob}).to_csv("lstm_test_predictions.csv", index=False)
print("Saved test predictions -> lstm_test_predictions.csv")
