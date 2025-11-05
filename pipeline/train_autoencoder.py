# ================================================
# TRAIN AUTOENCODER FOR EARTHQUAKE PRECURSORS
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# === 1️⃣ LOAD DATA ===
data_file = "autoencoder_training_data.csv"   # change to your combined normalized dataset
df = pd.read_csv(data_file)

if "timestamp" in df.columns:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop(columns=["timestamp"])

print(f"✅ Loaded dataset with shape {df.shape}")

# === 2️⃣ HANDLE MISSING VALUES (if any) ===
df = df.interpolate().dropna()

# === 3️⃣ SCALE AGAIN (just to ensure consistent 0–1 range) ===
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)
joblib.dump(scaler, "scaler.save")

print("✅ Data scaled and scaler saved as 'scaler.save'")

# === 4️⃣ TRAIN/VALIDATION SPLIT ===
X_train, X_val = train_test_split(data_scaled, test_size=0.2, shuffle=False)

# === 5️⃣ DEFINE AUTOENCODER ARCHITECTURE ===
input_dim = X_train.shape[1]

autoencoder = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu', name="bottleneck"),  # compressed representation
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# === 6️⃣ TRAIN MODEL ===
EPOCHS = 100
BATCH_SIZE = 16

history = autoencoder.fit(
    X_train, X_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, X_val),
    shuffle=True,
    verbose=1
)

# === 7️⃣ SAVE MODEL ===
autoencoder.save("autoencoder_model.keras")
print("✅ Autoencoder saved as 'autoencoder_model.keras'")

# === 8️⃣ PLOT TRAINING LOSS ===
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Autoencoder Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === 9️⃣ SAVE RECONSTRUCTION ERRORS (OPTIONAL) ===
reconstructed = autoencoder.predict(X_val)
mse = np.mean(np.power(X_val - reconstructed, 2), axis=1)
mse_df = pd.DataFrame({"Reconstruction_Error": mse})
mse_df.to_csv("validation_reconstruction_error.csv", index=False)

print("✅ Training complete. Reconstruction errors saved for analysis.")
