# ===============================
# Hourly Quake Forecast Pipeline
# ===============================

import os
import time
import zipfile
import tempfile
import requests
import datetime as dt

import numpy as np
import pandas as pd
import joblib
import georinex as gr

from tensorflow.keras.models import load_model
from supabase import create_client
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv

# -------------------------------
# 0. ENV + PATHS
# -------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = os.getenv("SUPABASE_BUCKET", "quake-temp-files")

AE_MODEL_PATH = "models/autoencoder_model.keras"
DT_MODEL_PATH = "models/decision_tree_model.pkl"
LSTM_MODEL_PATH = "models/lstm_best.keras"
SCALER_PATH = "models/scaler.save"

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

NOW = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
TIMESTAMP_STR = NOW.strftime("%Y%m%d_%H%M")

# -------------------------------
# 1. SELENIUM SETUP
# -------------------------------

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)

# -------------------------------
# 2. FETCH EM DATA (USGS)
# -------------------------------

def fetch_em_channel(channel):
    url = (
        "https://geomag.usgs.gov/plots/"
        f"?stations=SJG&channels={channel}"
        "&dataView=Channel&timeRange=hour"
        "&dataType=variation"
    )

    driver = get_driver()
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "Details"))
        ).click()

        WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "IAGA-2002"))
        ).click()

        time.sleep(5)
        text = driver.page_source
    finally:
        driver.quit()

    lines = [l for l in text.split("\n") if not l.startswith("#")]
    rows = [l.split() for l in lines if len(l.split()) >= 4]

    df = pd.DataFrame(rows, columns=["DATE", "TIME", "DOY", f"SJG{channel}"])
    df[f"SJG{channel}"] = pd.to_numeric(df[f"SJG{channel}"], errors="coerce")

    return df.iloc[-1][f"SJG{channel}"]

def fetch_em():
    return {
        "H_norm": fetch_em_channel("H"),
        "D_norm": fetch_em_channel("D"),
        "Z_norm": fetch_em_channel("Z"),
    }

# -------------------------------
# 3. FETCH PRESSURE (CARICOOS)
# -------------------------------

def fetch_pressure():
    url = "http://gyre.umeoce.maine.edu/caricoos/PR2_surface.dat"
    df = pd.read_csv(url, delim_whitespace=True, comment="#")

    df["datetime"] = pd.to_datetime(
        df[["YY", "MM", "DD", "hh", "mm"]]
        .rename(columns={"YY": "year", "MM": "month", "DD": "day",
                          "hh": "hour", "mm": "minute"})
    )

    return df.iloc[-1]["PRES"]

# -------------------------------
# 4. FETCH TEC (NOAA UFCORS)
# -------------------------------

def fetch_tec():
    driver = get_driver()
    driver.get("https://geodesy.noaa.gov/UFCORS/")

    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "site")))

        driver.find_element(By.ID, "site").send_keys("N240")
        driver.find_element(By.ID, "startDate").send_keys(NOW.strftime("%m/%d/%Y"))
        driver.find_element(By.ID, "startTime").send_keys(NOW.strftime("%H:%M"))
        driver.find_element(By.ID, "duration").send_keys("1")

        driver.find_element(By.ID, "gps_legacy").click()
        driver.find_element(By.ID, "submit").click()

        time.sleep(10)

        link = driver.find_element(By.PARTIAL_LINK_TEXT, ".zip").get_attribute("href")
    finally:
        driver.quit()

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "tec.zip")
        with open(zip_path, "wb") as f:
            f.write(requests.get(link).content)

        with zipfile.ZipFile(zip_path) as z:
            rinex = [n for n in z.namelist() if n.endswith("o")][0]
            z.extract(rinex, tmp)

        ds = gr.load(os.path.join(tmp, rinex))
        tec = ds["slant_tec"].values
        return float(np.nanmean(tec))

# -------------------------------
# 5. COMBINE + SCALE
# -------------------------------

em = fetch_em()
pres = fetch_pressure()
tec = fetch_tec()

row = {
    "timestamp": NOW,
    **em,
    "PRES_normalized": pres,
    "STEC": tec,
}

df = pd.DataFrame([row])

scaler = joblib.load(SCALER_PATH)
X = scaler.transform(df.drop(columns=["timestamp"]))

# -------------------------------
# 6. AUTOENCODER
# -------------------------------

ae = load_model(AE_MODEL_PATH)
X_recon = ae.predict(X)
recon_error = np.mean((X - X_recon) ** 2, axis=1)[0]

ANOMALY = recon_error > 2.0  # sigma 2.0 threshold

# -------------------------------
# 7. DECISION TREE
# -------------------------------

dt_model = joblib.load(DT_MODEL_PATH)
dt_pred = int(dt_model.predict(X)[0])

# -------------------------------
# 8. LSTM RISK (5-DAY PROBABILITY)
# -------------------------------

lstm = load_model(LSTM_MODEL_PATH)
risk = float(lstm.predict(X.reshape(1, 1, -1))[0][0])

# -------------------------------
# 9. SAVE + UPLOAD
# -------------------------------

out_file = f"{OUTPUT_DIR}/snapshot_{TIMESTAMP_STR}.csv"
df.assign(
    recon_error=recon_error,
    is_anomaly=ANOMALY,
    dt_flag=dt_pred,
    quake_risk_5d=risk,
).to_csv(out_file, index=False)

with open(out_file, "rb") as f:
    supabase.storage.from_(BUCKET).upload(
        f"snapshot_{TIMESTAMP_STR}.csv", f
    )

supabase.table("anomaly_history").insert({
    "timestamp": NOW.isoformat(),
    "recon_error": recon_error,
    "dt_flag": dt_pred,
    "quake_risk_5d": risk,
}).execute()

print("✅ Hourly pipeline completed successfully.")
