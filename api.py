from fastapi import FastAPI
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.get("/latest")
def get_latest():
    res = (
        supabase
        .table("anomaly_history")
        .select("*")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if not res.data:
        return {"status": "no data yet"}

    row = res.data[0]

    return {
        "timestamp": row["timestamp"],
        "reconstruction_error": row["recon_error"],
        "anomaly_detected": bool(row["dt_flag"]),
        "quake_probability_5d": round(row["quake_risk_5d"] * 100, 2)
    }
