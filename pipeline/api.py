from fastapi import FastAPI
from datetime import datetime

app = FastAPI()

@app.get("/status")
def get_status():
    return {
        "last_update": datetime.utcnow().isoformat(),
        "anomalies": {
            "EM": False,
            "PRES": False,
            "TEC": False
        },
        "risk_5d_percent": 0
    }
    
