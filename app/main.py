import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict, field_validator

from src.mp167.checks import norm_service


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

FEATURES = config["features"]

MODEL_PATH = ROOT / config["paths"]["model"]
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Parcel Delivery Prediction API")


class JsonFmt(logging.Formatter):
    def format(self, rec):
        out = {
            "time": self.formatTime(rec),
            "level": rec.levelname,
            "msg": rec.getMessage(),
        }
        out.update(getattr(rec, "extra_data", {}))
        return json.dumps(out)


h = logging.StreamHandler(sys.stdout)
h.setFormatter(JsonFmt())
log = logging.getLogger("mp167")
log.setLevel(logging.INFO)
log.addHandler(h)
log.propagate = False


class ParcelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float
    weight_kg: float
    hub_backlog: float
    pin_code_tier: int
    declared_value_rs: float = Field(..., ge=50)
    transfers: int = Field(..., le=15)
    service: str
    destination: str

    @field_validator("service")
    @classmethod
    def check_service(cls, value):
        try:
            return norm_service(value)
        except KeyError:
            raise ValueError("Invalid service")

    @field_validator("destination")
    @classmethod
    def check_destination(cls, value):
        if value not in ["Metro", "Tier-2", "Rural"]:
            raise ValueError("Invalid destination")
        return value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: ParcelInput):
    t0 = time.time()
    df = pd.DataFrame([data.model_dump()], columns=FEATURES)
    try:
        result = float(model.predict(df)[0])
    except Exception:
        log.exception("prediction failed")
        raise
    log.info("prediction", extra={"extra_data": {
        "prediction": result,
        "service": data.service,
        "destination": data.destination,
        "ms": round((time.time() - t0) * 1000, 2),
    }})
    return {"prediction": result}