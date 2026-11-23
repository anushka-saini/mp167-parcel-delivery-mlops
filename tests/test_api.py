import io
import json
import logging

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def get_data():
    return {
        "distance_km": 215,
        "weight_kg": 5.53,
        "hub_backlog": 2965,
        "pin_code_tier": 3,
        "declared_value_rs": 6888,
        "transfers": 6,
        "service": "standard",
        "destination": "Tier-2",
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction():
    response = client.post("/predict", json=get_data())
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_transfers_above_limit():
    data = get_data()
    data["transfers"] = 16
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_transfers_at_limit():
    data = get_data()
    data["transfers"] = 15
    response = client.post("/predict", json=data)
    assert response.status_code == 200


def test_declared_value_below_limit():
    data = get_data()
    data["declared_value_rs"] = 49
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_declared_value_at_limit():
    data = get_data()
    data["declared_value_rs"] = 50
    response = client.post("/predict", json=data)
    assert response.status_code == 200


def test_missing_field():
    data = get_data()
    del data["distance_km"]
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_unknown_service():
    data = get_data()
    data["service"] = "premium"
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_unknown_destination():
    data = get_data()
    data["destination"] = "Unknown"
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_messy_service():
    data = get_data()
    data["service"] = " STANDARD"
    response = client.post("/predict", json=data)
    assert response.status_code == 200

def test_json_fmt():
    import json
    import logging
    from app.main import JsonFmt

    rec = logging.LogRecord("x", logging.INFO, "", 0, "hi", None, None)
    rec.extra_data = {"destination": "Metro"}
    out = json.loads(JsonFmt().format(rec))
    assert out["msg"] == "hi"
    assert out["destination"] == "Metro"


def test_predict_logs():
    from app.main import log, JsonFmt

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFmt())
    log.addHandler(h)
    try:
        client.post("/predict", json=get_data())
    finally:
        log.removeHandler(h)
    out = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert out["msg"] == "prediction"
    assert out["destination"] == "Tier-2"
