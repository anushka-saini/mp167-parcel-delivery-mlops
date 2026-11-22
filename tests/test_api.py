
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
