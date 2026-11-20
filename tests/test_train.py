import pandas as pd
from src.mp167.train import fit

FEATS = ["distance_km", "weight_kg", "hub_backlog", "pin_code_tier",
         "declared_value_rs", "transfers", "service", "destination"]


def test_fit_predicts_on_request_columns():
    df = pd.DataFrame({
        "distance_km": [100, 500, 1200, 300],
        "weight_kg": [1.0, 5.5, 12.0, 3.2],
        "hub_backlog": [200, 1500, 3000, 800],
        "pin_code_tier": [1, 2, 3, 2],
        "declared_value_rs": [1000, 5000, 12000, 3000],
        "transfers": [1, 3, 5, 2],
        "service": ["standard", "express", "economy", "standard"],
        "destination": ["Metro", "Tier-2", "Rural", "Metro"],
        "delivery_days": [2.0, 4.5, 9.0, 3.0],
    })
    params = {
        "features": FEATS,
        "cat_features": ["service", "destination"],
        "target": "delivery_days",
        "seed": 42,
        "model": {"type": "Ridge", "args": {"alpha": 1.0}},
    }
    pipe = fit(df, params)
    pred = pipe.predict(df[FEATS])
    assert len(pred) == len(df)
