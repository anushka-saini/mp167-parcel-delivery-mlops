import numpy as np
import pandas as pd
from src.mp167.monitor import psi, mae_by_group

def test_psi_same_distribution():
    a = [1, 2, 3, 4]
    b = [1, 2, 3, 4]
    assert psi(a, b) == 0.0
def test_psi_detects_distribution_shift():
    a = [1, 2, 3, 4]
    b = [2, 3, 4, 5]
    assert psi(a, b) > 0

def test_mae_by_group():
    class DummyModel:
        def predict(self, X):
            return X["x"].to_numpy()

    df = pd.DataFrame({
        "destination": ["Metro", "Tier-2", "Rural"],
        "x": [1, 2, 3],
        "delivery_days": [2, 4, 8]
    })

    scores, worst_group = mae_by_group(
        DummyModel(), df, ["x"]
    )

    assert scores == {"Metro": 1.0, "Tier-2": 2.0, "Rural": 5.0}
    assert worst_group == "Rural"