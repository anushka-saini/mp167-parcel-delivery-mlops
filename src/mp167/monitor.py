
"""Drift and v1 performance monitoring for MP167."""

import numpy as np
import pandas as pd
import yaml
import joblib

from sklearn.metrics import mean_absolute_error
from src.mp167.clean import clean_train


def psi(a, b, bins=10):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both arrays must contain valid values")

    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))

    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)

    eps = 1e-6
    pa = pa + eps
    pb = pb + eps

    return float(np.sum((pb - pa) * np.log(pb / pa)))

def mae_by_group(model, df, feats, target="delivery_days"):
    scores = {}

    for group in ["Metro", "Tier-2", "Rural"]:
        subset = df[df["destination"] == group]

        if len(subset) == 0:
            raise ValueError(f"No rows found for group: {group}")

        predictions = model.predict(subset[feats])

        scores[group] = float(
            mean_absolute_error(
                subset[target],
                predictions
            )
        )

    worst_group = max(scores, key=scores.get)

    return scores, worst_group
def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    # Load and clean training data
    train = clean_train(pd.read_csv(p["paths"]["train"]))

    # Later data is already clean. Do NOT clean it again.
    later = pd.read_csv(p["paths"]["later"])
    later = later.drop(columns=["days_from_tracking_page"])

    # Load v1 model
    model = joblib.load(p["paths"]["model"])

    features = p["features"]
    numeric_features = [
        c for c in features if c not in p["cat_features"]
    ]

    # Calculate PSI for all six numeric columns
    psi_scores = {}

    for col in numeric_features:
        psi_scores[col] = psi(train[col], later[col], bins=10)

    most_drifted = max(psi_scores, key=psi_scores.get)

    mae_scores, worst_group = mae_by_group(model, later, features, target=p["target"])

    print("\nPSI SCORES:")
    for col, score in psi_scores.items():
        print(f"{col}: {score:.6f}")

    print("\nMOST DRIFTED FEATURE:", most_drifted)

    print("\nV1 MAE BY GROUP:")
    for group, score in mae_scores.items():
        print(f"{group}: {score:.6f}")

    print("\nWORST SERVED GROUP:", worst_group)


if __name__ == "__main__":
    main()
