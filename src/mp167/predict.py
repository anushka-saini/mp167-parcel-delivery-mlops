"""Predict delivery days with the champion installed by the promotion step."""

from pathlib import Path

import pandas as pd


def predict_holdout(model, df, feats):
    out = df[["record_id"]].copy()
    out["prediction"] = model.predict(df[feats])
    if out.isna().any().any():
        raise ValueError("Every holdout row needs a record_id and a prediction.")
    return out


def main():
    import joblib
    import yaml

    cfg = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    df = pd.read_csv(paths["holdout"], dtype={"record_id": "string"})
    if len(df) != 250:
        raise ValueError(f"Expected 250 holdout rows, received {len(df)}.")

    # Retraining replaces this file only when v2 becomes champion.
    model = joblib.load(paths["model"])
    out = predict_holdout(model, df, cfg["features"])
    path = Path(paths["preds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} predictions to {path}")


if __name__ == "__main__":
    main()
