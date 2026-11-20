"""Training for MP167: predict delivery_days (regression, MAE)."""
import hashlib

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.mp167.checks import load
from src.mp167.clean import clean_train

# model "type" names used in params.yaml
REGS = {
    "Ridge": Ridge,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
}


def fit(df, params):
    """Build the pipeline for params["model"] and fit it on a cleaned df."""
    feats = params["features"]
    cats = params["cat_features"]
    nums = [c for c in feats if c not in cats]

    m = params["model"]
    reg = REGS[m["type"]](**m.get("args", {}))
    if "random_state" in reg.get_params():
        reg.set_params(random_state=params["seed"])

    # one-hot the text columns, pass numbers through unchanged
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", "passthrough", nums),
    ])
    pipe = Pipeline([("prep", prep), ("reg", reg)])
    pipe.fit(df[feats], df[params["target"]])
    return pipe


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)
    feats, y = p["features"], p["target"]

    # fingerprint of the original, untouched train.csv
    with open(p["paths"]["train"], "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    # time-based split: oldest 80% train, newest 20% validation
    df = clean_train(load(p["paths"]["train"]))
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    n = int(len(df) * (1 - p["val_frac"]))
    tr, va = df.iloc[:n], df.iloc[n:]

    # later.csv is already clean (weights in kg, 3 service labels), so
    # clean_train would wrongly divide its Tier-2 weights by 1000.
    # Only the leak column is dropped here.
    lt = load(p["paths"]["later"]).drop(columns=["days_from_tracking_page"])

    mlflow.set_tracking_uri(p["mlflow"]["uri"])
    mlflow.set_experiment(p["mlflow"]["experiment"])

    runs = []  # (run_id, mae_val, model_uri, pipe)
    for m in p["models"]:
        with mlflow.start_run() as r:
            pipe = fit(tr, {**p, "model": m})
            mae_val = mean_absolute_error(va[y], pipe.predict(va[feats]))
            mae_later = mean_absolute_error(lt[y], pipe.predict(lt[feats]))

            mlflow.log_param("data_sha256", sha)
            mlflow.log_param("model", m["type"])
            mlflow.log_params(m.get("args", {}))
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("mae_later", mae_later)  # info only, not used to pick
            info = mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle"
            )
            runs.append((r.info.run_id, mae_val, info.model_uri, pipe, m))

    # pick the best run by validation MAE only
    best = min(runs, key=lambda x: x[1])
    name = p["mlflow"]["model_name"]
    mv = mlflow.register_model(best[2], name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)
    joblib.dump(best[3], p["paths"]["model"])

    print("experiment:", p["mlflow"]["experiment"])
    print("best run id:", best[0])
    print("registered model:", name)
    print("champion version:", mv.version)
    for rid, mae, _, _, m in runs:
        print(f"  {rid}  {m['type']:<16} {m.get('args', {})}  mae_val={mae:.4f}")


if __name__ == "__main__":
    main()
