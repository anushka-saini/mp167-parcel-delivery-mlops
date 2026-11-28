"""Compare v1 and v2 on the newest parcels and apply the client's MAE rule."""

import argparse
import hashlib
import re
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import url2pathname

import pandas as pd


def split_later(df, frac=0.7):
    """Split in date order, retaining input order for parcels on the same date."""
    if df["date"].isna().any():
        raise ValueError("Every parcel in later.csv needs a date.")
    ordered = df.sort_values("date", key=pd.to_datetime, kind="stable").reset_index(drop=True)
    n = int(len(ordered) * frac)
    return ordered.iloc[:n], ordered.iloc[n:]


def promote(mae1, mae2, rule=0.9):
    return mae2 <= rule * mae1


def fill_findings(text, mae1, mae2):
    promoted = promote(mae1, mae2)
    values = {
        "mae_v1_on_check": mae1,
        "mae_v2_on_check": mae2,
        "decision": "promote" if promoted else "keep",
        "champion_version": 2 if promoted else 1,
    }
    # Replace the existing scalar values so the rest of the form, including
    # teammates' comments and quoted labels, is preserved verbatim.
    for key, value in values.items():
        pattern = rf"^(  {key}:[ \t]+)[^\s#]+"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Expected one {key} field in findings.yaml.")
    return text


def compare(v1, train, later, feats, target):
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error

    early, check = split_later(later)
    if early.empty or check.empty:
        raise ValueError("later.csv needs rows in both the training and check sets.")
    cols = [*feats, target]
    combined = pd.concat([train[cols], early[cols]], ignore_index=True)
    v2 = clone(v1)
    v2.fit(combined[feats], combined[target])

    x, y = check[feats], check[target]
    mae1 = float(mean_absolute_error(y, v1.predict(x)))
    mae2 = float(mean_absolute_error(y, v2.predict(x)))
    return v2, mae1, mae2


def publish_champion(client, name, version, model, path):
    import joblib

    path = Path(path)
    # Stage the file on the same drive so replace() installs it in one operation.
    with TemporaryDirectory(dir=path.parent) as folder:
        staged = Path(folder) / path.name
        joblib.dump(model, staged)
        client.set_registered_model_alias(name, "champion", version)
        try:
            staged.replace(path)
        except OSError:
            client.set_registered_model_alias(name, "champion", "1")
            raise


def use_checkout_artifacts(uri, experiment, model_path):
    """Give new runs a local artifact folder if the original drive is missing."""
    old = experiment.artifact_location
    location = urlsplit(old)
    if not uri.startswith("sqlite:///") or location.scheme != "file" or location.netloc:
        return
    if Path(url2pathname(location.path)).is_dir():
        return

    # The team's SQLite experiment was created on another machine. MLflow
    # stores a separate artifact URI for each existing run and model.
    folder = Path(model_path).parent / "mlflow" / str(experiment.experiment_id)
    with sqlite3.connect(uri.removeprefix("sqlite:///")) as db:
        db.execute(
            "UPDATE experiments SET artifact_location = ? "
            "WHERE experiment_id = ? AND artifact_location = ?",
            (folder.as_posix(), experiment.experiment_id, old),
        )


def register_v2(cfg, v2, mae1, mae2):
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    paths = cfg["paths"]
    tracking = cfg["mlflow"]
    name = tracking["model_name"]
    mlflow.set_tracking_uri(tracking["uri"])
    client = MlflowClient()

    # A second invocation must stop before it can register an accidental v3.
    versions = client.search_model_versions(f"name = '{name}'")
    champion = client.get_model_version_by_alias(name, "champion")
    if {str(v.version) for v in versions} != {"1"} or str(champion.version) != "1":
        raise RuntimeError(f"Expected only {name} version 1 with alias champion.")

    experiment = mlflow.set_experiment(tracking["experiment"])
    use_checkout_artifacts(tracking["uri"], experiment, paths["model"])
    with mlflow.start_run(run_name="retrain-v2"):
        mlflow.log_params({
            "data_sha256": hashlib.sha256(Path(paths["train"]).read_bytes()).hexdigest(),
            "later_sha256": hashlib.sha256(Path(paths["later"]).read_bytes()).hexdigest(),
            "model": type(v2.named_steps["reg"]).__name__.removesuffix("Regressor"),
        })
        mlflow.log_metrics({"mae_v1_check": mae1, "mae_v2_check": mae2})
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        version = mlflow.register_model(info.model_uri, name).version
        if str(version) != "2":
            raise RuntimeError(f"Expected {name} version 2, received {version}.")
        if promote(mae1, mae2):
            publish_champion(client, name, version, v2, paths["model"])
    champion = client.get_model_version_by_alias(name, "champion")
    print(f"Champion: {name} version {champion.version}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="Compare models in memory only.")
    args = parser.parse_args()

    import joblib
    import yaml
    from .checks import load
    from .clean import clean_train

    cfg = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    v1 = joblib.load(paths["model"])
    train = clean_train(load(paths["train"]))
    # later.csv already uses the corrected weight units.
    later = load(paths["later"]).drop(columns=["days_from_tracking_page"], errors="ignore")
    v2, mae1, mae2 = compare(v1, train, later, cfg["features"], cfg["target"])
    print(f"mae_v1_on_check: {mae1}")
    print(f"mae_v2_on_check: {mae2}")
    print(f"decision: {'promote' if promote(mae1, mae2) else 'keep'}")
    if args.dry:
        return

    report = Path(paths["findings"])
    contents = fill_findings(report.read_text(encoding="utf-8"), mae1, mae2)
    with TemporaryDirectory(dir=report.parent) as folder:
        staged = Path(folder) / report.name
        staged.write_text(contents, encoding="utf-8")
        register_v2(cfg, v2, mae1, mae2)
        staged.replace(report)


if __name__ == "__main__":
    main()
