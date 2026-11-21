# MP167 - Parcel Delivery Days (Jaipur)

SCSE3040 MLOps mini project.

- Group: **G218**
- Problem: **MP167**
- Faculty: Dr. Upasna Joshi (Batch 12)

| Enrolment no. | Name | GitHub | Role |
|---|---|---|---|
| S24CSEU1046 | Anushka Saini | `anushka-saini` | Data lead / Model lead |
| S24CSEU1075 | Poorvi Mahajan | `<username>` | TBD |
| S24CSEU1051 | Abhinav | `<username>` | TBD |
| S24CSEU1070 | Khushi Tarneja | `khushitarneja` | Service lead |

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
```

## Data
The raw CSVs are tracked with DVC (`data/raw/train.csv.dvc`, `data/raw/later.csv.dvc`) and are not stored in git.
Copy the original `train.csv` and `later.csv` into `data/raw/` (never open them in Excel and save).
`dvc status` should then say the data is up to date. `holdout_features.csv` is committed as a normal file.

## Run
1. Data checks: `python -m src.mp167.checks`
2. Clean data (sanity check): `python -m src.mp167.clean`
3. Train v1: `python -m src.mp167.train` (already run: v1 is registered as `mp167` version 1 with alias `champion`; do not re-run)
4. MLflow dashboard: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
5. Tests: `pytest`
6. Service: `uvicorn app.main:app --port 8000` (in progress)
7. Docker: `docker build -t mp167-api:1.0 .` (in progress)
8. Drift + v2: `python -m src.mp167.monitor` (in progress)
9. Holdout: `python -m src.mp167.predict` (in progress)

## Layout
- `src/mp167/`: checks, cleaning, training, monitoring, prediction
- `app/`: FastAPI service
- `tests/`: pytest tests
- `models/model.joblib`: the champion model the service loads
- `mlflow.db`: MLflow tracking and registry
- `reports/findings.yaml`: the form read by the marking tool
- `k8s/`, `Dockerfile`, `.github/workflows/ci.yml`: deployment and CI
