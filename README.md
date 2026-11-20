# MP167 - Parcel Delivery Days (Jaipur)

SCSE3040 MLOps mini project.

- Group: **G218**
- Problem: **MP167**
- Faculty: Dr. Upasna Joshi (Batch 12)

| Enrolment no. | Name | GitHub | Role |
|---|---|---|---|
| S24CSEU1046 | Anushka Saini | `<username>` | TBD |
| S24CSEU1075 | Poorvi Mahajan | `<username>` | TBD |
| S24CSEU1051 | Abhinav | `<username>` | TBD |
| S24CSEU1070 | Khushi Tarneja | `<username>` | TBD |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Put the original `train.csv` and `later.csv` in `data/raw/` (never open them in Excel).

## Run (fill in as we build)
1. Data checks: `python -m src.mp167.checks`
2. Train v1: `python -m src.mp167.train`
3. Service: `uvicorn app.main:app --port 8000`
4. Docker: `docker build -t mp167-api:1.0 .`
5. Drift + v2: `python -m src.mp167.monitor`
6. Holdout: `python -m src.mp167.predict`
7. Tests: `pytest`
