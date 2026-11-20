import pandas as pd
from src.mp167.clean import clean_train
from src.mp167.checks import check_dups, check_leak


def small():
    return pd.DataFrame({
        "record_id": ["a", "b", "c"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "weight_kg": [5.0, 7000.0, 3.0],
        "service": [" STANDARD", "exp.", "economy"],
        "destination": ["Metro", "Tier-2", "Rural"],
        "days_from_tracking_page": [4.0, 5.0, 60.0],
        "delivery_days": [4.1, 5.1, 61.0],
    })


def test_clean_service():
    c = clean_train(small())
    assert list(c["service"]) == ["standard", "express", "economy"]


def test_clean_weight_unit():
    c = clean_train(small())
    assert c.loc[1, "weight_kg"] == 7.0       # Tier-2 grams -> kg
    assert c.loc[0, "weight_kg"] == 5.0       # Metro untouched


def test_clean_typo_and_leak():
    c = clean_train(small())
    assert c.loc[2, "delivery_days"] == 6.1   # 61.0 / 10
    assert "days_from_tracking_page" not in c.columns


def test_check_dups():
    d = pd.DataFrame({"record_id": ["a", "a"], "x": [1, 1]})
    assert check_dups(d) == (1, 1, 1)


def test_check_leak():
    tr = pd.DataFrame(columns=["a", "b", "leak", "delivery_days"])
    ho = pd.DataFrame(columns=["a", "b"])
    assert check_leak(tr, ho) == ["leak"]