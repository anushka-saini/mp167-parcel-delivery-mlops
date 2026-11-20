"""Cleaning for train.csv (MP167). Fixes only the problems found in checks.py."""
from src.mp167.checks import load, norm_service


def clean_train(df, thr=20):
    d = df.copy()
    d["service"] = d["service"].map(norm_service)         # spelling: 18 labels -> 3
    m = d["destination"] == "Tier-2"
    d.loc[m, "weight_kg"] = d.loc[m, "weight_kg"] / 1000  # unit mixup: grams -> kg
    d.loc[d["delivery_days"] > thr, "delivery_days"] /= 10  # 14 typos: extra zero
    d = d.drop(columns=["days_from_tracking_page"])        # leak: not known at prediction time
    return d


if __name__ == "__main__":
    tr = load("data/raw/train.csv")
    c = clean_train(tr)
    print(c.shape)
    print(c.groupby("destination")["weight_kg"].median())
    print(c["delivery_days"].max())
    print(sorted(c["service"].unique()))