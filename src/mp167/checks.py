"""Data checks for train.csv (MP167)."""
import pandas as pd


def load(path):
    return pd.read_csv(path)


def check_dups(df):
    """extra copies: exact rows, repeated ids, rows equal apart from the id"""
    full = int(df.duplicated().sum())
    ids = int(df["record_id"].duplicated().sum())
    body = int(df.drop(columns="record_id").duplicated().sum())
    return full, ids, body

def check_leak(tr, ho, target="delivery_days"):
    """columns in train that are missing from holdout, minus the target"""
    return sorted(set(tr.columns) - set(ho.columns) - {target})

if __name__ == "__main__":
    tr = load("data/raw/train.csv")
    ho = load("data/raw/holdout_features.csv")
    print("duplicates (full, id, ignoring id):", check_dups(tr))
    print("leak:", check_leak(tr, ho))
