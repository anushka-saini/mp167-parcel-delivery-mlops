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


def norm_service(x):
    """strip spaces, lower case, expand abbreviations by first 3 letters"""
    x = x.strip().lower()
    full = {"sta": "standard", "eco": "economy", "exp": "express"}
    return full[x[:3]]


def check_spelling(df, col, clean_df):
    """raw labels, true levels after cleaning, whether they match the clean file"""
    raw = df[col].nunique()
    levels = df[col].map(norm_service).nunique()
    ok = set(df[col].map(norm_service)) == set(clean_df[col])
    return raw, levels, ok


if __name__ == "__main__":
    tr = load("data/raw/train.csv")
    ho = load("data/raw/holdout_features.csv")
    lt = load("data/raw/later.csv")
    print("duplicates (full, id, ignoring id):", check_dups(tr))
    print("leak:", check_leak(tr, ho))
    print("spelling (raw, true levels, matches later):", check_spelling(tr, "service", lt))