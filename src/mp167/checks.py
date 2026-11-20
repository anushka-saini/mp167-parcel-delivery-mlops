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

def check_sentinel(df, cands=(-1, -9, -99, -999, 0, 99, 999, 9999, 99999)):
    """numeric columns that contain a suspicious placeholder number, with counts"""
    out = {}
    for c in df.select_dtypes("number").columns:
        if c == "delivery_days":
            continue
        for v in cands:
            n = int((df[c] == v).sum())
            if n:
                out[(c, v)] = n
    return out

def check_typos(df, target="delivery_days"):
    """find the biggest gap in the sorted target; count the values above it"""
    s = df[target].sort_values().reset_index(drop=True)
    g = s.diff()
    i = int(g.idxmax())
    return len(s) - i, float(s[i - 1]), float(s[i])

def check_stuck(df, cols):
    """per column: the value of the longest run on consecutive dated rows, its length, first and last date"""
    s = df.sort_values("date").reset_index(drop=True)
    out = {}
    for c in cols:
        run = (s[c] != s[c].shift()).cumsum()
        g = s.groupby(run)
        n = g[c].transform("size")
        i = n.idxmax()
        out[c] = (s.loc[i, c], int(n[i]),
                  s.loc[run == run[i], "date"].min(),
                  s.loc[run == run[i], "date"].max())
    return out

def check_units(df, col, grp):
    """per group: median of col, and each group's median divided by the smallest"""
    m = df.groupby(grp)[col].median()
    return m.round(2).to_dict(), (m / m.min()).round(1).to_dict()

if __name__ == "__main__":
    tr = load("data/raw/train.csv")
    ho = load("data/raw/holdout_features.csv")
    lt = load("data/raw/later.csv")
    print("duplicates (full, id, ignoring id):", check_dups(tr))
    print("leak:", check_leak(tr, ho))
    print("spelling (raw, true levels, matches later):", check_spelling(tr, "service", lt))
    print("sentinel candidates:", check_sentinel(tr))
    print("typos (count above gap, below gap, above gap):", check_typos(tr))
    cols = ["distance_km", "weight_kg", "hub_backlog", "pin_code_tier", "declared_value_rs", "transfers"]
    print("stuck:", check_stuck(tr, cols))
    print("units weight_kg:", check_units(tr, "weight_kg", "destination"))