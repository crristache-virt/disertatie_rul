import pandas as pd

def compute_rul(df):
    df = df.copy()
    df["failure_date"] = df.groupby("serial_number")["date"].transform(lambda x: x.max() if df.loc[x.index, "failure"].any() else pd.NaT)
    df["RUL"] = (df["failure_date"] - df["date"]).dt.days
    df["RUL"] = df["RUL"].where(df["failure"] == 1, None)
    return df

def label_failure_within_n_days(df, n=30):
    df = df.copy()
    df["failure_within_n_days"] = (df["RUL"] <= n).astype(int)
    return df
