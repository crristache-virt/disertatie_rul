import pandas as pd

def handle_missing_values(df):
    return df.dropna()

def drop_constant_features(df):
    return df.loc[:, (df != df.iloc[0]).any()]

def normalize_columns(df, exclude=["date", "serial_number", "model", "failure"]):
    num_cols = [c for c in df.columns if c.startswith("smart_") and c not in exclude]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    return df
