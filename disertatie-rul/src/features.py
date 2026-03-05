import pandas as pd

def generate_time_based_features(df):
    # Example: rolling mean for SMART features
    smart_cols = [c for c in df.columns if c.startswith("smart_")]
    for col in smart_cols:
        df[f"{col}_rolling_mean"] = df.groupby("serial_number")[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    return df
