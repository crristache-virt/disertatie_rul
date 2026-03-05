import pandas as pd

def sort_by_serial_and_date(df):
    return df.sort_values(["serial_number", "date"])

def convert_date_column(df, date_col="date"):
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def prepare_time_series(df):
    disks = df["serial_number"].unique()
    return {sn: df[df["serial_number"] == sn].sort_values("date") for sn in disks}
