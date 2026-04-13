import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_serial_number(df, test_size=0.2, random_state=42):
    serials = df["serial_number"].unique()
    train_serials, test_serials = train_test_split(serials, test_size=test_size, random_state=random_state)
    train_df = df[df["serial_number"].isin(train_serials)]
    test_df = df[df["serial_number"].isin(test_serials)]
    return train_df, test_df


def train_val_test_split_by_serial_number(
    df,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    serials = df["serial_number"].dropna().unique()
    train_serials, temp_serials = train_test_split(
        serials,
        test_size=(1.0 - train_size),
        random_state=random_state,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_serials, test_serials = train_test_split(
        temp_serials,
        test_size=relative_test_size,
        random_state=random_state,
    )

    train_df = df[df["serial_number"].isin(train_serials)].copy()
    val_df = df[df["serial_number"].isin(val_serials)].copy()
    test_df = df[df["serial_number"].isin(test_serials)].copy()
    return train_df, val_df, test_df
