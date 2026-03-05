import pandas as pd
from sklearn.model_selection import train_test_split

def split_by_serial_number(df, test_size=0.2, random_state=42):
    serials = df["serial_number"].unique()
    train_serials, test_serials = train_test_split(serials, test_size=test_size, random_state=random_state)
    train_df = df[df["serial_number"].isin(train_serials)]
    test_df = df[df["serial_number"].isin(test_serials)]
    return train_df, test_df
