from src.cleaning import drop_constant_features
from src.evaluation import regression_metrics
from src.io import load_quarterly_rul_datasets
from src.labeling import compute_rul
from src.models import get_baseline_regressors
from src.preprocessing import convert_date_column, sort_by_serial_and_date
from src.split import train_val_test_split_by_serial_number


IDENTIFIER_COLUMNS = {
    "date",
    "serial_number",
    "model",
    "failure",
    "RUL",
    "failure_date",
    "source_dataset",
}


def prepare_rul_dataframe():
    df = load_quarterly_rul_datasets()
    if df is None:
        raise FileNotFoundError("Quarterly RUL datasets were not found in DATA_DIR.")

    if "date" in df.columns:
        df = convert_date_column(df)
        df = sort_by_serial_and_date(df)

    if "RUL" not in df.columns:
        required_for_rul = {"date", "serial_number", "failure"}
        missing_columns = required_for_rul.difference(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                "Dataset must contain an 'RUL' column or enough fields to compute it. "
                f"Missing: {missing}"
            )
        df = compute_rul(df)

    df = df[df["RUL"].notna()].copy()
    df["RUL"] = df["RUL"].astype(float)
    df = drop_constant_features(df)
    return df


def get_feature_columns(df):
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    return [column for column in numeric_columns if column not in IDENTIFIER_COLUMNS]


def main():
    df = prepare_rul_dataframe()
    train_df, val_df, test_df = train_val_test_split_by_serial_number(df)

    feature_columns = get_feature_columns(train_df)
    if not feature_columns:
        raise ValueError("No numeric feature columns were found for RUL training.")

    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df["RUL"]
    X_val = val_df[feature_columns].fillna(0)
    y_val = val_df["RUL"]
    X_test = test_df[feature_columns].fillna(0)
    y_test = test_df["RUL"]

    print("Loaded rows:", len(df))
    print("Feature count:", len(feature_columns))
    print(
        "Split sizes:",
        {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_serials": train_df["serial_number"].nunique(),
            "val_serials": val_df["serial_number"].nunique(),
            "test_serials": test_df["serial_number"].nunique(),
        },
    )

    models = get_baseline_regressors()
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)

        print(f"\nModel: {name}")
        print("Validation:", regression_metrics(y_val, val_predictions))
        print("Test:", regression_metrics(y_test, test_predictions))


if __name__ == "__main__":
    main()
