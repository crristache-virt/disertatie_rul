import pandas as pd
from pathlib import Path
from .config import DATA_DIR


def _read_dataset(file_path):
    if str(file_path).endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_parquet(file_path)


def load_backblaze_smart(file_path=None):
    """
    Load Backblaze SMART dataset (CSV or Parquet).
    Expected columns: date, serial_number, model, failure, smart_*
    """
    if file_path is None:
        file_path = next(DATA_DIR.glob("*.csv"), None) or next(DATA_DIR.glob("*.parquet"), None)
    if file_path is None or not Path(file_path).exists():
        print(f"Dataset not found in {DATA_DIR}. Please place Backblaze SMART files here.")
        return None
    return _read_dataset(file_path)


def load_quarterly_rul_datasets(
    dataset_names=None,
    data_dir=None,
):
    """
    Load and concatenate the expected quarterly RUL datasets.

    The loader looks for CSV or Parquet files named like:
    q1_rul_dataset.csv / .parquet ... q4_rul_dataset.csv / .parquet
    """
    if dataset_names is None:
        dataset_names = [
            "q1_rul_dataset",
            "q2_rul_dataset",
            "q3_rul_dataset",
            "q4_rul_dataset",
        ]

    root_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    datasets = []
    missing = []

    for dataset_name in dataset_names:
        file_path = None
        for extension in (".csv", ".parquet"):
            candidate = root_dir / f"{dataset_name}{extension}"
            if candidate.exists():
                file_path = candidate
                break

        if file_path is None:
            missing.append(dataset_name)
            continue

        quarterly_df = _read_dataset(file_path)
        quarterly_df["source_dataset"] = dataset_name
        datasets.append(quarterly_df)

    if missing:
        print(
            "Missing quarterly datasets in "
            f"{root_dir}: {', '.join(missing)}. "
            "Expected matching .csv or .parquet files."
        )

    if not datasets:
        return None

    return pd.concat(datasets, ignore_index=True)
