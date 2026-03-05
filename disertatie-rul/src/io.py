import pandas as pd
from pathlib import Path
from .config import DATA_DIR

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
    if str(file_path).endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)
    return df
