from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_path(relative_path: str) -> Path:
    p = PROJECT_ROOT / relative_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_csv(relative_path: str, parse_dates: list = None) -> pd.DataFrame:
    path = get_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)

def save_csv(df: pd.DataFrame, relative_path: str) -> None:
    path = get_path(relative_path)
    df.to_csv(path, index=False)