from pathlib import Path

import pandas as pd


def read_csv(path: str, npy_col: str = "npy") -> pd.DataFrame:
    path = Path(path)
    assert path.is_file(), f"{path} is not a file"

    df = pd.read_csv(path)
    df[npy_col] = df[npy_col].apply(lambda x: Path(x))
    return df
