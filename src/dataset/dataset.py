import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        fold_col: str,
        data_col: str,
        fold: int,
        mode: str,
        transform=None,
        silent: bool = False,
    ):

        if mode == "train" or mode == "valid":
            assert target_col in df.columns, f"{target_col} is not in df.columns"
            assert fold_col in df.columns, f"{fold_col} is not in df.columns"
        assert data_col in df.columns, f"{data_col} is not in df.columns"
        assert mode in [
            "train",
            "valid",
            "test",
        ], f"{mode} is not in ['train', 'valid', 'test']"

        if mode == "train":
            # print("Original Length : ", len(df))
            df = df[df[fold_col] != fold]
            if not silent:
                print("Train Length : ", len(df))
        elif mode == "valid":
            # print("Original Length : ", len(df))
            df = df[df[fold_col] == fold]
            if not silent:
                print("Valid Length : ", len(df))
        elif mode == "test":
            pass
        else:
            raise ValueError(f"{mode} is not in ['train', 'valid', 'test']")

        self.npy_pathes = df[data_col].values

        if mode == "train" or mode == "valid":
            self.labels = df[target_col].values

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.npy_pathes)

    def __getitem__(self, idx):
        assert self.npy_pathes[idx].is_file(), f"{self.npy_pathes[idx]} is not a file"

        signal = np.load(self.npy_pathes[idx]).astype(np.float32)

        if self.transform:
            signal = self.transform(signal)

        if self.mode == "test":
            return signal, -1
        elif self.mode == "train" or self.mode == "valid":
            label = self.labels[idx]
            return signal, label
        else:
            raise ValueError(f"{self.mode} is not in ['train', 'valid', 'test']")
