import numpy as np
import torch
from src.constants import (
    CLASSIFICATION1_TEST_CSV,
    CLASSIFICATION1_TRAIN_CSV,
    CLASSIFICATION2_TEST_CSV,
    CLASSIFICATION2_TRAIN_CSV,
)
from src.dataset.csv import read_csv
from src.dataset.eeg import butter_bandpass_filter, normalize


class CFG:
    target_col = "oe_label"
    fold_col = "oe_train"
    data_col = "npy"
    target_dict = {"backside_kickturn": 0, "frontside_kickturn": 1, "pumping": 2}
    fold_dict = {"train1": 0, "train2": 1, "train3": 2}

    def __init__(self, exp_name: str, target_subjectid: str, target="classification1"):
        self.exp_name = exp_name
        self.target_subjectid = target_subjectid
        self.target_dataset = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_transform = self.train_preprocess
        self.test_transform = self.test_preprocess

        self.lr = self.training_parameters()["lr"]
        self.epochs = self.training_parameters()["epochs"]

        self.train_df, self.test_df, self.oof_df = self.read_df(target)

    @staticmethod
    def train_preprocess(signal: np.ndarray):
        signal = normalize(signal)
        signal = butter_bandpass_filter(signal, 2, 125, 500)
        return signal

    @staticmethod
    def test_preprocess(signal: np.ndarray):
        signal = normalize(signal)
        signal = butter_bandpass_filter(signal, 2, 125, 500)
        return signal

    @staticmethod
    def model_parameters():
        raise NotImplementedError()

    @staticmethod
    def training_parameters():
        return {"lr": 1e-3, "epochs": 20}

    def read_df(self, target: str):
        if target == "classification1":
            train_path = CLASSIFICATION1_TRAIN_CSV
            test_path = CLASSIFICATION1_TEST_CSV
        elif target == "classification2":
            train_path = CLASSIFICATION2_TRAIN_CSV
            test_path = CLASSIFICATION2_TEST_CSV
        else:
            raise ValueError(f"Invalid target: {target}")

        train_df = read_csv(train_path, self.data_col)
        test_df = read_csv(test_path, self.data_col)

        if self.target_subjectid is not None:
            train_df = train_df[train_df["subject_id"] == self.target_subjectid]

        oof_df = train_df[[self.data_col, self.target_col, self.fold_col]].copy()
        oof_df[list(self.target_dict.keys()) + ["pred"]] = -1.0

        return train_df, test_df, oof_df
