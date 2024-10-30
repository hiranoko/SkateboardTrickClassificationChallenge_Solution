from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import SUBJECT_IDS
from src.config import CFG
from src.dataset.csv import read_csv
from src.dataset.eeg import butter_bandpass_filter
from src.model.encoder import Encoder1D
from src.runner.base import run_fold
from src.utils.util import seed_everything


class CustomConfig(CFG):
    def get_in_channels(self):
        if self.target_subjectid == "subject0":
            return 66
        elif self.target_subjectid == "subject1":
            return 60
        elif self.target_subjectid == "subject2":
            return 67
        elif self.target_subjectid == "subject3":
            return 68
        elif self.target_subjectid == "subject4":
            return 65
        else:
            raise ValueError("Invalid target_subjectid")

    def remove_non_use_ch(self, signal: np.ndarray):
        if self.target_subjectid == "subject0":
            # 72ch -> 66ch
            signal = np.delete(signal, [13, 28, 64, 68, 70, 71], axis=0)
        elif self.target_subjectid == "subject1":
            # 72ch -> 60ch
            signal = np.delete(
                signal, [8, 12, 13, 14, 18, 39, 56, 58, 63, 66, 68, 70], axis=0
            )
        elif self.target_subjectid == "subject2":
            # 72ch -> 67ch
            signal = np.delete(signal, [39, 52, 66, 68, 70], axis=0)
            # 52 ch is large std
        elif self.target_subjectid == "subject3":
            # 72ch -> 68ch
            signal = np.delete(signal, [13, 39, 56, 70], axis=0)
        elif self.target_subjectid == "subject4":
            # 72ch -> 65ch
            signal = np.delete(signal, [11, 13, 24, 39, 65, 66, 70], axis=0)
        else:
            raise ValueError("Invalid target_subjectid")
        return signal

    def train_preprocess(self, signal: np.ndarray):
        signal = signal - signal.mean(axis=1, keepdims=True)
        signal = self.remove_non_use_ch(signal)
        signal = butter_bandpass_filter(signal, 2, 125, 500)
        return signal

    def test_preprocess(self, signal: np.ndarray):
        signal = signal - signal.mean(axis=1, keepdims=True)
        signal = self.remove_non_use_ch(signal)
        signal = butter_bandpass_filter(signal, 2, 125, 500)
        return signal

    def model_parameters(self):
        in_chanaels = self.get_in_channels()
        return {
            "in_channels": in_chanaels,
            "num_classes": 3,
            "hidden_channels": [728, 259, 253],
            "conv_types": ["standard", "standard", "standard"],
            "kernel_size": 3,
            "pooling_types": ["max", "max", "avg"],
            "activation": "leaky_relu",
        }

    @staticmethod
    def training_parameters():
        return {"lr": 1e-3, "epochs": 20}


if __name__ == "__main__":
    exp_name = "encoder1d"
    target = "classification1"

    for subject_id in SUBJECT_IDS:
        print(f"Subject ID : {subject_id}")
        seed_everything(42)
        cfg = CustomConfig(
            exp_name=exp_name, target_subjectid=subject_id, target=target
        )
        working_dir = Path(f"./output/{target}/{cfg.exp_name}/{cfg.target_subjectid}")

        # Train and validate for each fold
        for fold in range(len(cfg.fold_dict)):
            run_fold(
                cfg=cfg,
                model=Encoder1D(**cfg.model_parameters()),
                df=cfg.train_df.copy(),
                fold=fold,
                working_dir=working_dir,
                silent=True,
                visualize=False,
            )

    test_df = read_csv(f"./input/{target}/test.csv", "npy")
    working_dir = Path(f"./output/{target}/{exp_name}")

    pred_results = []
    for idx, row in test_df.iterrows():
        npy_path = row["npy"]
        subject_id = row["subject_id"]

        # モデルの初期化
        cfg = CustomConfig(
            exp_name=exp_name, target_subjectid=subject_id, target=target
        )
        model = Encoder1D(**cfg.model_parameters()).to(cfg.device)
        model.eval()

        # モデルパスの取得
        model_pathes = sorted(list((working_dir / f"{subject_id}").rglob("*.pth")))

        # 信号データの読み込みと前処理
        signal = (
            torch.tensor(cfg.test_preprocess(np.load(npy_path)).astype(np.float32))
            .unsqueeze(0)
            .to(cfg.device)
        )

        # 複数モデルの予測を蓄積
        raw_preds = []
        with torch.no_grad():
            for model_path in model_pathes:
                model.load_state_dict(torch.load(model_path))
                pred = model(signal).cpu()  # 予測を取得
                raw_preds.append(pred)

        # 平均値を計算して分類
        raw_preds = torch.stack(raw_preds).mean(0).softmax(-1).numpy()[0]
        pred_cls = np.argmax(raw_preds)
        pred_str = list(cfg.target_dict.keys())[pred_cls]

        # 結果を保存
        pred_results.append(
            [str(npy_path.stem), raw_preds[0], raw_preds[1], raw_preds[2], pred_str]
        )

    # 結果を保存
    submit_df = pd.DataFrame(
        pred_results,
        columns=["npy", "backside_kickturn", "frontside_kickturn", "pumping", "pred"],
    )
    submit_df[["npy", "pred"]].to_csv(
        working_dir / "submission.csv", index=False, header=False
    )
