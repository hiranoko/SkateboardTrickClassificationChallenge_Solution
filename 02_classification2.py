from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import CFG
from src.constants import SUBJECT_IDS
from src.dataset.csv import read_csv
from src.model.encoder2 import Encoder1D
from src.runner.base import run_fold
from src.utils.util import seed_everything


class CustomConfig(CFG):
    @staticmethod
    def train_preprocess(signal: np.ndarray):
        return signal

    @staticmethod
    def test_preprocess(signal: np.ndarray):
        return signal

    @staticmethod
    def model_parameters():
        return {
            "in_channels": 72,
            "num_classes": 3,
            "activation": "leaky_relu",
            "use_gap": True,
            "hidden_channels": 576,
            "kernel_size": 7,
            "conv_type": "separable",
            "pooling_type": "both",
            "num_layers": 2,
        }

    @staticmethod
    def training_parameters():
        return {"lr": 1e-3, "epochs": 20}


if __name__ == "__main__":
    exp_name = "encoder1d"
    target = "classification2"

    for subject_id in SUBJECT_IDS:
        print(f"Subject ID : {subject_id}")
        cfg = CustomConfig(
            exp_name=exp_name, target_subjectid=subject_id, target=target
        )
        seed_everything(42)
        working_dir = Path(f"./output/{target}/{cfg.exp_name}/{cfg.target_subjectid}")

        # Train and validate for each fold
        for fold in range(len(cfg.fold_dict)):
            model = Encoder1D(**cfg.model_parameters())
            run_fold(
                cfg,
                model,
                cfg.train_df.copy(),
                fold=fold,
                working_dir=working_dir,
                silent=True,
                visualize=False,
            )

    test_df = read_csv(f"./input/{target}/test.csv", "npy")
    working_dir = Path(f"./output/{target}/{exp_name}")

    # モデルの初期化と設定
    model = Encoder1D(**cfg.model_parameters())
    model.to(cfg.device)
    model.eval()

    pred_results = []

    # モデルごとの予測処理
    for idx, row in test_df.iterrows():
        npy_path = row["npy"]
        subject_id = row["subject_id"]

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
