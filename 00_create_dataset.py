from pathlib import Path

import numpy as np
import pandas as pd
from pymatreader import read_mat
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


LABELS = {
    11: "frontside_kickturn",
    12: "backside_kickturn",
    13: "pumping",
    21: "frontside_kickturn",
    22: "backside_kickturn",
    23: "pumping",
}


def create_train_dataset(input_dir: str, output_dir: str):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    train_mats = sorted(list(input_dir.rglob("*.mat")))

    for train_mat in train_mats:
        print(train_mat)
        subject_id = str(train_mat.parent.name)

        data = read_mat(train_mat)

        event = pd.DataFrame(data["event"]).astype(
            {"type": int, "init_index": int, "init_time": float}
        )

        ts = pd.DataFrame(
            np.concatenate([np.array([data["times"]]), data["data"]]).T,
            columns=["Time"] + list(data["ch_labels"]),
        )
        ts["Time"] = ts["Time"] / 1000
        ts["Time"] = ts["Time"]

        signal = ts.drop(columns="Time").T.values  # (L, 72)

        for idx in tqdm(range(len(event))):
            type = event.type.iloc[idx]
            init_time = event.init_time.iloc[idx]
            init_index = event.init_index.iloc[idx]

            start_time = init_time + 0.2
            start_idx = ts[ts["Time"] >= start_time].index[0]
            data = signal[:, start_idx : start_idx + 250]

            data = data.astype(np.float32)

            if type in [11, 12, 13]:
                direction = "LED"  # LED側の傾斜台へ向かっての動作
            elif type in [21, 22, 23]:
                direction = "Laser"  # Laser側の傾斜台へ向かっての動作
            else:
                raise ValueError(f"Invalid type: {type}")

            npy_path = (
                output_dir
                / subject_id
                / direction
                / LABELS[type]
                / train_mat.stem
                / f"{init_index:03d}.npy"
            )
            npy_path.parent.mkdir(parents=True, exist_ok=True)

            assert data.shape == (72, 250), f"shape mismatch: {data.shape}"
            np.save(npy_path, data)


def create_train_csv(dataset_dir: str):
    dataset_dir = Path(dataset_dir).resolve()
    npy_paths = sorted(dataset_dir.rglob("*.npy"))

    # データフレームの初期化
    df = pd.DataFrame(npy_paths, columns=["npy"])

    # パスから情報を抽出
    df["init_index"] = df["npy"].apply(lambda x: int(x.stem))
    df["train"] = df["npy"].apply(lambda x: x.parent.name)
    df["label"] = df["npy"].apply(lambda x: x.parent.parent.name)
    df["direction"] = df["npy"].apply(lambda x: x.parent.parent.parent.name)
    df["subject_id"] = df["npy"].apply(lambda x: x.parent.parent.parent.parent.name)

    # OrdinalEncoderの使用
    ordinal_encoder = OrdinalEncoder()
    columns_to_encode = ["subject_id", "direction", "label", "train"]
    df_encoded = pd.DataFrame(
        ordinal_encoder.fit_transform(df[columns_to_encode]),
        columns=["oe_" + col for col in columns_to_encode],
        dtype=int,
    )
    df = pd.concat([df, df_encoded], axis=1)

    # CSVファイルの保存
    df.to_csv(dataset_dir.parent / "train.csv", index=False)


def create_test_dataset(input_dir: str, output_dir: str):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    test_mats = sorted(list(input_dir.rglob("*.mat")))
    output_dir.mkdir(parents=True, exist_ok=True)

    for test_mat in test_mats:
        print(test_mat)
        data = read_mat(test_mat)

        data_length, ch, signal_length = data["data"].shape
        assert ch == 72, f"Invalid channel: {ch}"
        assert signal_length == 250, f"Invalid length: {signal_length}"

        subject_id = test_mat.stem

        for idx in tqdm(range(data_length)):
            output_path = output_dir / subject_id / f"{subject_id}_{idx:03d}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not output_path.is_file():
                assert data["data"][idx].shape == (
                    72,
                    250,
                ), f"Invalid shape: {data['data'][idx].shape}"
                np.save(output_path, data["data"][idx])


def create_test_csv(dataset_dir: str):
    dataset_dir = Path(dataset_dir).resolve()

    df = pd.DataFrame(columns=["npy"])
    df["npy"] = sorted(list(dataset_dir.rglob("*.npy")))
    df["subject_id"] = df["npy"].apply(lambda x: x.parent.name)

    ordinal_encoder = OrdinalEncoder()
    df[["oe_subject_id"]] = ordinal_encoder.fit_transform(df[["subject_id"]]).astype(
        int
    )

    df.to_csv(dataset_dir.parent / "test.csv", index=False)


if __name__ == "__main__":
    create_train_dataset(
        "./data/classification1/train", "./input/classification1/train/"
    )
    create_train_csv("./input/classification1/train")

    create_train_dataset(
        "./data/classification2/train", "./input/classification2/train/"
    )
    create_train_csv("./input/classification2/train")

    create_test_dataset("./data/classification1/test", "./input/classification1/test")
    create_test_csv("./input/classification1/test")

    create_test_dataset("./data/classification2/test", "./input/classification2/test")
    create_test_csv(Path("./input/classification2/test"))
