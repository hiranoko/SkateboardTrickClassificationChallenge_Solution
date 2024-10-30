# SkateboardTrickClassificationChallenge_Solution

[Motion Decoding Using Biosignals スケートボードトリック分類チャレンジ](https://signate.jp/competitions/1429)

解法はこちらを参照ください

## Directory Layout

```
.
├── doc
├── data # コンテストのデータセットを配置する
│   ├── classification1
│   │   ├── test
│   │   └── train
│   │       ├── subject0
│   │       ├── subject1
│   │       ├── subject2
│   │       ├── subject3
│   │       └── subject4
│   └── classification2
│       ├── test
│       └── train
│           ├── subject0
│           ├── subject1
│           ├── subject2
│           ├── subject3
│           └── subject4
├── input # data/のmatファイルをnpy形式で保管し、学習に使用する
├── notebooks
├── output
└── src
```

## Usage

### environment

conda環境を作成して有効化する。

```terminal
conda env create -f env.yml
conda activate signate
```

### Prepare dataset

提供データを`directory layout`に沿って配置する。

```
python 00_create_dataset.py
```

`src/constants.py`を修正

```python
CLASSIFICATION1_TRAIN_CSV = Path(
    "/{your_environment}/input/classification1/train.csv"
)
CLASSIFICATION1_TEST_CSV = Path(
    "/{your_environment}/input/classification1/test.csv"
)
CLASSIFICATION2_TRAIN_CSV = Path(
    "/{your_environment}/input/classification2/train.csv"
)
CLASSIFICATION2_TEST_CSV = Path(
    "/{your_environment}/input/classification2/test.csv"
)
```


### Train and Inference

```
python 01_classification1.py
python 02_classification2.py
```

### Submit

- データ1の提出は`output/classification1/submission.csv`を使用した
- データ2の提出は`output/classification2/submission.csv`を使用した

## Environment

|        |                            |
| :----: | :------------------------: |
|   OS   |     Ubuntu 24.04.1 LTS     |
|  GPU   | NVIDIA GeForce RTX 3080 Ti |
|  CUDA  |            12.1            |
| Python |          3.10.14           |