from pathlib import Path

CLASSIFICATION1_TRAIN_CSV = Path(
    "/home/kota/Contests/SkateboardTrickClassificationChallenge_Solution/input/classification1/train.csv"
)
CLASSIFICATION1_TEST_CSV = Path(
    "/home/kota/Contests/SkateboardTrickClassificationChallenge_Solution/input/classification1/test.csv"
)
CLASSIFICATION2_TRAIN_CSV = Path(
    "/home/kota/Contests/SkateboardTrickClassificationChallenge_Solution/input/classification2/train.csv"
)
CLASSIFICATION2_TEST_CSV = Path(
    "/home/kota/Contests/SkateboardTrickClassificationChallenge_Solution/input/classification2/test.csv"
)

assert CLASSIFICATION1_TRAIN_CSV.exists()
assert CLASSIFICATION1_TEST_CSV.exists()
assert CLASSIFICATION2_TRAIN_CSV.exists()
assert CLASSIFICATION2_TEST_CSV.exists()

SUBJECT_IDS = [
    "subject0",
    "subject1",
    "subject2",
    "subject3",
    "subject4",
]

LABELS = {
    11: "frontside_kickturn",
    12: "backside_kickturn",
    13: "pumping",
    21: "frontside_kickturn",
    22: "backside_kickturn",
    23: "pumping",
}

CHANNEL_COLUMNS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
    "FC5",
    "FC6",
    "CP5",
    "CP6",
    "FT9",
    "FT10",
    "FCz",
    "AFz",
    "F1",
    "F2",
    "C1",
    "C2",
    "P1",
    "P2",
    "AF3",
    "AF4",
    "FC3",
    "FC4",
    "CP3",
    "CP4",
    "PO3",
    "PO4",
    "F5",
    "F6",
    "C5",
    "C6",
    "P5",
    "P6",
    "AF7",
    "AF8",
    "FT7",
    "FT8",
    "TP7",
    "TP8",
    "PO7",
    "PO8",
    "Fpz",
    "CPz",
    "POz",
    "Iz",
    "F9",
    "F10",
    "P9",
    "P10",
    "PO9",
    "PO10",
    "O9",
    "O10",
]

assert len(CHANNEL_COLUMNS) == 72
