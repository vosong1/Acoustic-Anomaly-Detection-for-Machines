import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def split_feature(feature_name):

    cfg = load_config()

    feature_root = Path(cfg["dataset"]["feature_dir"]) / feature_name
    machines = cfg["dataset"]["machines"]
    snr_levels = cfg["dataset"]["snr_levels"]

    for machine in machines:
        for snr in snr_levels + ["all_db"]:

            folder = feature_root / machine / snr

            normal_file = folder / "normal.csv"
            abnormal_file = folder / "abnormal.csv"

            if not normal_file.exists():
                continue

            print(f"Processing {feature_name} | {machine} | {snr}")

            normal_df = pd.read_csv(normal_file)

            train_df, test_normal_df = train_test_split(
                normal_df,
                test_size=0.2,
                random_state=42,
                shuffle=True
            )

            train_df.to_csv(folder / "train_normal.csv", index=False)
            test_normal_df.to_csv(folder / "test_normal.csv", index=False)

            if abnormal_file.exists():
                abnormal_df = pd.read_csv(abnormal_file)
                abnormal_df.to_csv(folder / "test_abnormal.csv", index=False)


def main():

    features = [
        "mfcc",
        "logmel",
        "stft",
        "chroma"
    ]

    for f in features:
        split_feature(f)


if __name__ == "__main__":
    main()