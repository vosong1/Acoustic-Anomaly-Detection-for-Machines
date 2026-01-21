# extract_mfcc.py
import argparse
import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from load_data import collect_all

SR = 16000
N_MFCC = 20
MAX_LEN = 400
logging.basicConfig(
    filename="logs/mfcc.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


def mfcc_to_vector(mfcc):
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std])


def plot_mfcc(mfcc, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_group(files, label, plot_dir, limit_plot=3):
    X, y = [], []

    for i, f in enumerate(files):
        try:
            mfcc = extract_mfcc(f)
            vec = mfcc_to_vector(mfcc)

            X.append(vec)
            y.append(label)

            if i < limit_plot:
                fname = os.path.basename(f).replace(".wav", ".png")
                plot_mfcc(mfcc, f"{label}-{fname}",
                          os.path.join(plot_dir, fname))

        except Exception as e:
            logging.info(f"{f} ERROR {str(e)}")

    return np.array(X), np.array(y)


def save_split(X, y, out_csv):
    cols = [f"mfcc_mean_{i}" for i in range(N_MFCC)] + \
           [f"mfcc_std_{i}" for i in range(N_MFCC)] + ["label"]

    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


def main(machine_type):
    data = collect_all(machine_type)

    out_dir = f"features/mfcc/{machine_type}"
    plot_dir = f"results/mfcc_plots/{machine_type}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    outputs = {
        "train_normal": "train_normal.csv",
        "test_normal": "test_normal.csv",
        "test_abnormal": "test_abnormal.csv"
    }

    scaler = StandardScaler()

    X_train, y_train = process_group(
        data["train_normal"], 0, plot_dir
    )
    X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train,
               os.path.join(out_dir, outputs["train_normal"]))
    X_test_n, y_test_n = process_group(
        data["test_normal"], 0, plot_dir
    )
    X_test_n = scaler.transform(X_test_n)
    save_split(X_test_n, y_test_n,
               os.path.join(out_dir, outputs["test_normal"]))
    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1, plot_dir
    )
    X_test_a = scaler.transform(X_test_a)
    save_split(X_test_a, y_test_a,
               os.path.join(out_dir, outputs["test_abnormal"]))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="valve")
    args = parser.parse_args()
    main(args.machine)
