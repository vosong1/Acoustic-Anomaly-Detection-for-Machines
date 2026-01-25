import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_data import collect_all
from audio_utils import load_audio
from dsp.dsp_mfcc import compute_mfcc

SR = 16000
N_MFCC = 20


def mfcc_to_vector(mfcc):
    """
    mfcc shape: (time, n_mfcc)
    return: (2 * n_mfcc,) = (40,)
    """
    mean = mfcc.mean(axis=0)   # (20,)
    std  = mfcc.std(axis=0)    # (20,)
    return np.concatenate([mean, std])


def process_group(files, label):
    X, y = [], []

    for f in files:
        try:
            audio, sr = load_audio(f, SR)
            mfcc = compute_mfcc(audio, sr, n_mfcc=N_MFCC)

            print("RAW MFCC SHAPE:", mfcc.shape) 

            if mfcc.ndim != 2:
                raise ValueError(f"MFCC shape invalid: {mfcc.shape}")

            if mfcc.shape[0] == N_MFCC:
                mfcc = mfcc.T     
            elif mfcc.shape[1] != N_MFCC:
                raise ValueError(f"MFCC unexpected shape: {mfcc.shape}")

            vec = mfcc_to_vector(mfcc)

            if vec.shape[0] != 2 * N_MFCC:
                raise ValueError(f"Vector size invalid: {vec.shape}")

            X.append(vec)
            y.append(label)

        except Exception as e:
            print("[ERROR]", f, e)

    return np.array(X), np.array(y)

def save_split(X, y, out_csv):
    cols = (
        [f"mfcc_mean_{i}" for i in range(N_MFCC)] +
        [f"mfcc_std_{i}" for i in range(N_MFCC)] +
        ["label"]
    )

    df = pd.DataFrame(
        np.column_stack([X, y]),
        columns=cols
    )
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


def main(machine_type):
    data = collect_all(machine_type)

    out_dir = f"features/mfcc/{machine_type}"
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()

    # ---- TRAIN NORMAL ----
    X_train, y_train = process_group(data["train_normal"], 0)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    if len(X_train) == 0:
        print("[WARN] No train_normal data, skip.")
        return

    X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train,
               os.path.join(out_dir, "train_normal.csv"))

    # ---- TEST NORMAL ----
    X_test_n, y_test_n = process_group(data["test_normal"], 0)
    print("X_test_n shape:", X_test_n.shape)

    if len(X_test_n) > 0:
        X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n,
                   os.path.join(out_dir, "test_normal.csv"))
    else:
        print("[WARN] No test_normal data.")

    # ---- TEST ABNORMAL ----
    X_test_a, y_test_a = process_group(data["test_abnormal"], 1)
    print("X_test_a shape:", X_test_a.shape)

    if len(X_test_a) > 0:
        X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a,
                   os.path.join(out_dir, "test_abnormal.csv"))
    else:
        print("[WARN] No test_abnormal data.")

    print("Done MFCC (hard) for:", machine_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)
