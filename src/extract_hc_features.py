import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_data import collect_all
from audio_utils import load_audio
from dsp.handcraft import extract_all_ml_features

SR = 16000

FEATURE_NAMES = [
    "rms_mean", "rms_std", 
    "zcr_mean", "zcr_std", 
    "centroid_mean", "centroid_std", 
    "rolloff_mean", "rolloff_std", 
    "flatness_mean", "flatness_std"
]

def process_group(files, label, prefix):
    X, y = [], []
    for i, f in enumerate(files): 
        try:
            audio, sr = load_audio(f, SR)
            vec = extract_all_ml_features(audio, sr)
            
            X.append(vec)
            y.append(label)
            if i == 0:
                print(f"\nĐặc trưng mẫu của file {prefix}:")
                for name, value in zip(FEATURE_NAMES, vec):
                    print(f"  {name}: {value:.4f}")

        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    return np.array(X), np.array(y)
def save_split(X, y, out_csv):
    cols = FEATURE_NAMES + ["label"]
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

def main(machine_type):
    data = collect_all(machine_type)

    out_dir = f"features/handcrafted/{machine_type}"
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()

    X_train, y_train = process_group(data["train_normal"], 0, "train_normal")
    print(f"X_train shape: {X_train.shape}")

    if len(X_train) == 0:
        print("[WARN] No training data found.")
        return

    X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"))

    X_test_n, y_test_n = process_group(data["test_normal"], 0, "test_normal")
    if len(X_test_n) > 0:
        X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"))
    X_test_a, y_test_a = process_group(data["test_abnormal"], 1, "test_abnormal")
    if len(X_test_a) > 0:
        X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"))

    print(f"Successfully processed features for: {machine_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)