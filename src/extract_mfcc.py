import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler

from load_data import collect_all
from audio_utils import load_audio
from dsp.dsp_mfcc import compute_mfcc

SR = 16000
N_MFCC = 20
MAX_PLOT = 50  # số file tối đa để vẽ ảnh mỗi group


# =========================
# Plot helpers
# =========================
def plot_waveform(y, sr, title, save_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_spectrogram(y, sr, title, save_path):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_mfcc(mfcc, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis="time")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# Feature helpers
# =========================
def mfcc_to_vector(mfcc):
    """
    mfcc shape: (T, N_MFCC)
    output: (2 * N_MFCC,)
    """
    mean = mfcc.mean(axis=0)
    std = mfcc.std(axis=0)
    return np.concatenate([mean, std])


def process_group(files, label, plot_dir, prefix):
    X, y = [], []

    os.makedirs(plot_dir, exist_ok=True)

    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(f, SR)
            mfcc = compute_mfcc(audio, sr, n_mfcc=N_MFCC)

            if mfcc.ndim != 2:
                raise ValueError(f"MFCC shape invalid: {mfcc.shape}")

            # đảm bảo mfcc = (T, N_MFCC)
            if mfcc.shape[1] != N_MFCC and mfcc.shape[0] == N_MFCC:
                mfcc = mfcc.T

            vec = mfcc_to_vector(mfcc)
            X.append(vec)
            y.append(label)

            # ====== SAVE PLOTS ======
            if i < MAX_PLOT:
                base = f"{prefix}_{i:04d}"

                plot_waveform(
                    audio, sr,
                    f"{base} waveform",
                    os.path.join(plot_dir, f"{base}_waveform.png")
                )

                plot_spectrogram(
                    audio, sr,
                    f"{base} spectrogram",
                    os.path.join(plot_dir, f"{base}_spectrogram.png")
                )

                plot_mfcc(
                    mfcc,
                    f"{base} MFCC",
                    os.path.join(plot_dir, f"{base}_mfcc.png")
                )

        except Exception as e:
            print("[ERROR]", f, e)

    return np.array(X), np.array(y)


def save_split(X, y, out_csv):
    cols = (
        [f"mfcc_mean_{i}" for i in range(N_MFCC)] +
        [f"mfcc_std_{i}" for i in range(N_MFCC)] +
        ["label"]
    )

    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


# =========================
# Main
# =========================
def main(machine_type):
    data = collect_all(machine_type)

    out_dir = f"features/mfcc/{machine_type}"
    plot_base = f"results/mfcc/{machine_type}"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_base, exist_ok=True)

    plot_dirs = {
        "train_normal": os.path.join(plot_base, "train_normal"),
        "test_normal": os.path.join(plot_base, "test_normal"),
        "test_abnormal": os.path.join(plot_base, "test_abnormal"),
    }

    scaler = StandardScaler()

    # -------- TRAIN NORMAL --------
    X_train, y_train = process_group(
        data["train_normal"], 0,
        plot_dirs["train_normal"], "normal"
    )

    print("X_train shape:", X_train.shape)

    if len(X_train) == 0:
        print("[WARN] No train_normal data")
        return

    X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"))

    # -------- TEST NORMAL --------
    X_test_n, y_test_n = process_group(
        data["test_normal"], 0,
        plot_dirs["test_normal"], "normal"
    )

    print("X_test_normal shape:", X_test_n.shape)

    if len(X_test_n) > 0:
        X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"))
    else:
        print("[WARN] No test_normal data")

    # -------- TEST ABNORMAL --------
    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1,
        plot_dirs["test_abnormal"], "abnormal"
    )

    print("X_test_abnormal shape:", X_test_a.shape)

    if len(X_test_a) > 0:
        X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"))
    else:
        print("[WARN] No test_abnormal data")

    print("Done MFCC for:", machine_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)
