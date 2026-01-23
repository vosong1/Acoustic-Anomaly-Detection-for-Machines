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
N_CHROMA = 12
MAX_LEN = 400

logging.basicConfig(
    filename="logs/chroma.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
def extract_chroma(file_path):
    y, sr = librosa.load(file_path, sr=SR, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)

    if chroma.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - chroma.shape[1]
        chroma = np.pad(chroma, ((0, 0), (0, pad_width)), mode="constant")
    else:
        chroma = chroma[:, :MAX_LEN]

    return chroma
def chroma_to_vector(chroma):
    mean = chroma.mean(axis=1)
    std = chroma.std(axis=1)
    return np.concatenate([mean, std])   
def plot_chroma(chroma, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis="time", y_axis="chroma")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def process_group(files, label, plot_dir, limit_plot=50):
    X, y = [], []

    for i, f in enumerate(files):
        try:
            chroma = extract_chroma(f)
            vec = chroma_to_vector(chroma)

            X.append(vec)
            y.append(label)

            if i < limit_plot:
                base = os.path.basename(f).replace(".wav", "")
                parent = os.path.basename(os.path.dirname(f))
                fname = f"{parent}_{base}.png"

                plot_chroma(
                    chroma,
                    f"{label}-{parent}-{base}",
                    os.path.join(plot_dir, fname)
                )

        except Exception as e:
            logging.info(f"{f} ERROR {str(e)}")

    return np.array(X), np.array(y)
def save_split(X, y, out_csv):
    cols = [f"chroma_mean_{i}" for i in range(N_CHROMA)] + \
           [f"chroma_std_{i}" for i in range(N_CHROMA)] + ["label"]

    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
def main(machine_type):
    data = collect_all(machine_type)

    out_dir = f"features/chroma/{machine_type}"
    plot_base = f"results/chroma_plots/{machine_type}"
    log_dir = "logs"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_base, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    plot_dirs = {
        "train_normal": os.path.join(plot_base, "train_normal"),
        "test_normal": os.path.join(plot_base, "test_normal"),
        "test_abnormal": os.path.join(plot_base, "test_abnormal"),
    }
    for d in plot_dirs.values():
        os.makedirs(d, exist_ok=True)
    outputs = {
        "train_normal": "train_normal.csv",
        "test_normal": "test_normal.csv",
        "test_abnormal": "test_abnormal.csv"
    }
    scaler = StandardScaler()
    X_train, y_train = process_group(
        data["train_normal"], 0, plot_dirs["train_normal"]
    )
    X_train = scaler.fit_transform(X_train)
    save_split(
        X_train, y_train,
        os.path.join(out_dir, outputs["train_normal"])
    )
    X_test_n, y_test_n = process_group(
        data["test_normal"], 0, plot_dirs["test_normal"]
    )
    X_test_n = scaler.transform(X_test_n)
    save_split(
        X_test_n, y_test_n,
        os.path.join(out_dir, outputs["test_normal"])
    )

    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1, plot_dirs["test_abnormal"]
    )
    X_test_a = scaler.transform(X_test_a)
    save_split(
        X_test_a, y_test_a,
        os.path.join(out_dir, outputs["test_abnormal"])
    )

    print("Done Chroma extraction for:", machine_type)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="valve")
    args = parser.parse_args()
    main(args.machine)
