import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
import librosa
from sklearn.preprocessing import StandardScaler

try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
except Exception:
    from load_data import collect_all
    from audio_utils import load_audio

def plot_waveform(y, sr, title, save_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram(S_db, sr, hop_length, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_stft_magnitude(y, n_fft, hop_length):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return np.abs(S)

def stft_to_vector(S_mag):
    # log scale (dB)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    mean = np.mean(S_db, axis=1)
    std = np.std(S_db, axis=1)
    return np.concatenate([mean, std])

def process_group(files, label, plot_dir, prefix, sr, n_fft, hop_length, max_plot):
    X, y = [], []
    os.makedirs(plot_dir, exist_ok=True)

    for i, f in enumerate(files):
        try:
            audio, _sr = load_audio(f, sr)
            S_mag = compute_stft_magnitude(audio, n_fft=n_fft, hop_length=hop_length)
            vec = stft_to_vector(S_mag)
            X.append(vec)
            y.append(label)

            if i < max_plot:
                base = f"{prefix}_{i:04d}"
                plot_waveform(audio, _sr, f"{base} waveform", os.path.join(plot_dir, f"{base}_waveform.png"))
                S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
                plot_spectrogram(S_db, _sr, hop_length, f"{base} spectrogram", os.path.join(plot_dir, f"{base}_spectrogram.png"))

        except Exception as e:
            print(f"[ERROR] File: {f} - {e}")

    return np.array(X), np.array(y)

def save_split(X, y, out_csv, n_bins):
    cols = (
        [f"freq_mean_{i}" for i in range(n_bins)] +
        [f"freq_std_{i}" for i in range(n_bins)] +
        ["label"]
    )
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

def apply_config(args):
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.machine = cfg.get("machine", args.machine)
    args.feature = cfg.get("feature_stft", args.feature)

    stft_cfg = cfg.get("stft", {}) if isinstance(cfg.get("stft", {}), dict) else {}
    args.sr = stft_cfg.get("sr", args.sr)
    args.n_fft = stft_cfg.get("n_fft", args.n_fft)
    args.hop_length = stft_cfg.get("hop_length", args.hop_length)
    args.max_plot = stft_cfg.get("max_plot", args.max_plot)

    if "scale_in_extract" in stft_cfg:
        args.scale_in_extract = bool(stft_cfg["scale_in_extract"])

    return args

def main(args):
    args = apply_config(args)
    data = collect_all(args.machine)

    out_dir = os.path.join("features", args.feature, args.machine)
    plot_base = os.path.join("results", args.feature, args.machine)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_base, exist_ok=True)

    plot_dirs = {
        "train_normal": os.path.join(plot_base, "train_normal"),
        "test_normal": os.path.join(plot_base, "test_normal"),
        "test_abnormal": os.path.join(plot_base, "test_abnormal"),
    }

    scaler = StandardScaler()
    n_bins = args.n_fft // 2 + 1

    print("Processing Train Normal...")
    X_train, y_train = process_group(
        data["train_normal"], 0, plot_dirs["train_normal"], "normal",
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, max_plot=args.max_plot
    )

    if len(X_train) == 0:
        print("[WARN] No train_normal data")
        return

    if args.scale_in_extract:
        X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"), n_bins)

    print("Processing Test Normal...")
    X_test_n, y_test_n = process_group(
        data["test_normal"], 0, plot_dirs["test_normal"], "normal",
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, max_plot=args.max_plot
    )
    if len(X_test_n) > 0:
        if args.scale_in_extract:
            X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"), n_bins)

    print("Processing Test Abnormal...")
    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1, plot_dirs["test_abnormal"], "abnormal",
        sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, max_plot=args.max_plot
    )
    if len(X_test_a) > 0:
        if args.scale_in_extract:
            X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"), n_bins)

    print(f"Done STFT for: {args.machine} | feature={args.feature} | vector_dim={X_train.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file JSON config")
    parser.add_argument("--machine", type=str, default="fan")
    parser.add_argument("--feature", type=str, default="stft",
                        help="Tên folder feature (ví dụ: stft_fft1024_h512)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--max_plot", type=int, default=0)
    parser.add_argument("--scale_in_extract", action="store_true",
                        help="Nếu bật, sẽ StandardScaler ngay ở bước extract (mặc định: tắt).")
    args = parser.parse_args()
    main(args)
