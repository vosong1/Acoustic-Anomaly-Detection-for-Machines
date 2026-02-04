import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler

# Robust imports: chạy được cả "python -m src.extract_mfcc" lẫn "python src/extract_mfcc.py"
try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio  # cần có trong project của bạn
    from src.dsp.dsp_mfcc import compute_mfcc
except Exception:
    from load_data import collect_all
    from audio_utils import load_audio
    from dsp.dsp_mfcc import compute_mfcc

def plot_waveform(y, sr, title, save_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
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

def mfcc_to_vector(mfcc: np.ndarray) -> np.ndarray:
    """mfcc shape: (T, n_mfcc) -> vector: [mean(n_mfcc), std(n_mfcc)]"""
    mean = mfcc.mean(axis=0)
    std = mfcc.std(axis=0)
    return np.concatenate([mean, std])

def process_group(files, label, plot_dir, prefix, sr, n_mfcc, frame_size, hop_size, n_fft, max_plot):
    X, y = [], []
    os.makedirs(plot_dir, exist_ok=True)

    for i, f in enumerate(files):
        try:
            audio, _sr = load_audio(f, sr)

            mfcc = compute_mfcc(
                audio,
                _sr,
                n_mfcc=n_mfcc,
                frame_size=frame_size,
                hop_size=hop_size,
                n_fft=n_fft,
            )

            if mfcc.ndim != 2:
                raise ValueError(f"MFCC shape invalid: {mfcc.shape}")

            # đảm bảo mfcc có shape (T, n_mfcc)
            if mfcc.shape[1] != n_mfcc and mfcc.shape[0] == n_mfcc:
                mfcc = mfcc.T

            vec = mfcc_to_vector(mfcc)
            X.append(vec)
            y.append(label)

            if i < max_plot:
                base = f"{prefix}_{i:04d}"
                plot_waveform(audio, _sr, f"{base} waveform", os.path.join(plot_dir, f"{base}_waveform.png"))
                plot_mfcc(mfcc, f"{base} MFCC", os.path.join(plot_dir, f"{base}_mfcc.png"))

        except Exception as e:
            print("[ERROR]", f, e)

    return np.array(X), np.array(y)

def save_split(X, y, out_csv, n_mfcc):
    cols = (
        [f"mfcc_mean_{i}" for i in range(n_mfcc)] +
        [f"mfcc_std_{i}" for i in range(n_mfcc)] +
        ["label"]
    )

    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

def apply_config(args):
    """Override argparse args bằng JSON config (nếu có)."""
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.machine = cfg.get("machine", args.machine)
    args.feature = cfg.get("feature_mfcc", args.feature)  # optional shortcut

    mfcc_cfg = cfg.get("mfcc", {}) if isinstance(cfg.get("mfcc", {}), dict) else {}
    args.sr = mfcc_cfg.get("sr", args.sr)
    args.n_mfcc = mfcc_cfg.get("n_mfcc", args.n_mfcc)
    args.frame_size = mfcc_cfg.get("frame_size", args.frame_size)
    args.hop_size = mfcc_cfg.get("hop_size", args.hop_size)
    args.n_fft = mfcc_cfg.get("n_fft", args.n_fft)
    args.max_plot = mfcc_cfg.get("max_plot", args.max_plot)

    # Cho phép bật/tắt scale trong extract từ config
    if "scale_in_extract" in mfcc_cfg:
        args.scale_in_extract = bool(mfcc_cfg["scale_in_extract"])

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

    X_train, y_train = process_group(
        data["train_normal"], 0,
        plot_dirs["train_normal"], "normal",
        sr=args.sr, n_mfcc=args.n_mfcc,
        frame_size=args.frame_size, hop_size=args.hop_size, n_fft=args.n_fft,
        max_plot=args.max_plot
    )
    print("X_train shape:", X_train.shape)

    if len(X_train) == 0:
        print("[WARN] No train_normal data")
        return

    if args.scale_in_extract:
        X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"), args.n_mfcc)

    X_test_n, y_test_n = process_group(
        data["test_normal"], 0,
        plot_dirs["test_normal"], "normal",
        sr=args.sr, n_mfcc=args.n_mfcc,
        frame_size=args.frame_size, hop_size=args.hop_size, n_fft=args.n_fft,
        max_plot=args.max_plot
    )
    print("X_test_normal shape:", X_test_n.shape)

    if len(X_test_n) > 0:
        if args.scale_in_extract:
            X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"), args.n_mfcc)
    else:
        print("[WARN] No test_normal data")

    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1,
        plot_dirs["test_abnormal"], "abnormal",
        sr=args.sr, n_mfcc=args.n_mfcc,
        frame_size=args.frame_size, hop_size=args.hop_size, n_fft=args.n_fft,
        max_plot=args.max_plot
    )
    print("X_test_abnormal shape:", X_test_a.shape)

    if len(X_test_a) > 0:
        if args.scale_in_extract:
            X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"), args.n_mfcc)
    else:
        print("[WARN] No test_abnormal data")

    print(f"Done MFCC for: {args.machine} | feature={args.feature}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file JSON config")
    parser.add_argument("--machine", type=str, default="fan")
    parser.add_argument("--feature", type=str, default="mfcc",
                        help="Tên folder feature (ví dụ: mfcc_n11_fs1024_h512_fft1024)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=11)
    parser.add_argument("--frame_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=512)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--max_plot", type=int, default=0)
    parser.add_argument("--scale_in_extract", action="store_true",
                        help="Nếu bật, sẽ StandardScaler ngay ở bước extract (mặc định: tắt).")
    args = parser.parse_args()
    main(args)
