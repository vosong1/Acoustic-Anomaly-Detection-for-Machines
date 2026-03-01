# src/extract_mfcc.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .load_data import make_split, DEFAULT_SNR_LIST


# =========================
# Feature extraction
# =========================
def extract_mfcc_vector(
    wav_path: Path,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
):
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)

    feat = np.concatenate([mean, std], axis=0)
    return feat.astype(np.float32)


def build_dataframe(file_list, **kwargs):
    rows = []

    for p in file_list:
        feat = extract_mfcc_vector(p, **kwargs)
        row = {"path": str(p)}
        for i, v in enumerate(feat.tolist()):
            row[f"f{i}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # config support
    parser.add_argument("--config", type=str, default=None)

    # basic
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--machine", default="fan")
    parser.add_argument("--out_root", default="extract/features")

    # mfcc params
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)

    # split
    parser.add_argument("--train_snr", nargs="*", default=None)
    parser.add_argument("--test_snr", nargs="*", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # =========================
    # Load config nếu có
    # =========================
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # override top-level keys
        for key, value in cfg.items():
            setattr(args, key, value)

        # nếu config có block mfcc riêng
        if "mfcc" in cfg:
            mfcc_cfg = cfg["mfcc"]
            args.sr = mfcc_cfg.get("sr", args.sr)
            args.n_mfcc = mfcc_cfg.get("n_mfcc", args.n_mfcc)
            args.n_fft = mfcc_cfg.get("n_fft", args.n_fft)

            # hỗ trợ hop_size hoặc hop_length
            if "hop_size" in mfcc_cfg:
                args.hop_length = mfcc_cfg["hop_size"]
            elif "hop_length" in mfcc_cfg:
                args.hop_length = mfcc_cfg["hop_length"]

    # =========================
    # SNR setup
    # =========================
    train_snr = args.train_snr if args.train_snr else DEFAULT_SNR_LIST
    test_snr = args.test_snr if args.test_snr else DEFAULT_SNR_LIST

    # =========================
    # Make split
    # =========================
    split = make_split(
        data_root=args.data_root,
        machine=args.machine,
        train_snr=train_snr,
        test_snr=test_snr,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f"[INFO] Train normal: {len(split.train_normal)}")
    print(f"[INFO] Test normal: {len(split.test_normal)}")
    print(f"[INFO] Test abnormal: {len(split.test_abnormal)}")

    # =========================
    # Extract features
    # =========================
    mfcc_kwargs = dict(
        sr=args.sr,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    df_train = build_dataframe(split.train_normal, **mfcc_kwargs)
    df_test_n = build_dataframe(split.test_normal, **mfcc_kwargs)
    df_test_a = build_dataframe(split.test_abnormal, **mfcc_kwargs)

    # =========================
    # Save
    # =========================
    out_dir = Path(args.out_root) / "mfcc" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()