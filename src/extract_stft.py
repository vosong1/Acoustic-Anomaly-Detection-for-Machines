# src/extract_stft.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .load_data import make_split, DEFAULT_SNR_LIST


def extract_stft_vector(wav_path: Path, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    S = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    logmag = np.log1p(mag)

    mean = logmag.mean(axis=1)
    std = logmag.std(axis=1)
    feat = np.concatenate([mean, std], axis=0)
    return feat.astype(np.float32)


def build_dataframe(files, **kwargs) -> pd.DataFrame:
    rows = []
    for p in files:
        feat = extract_stft_vector(p, **kwargs)
        row = {"path": str(p)}
        for i, v in enumerate(feat.tolist()):
            row[f"f{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)

    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--out_root", default="extract/features")

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)

    ap.add_argument("--train_snr", nargs="*", default=None)
    ap.add_argument("--test_snr", nargs="*", default=None)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

        stft_cfg = cfg.get("stft", {})
        if isinstance(stft_cfg, dict) and stft_cfg:
            args.n_fft = stft_cfg.get("n_fft", args.n_fft)
            args.hop_length = stft_cfg.get("hop_length", args.hop_length)
            args.sr = stft_cfg.get("sr", args.sr)

    train_snr = args.train_snr if args.train_snr else DEFAULT_SNR_LIST
    test_snr = args.test_snr if args.test_snr else DEFAULT_SNR_LIST

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

    kwargs = dict(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length)

    df_train = build_dataframe(split.train_normal, **kwargs)
    df_test_n = build_dataframe(split.test_normal, **kwargs)
    df_test_a = build_dataframe(split.test_abnormal, **kwargs)

    out_dir = Path(args.out_root) / "stft" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()