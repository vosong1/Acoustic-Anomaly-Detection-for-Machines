# src/extract_mfcc.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import librosa

from load_data import make_split, DEFAULT_SNR_LIST


def extract_mfcc_vector(
    wav_path: Path,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Trả về 1 vector MFCC (mean + std theo thời gian) => size = 2*n_mfcc
    """
    y, _sr = librosa.load(str(wav_path), sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # mfcc shape: (n_mfcc, T)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    feat = np.concatenate([mean, std], axis=0)
    return feat.astype(np.float32)


def build_df(file_list: List[Path], **mfcc_kwargs) -> pd.DataFrame:
    rows = []
    for p in file_list:
        feat = extract_mfcc_vector(p, **mfcc_kwargs)
        row = {"path": str(p)}
        # f0..f(2*n_mfcc-1)
        for i, v in enumerate(feat.tolist()):
            row[f"f{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw", help="vd: data/raw")
    ap.add_argument("--machine", default="fan", help="fan/valve/...")
    ap.add_argument("--out_root", default="extract/features/mfcc", help="vd: extract/features/mfcc")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)

    # chọn SNR
    ap.add_argument("--train_snr", nargs="*", default=None, help='vd: 6db 0db (mặc định: "-6db 0db 6db")')
    ap.add_argument("--test_snr", nargs="*", default=None, help='vd: -6db (mặc định: "-6db 0db 6db")')

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_root) / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    train_snr = args.train_snr if args.train_snr is not None else DEFAULT_SNR_LIST
    test_snr = args.test_snr if args.test_snr is not None else DEFAULT_SNR_LIST

    split = make_split(
        data_root=data_root,
        machine=args.machine,
        train_snr=train_snr,
        test_snr=test_snr,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    mfcc_kwargs = dict(sr=args.sr, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length)

    print(f"[INFO] train_normal: {len(split.train_normal)} files")
    print(f"[INFO] test_normal: {len(split.test_normal)} files")
    print(f"[INFO] test_abnormal: {len(split.test_abnormal)} files")
    print(f"[INFO] train_snr={train_snr} | test_snr={test_snr}")

    df_train = build_df(split.train_normal, **mfcc_kwargs)
    df_test_n = build_df(split.test_normal, **mfcc_kwargs)
    df_test_a = build_df(split.test_abnormal, **mfcc_kwargs)

    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"✅ Saved:")
    print(f" - {out_dir / 'train_normal.csv'}")
    print(f" - {out_dir / 'test_normal.csv'}")
    print(f" - {out_dir / 'test_abnormal.csv'}")


if __name__ == "__main__":
    main()