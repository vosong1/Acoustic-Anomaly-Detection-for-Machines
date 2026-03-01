# src/extract_chroma.py

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
def extract_chroma_vector(
    wav_path: Path,
    sr: int,
    n_fft: int,
    hop_length: int,
):
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    mean = chroma.mean(axis=1)
    std = chroma.std(axis=1)

    feat = np.concatenate([mean, std], axis=0)
    return feat.astype(np.float32)


def build_dataframe(file_list, **kwargs):
    rows = []
    for p in file_list:
        feat = extract_chroma_vector(p, **kwargs)
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

    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--machine", default="fan")
    parser.add_argument("--out_root", default="extract/features")

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)

    parser.add_argument("--train_snr", nargs="*", default=None)
    parser.add_argument("--test_snr", nargs="*", default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ===== LOAD CONFIG =====
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        for key, value in cfg.items():
            setattr(args, key, value)

        if "stft" in cfg:
            stft_cfg = cfg["stft"]
            args.n_fft = stft_cfg.get("n_fft", args.n_fft)
            args.hop_length = stft_cfg.get("hop_length", args.hop_length)

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

    chroma_kwargs = dict(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    df_train = build_dataframe(split.train_normal, **chroma_kwargs)
    df_test_n = build_dataframe(split.test_normal, **chroma_kwargs)
    df_test_a = build_dataframe(split.test_abnormal, **chroma_kwargs)

    out_dir = Path(args.out_root) / "chroma" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()