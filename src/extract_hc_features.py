# src/extract_hc.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .load_data import make_split, DEFAULT_SNR_LIST


def extract_hc_vector(wav_path: Path, sr: int, n_fft: int, hop_length: int) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)

    feats = [rms, zcr, centroid, bandwidth, rolloff]

    vec = []
    for f in feats:
        vec.extend([float(f.mean()), float(f.std())])

    return np.array(vec, dtype=np.float32)


def build_dataframe(files, **kwargs) -> pd.DataFrame:
    rows = []
    for p in files:
        feat = extract_hc_vector(p, **kwargs)
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
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--roll_percent", type=float, default=0.85)

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

        # block handcrafted (hoặc hc)
        hc_cfg = cfg.get("handcrafted", cfg.get("hc", {}))
        if isinstance(hc_cfg, dict) and hc_cfg:
            # map hop_size -> hop_length nếu có
            if "hop_size" in hc_cfg:
                args.hop_length = hc_cfg["hop_size"]
            if "hop_length" in hc_cfg:
                args.hop_length = hc_cfg["hop_length"]
            if "n_fft" in hc_cfg:
                args.n_fft = hc_cfg["n_fft"]
            if "sr" in hc_cfg:
                args.sr = hc_cfg["sr"]
            if "roll_percent" in hc_cfg:
                args.roll_percent = hc_cfg["roll_percent"]

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

    out_dir = Path(args.out_root) / "hc" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()