import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from .load_data import make_split, DEFAULT_SNR_LIST


def extract_logmel_vector(
    wav_path: Path,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    stats=None,
) -> np.ndarray:
    """
    Extract log-mel statistics vector from wav file.

    stats supported:
    ["mean", "std", "max", "min", "p25", "p75"]
    """
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel)

    if stats is None:
        stats = ["mean", "std"]

    stat_funcs = {
        "mean": lambda x: x.mean(axis=1),
        "std": lambda x: x.std(axis=1),
        "max": lambda x: x.max(axis=1),
        "min": lambda x: x.min(axis=1),
        "p25": lambda x: np.percentile(x, 25, axis=1),
        "p75": lambda x: np.percentile(x, 75, axis=1),
    }

    feats = []
    for s in stats:
        if s not in stat_funcs:
            raise ValueError(
                f"Unsupported stat: {s}. Supported: {list(stat_funcs.keys())}"
            )
        feats.append(stat_funcs[s](logmel))

    feat = np.concatenate(feats, axis=0)
    return feat.astype(np.float32)


def build_dataframe(files, **kwargs) -> pd.DataFrame:
    rows = []
    for p in files:
        feat = extract_logmel_vector(p, **kwargs)
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

    # default logmel params
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--stats", nargs="*", default=["mean", "std"])

    ap.add_argument("--train_snr", nargs="*", default=None)
    ap.add_argument("--test_snr", nargs="*", default=None)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # ===============================
    # Load config
    # ===============================
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # override top-level keys
        for k, v in cfg.items():
            setattr(args, k, v)

        # nested logmel config
        lm_cfg = cfg.get("logmel", cfg.get("log_mel", {}))
        if isinstance(lm_cfg, dict) and lm_cfg:
            args.sr = lm_cfg.get("sr", args.sr)
            args.n_fft = lm_cfg.get("n_fft", args.n_fft)

            if "hop" in lm_cfg:
                args.hop_length = lm_cfg["hop"]
            args.hop_length = lm_cfg.get("hop_length", args.hop_length)

            args.n_mels = lm_cfg.get("n_mels", args.n_mels)
            args.stats = lm_cfg.get("stats", args.stats)

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

    kwargs = dict(
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        stats=args.stats,
    )

    df_train = build_dataframe(split.train_normal, **kwargs)
    df_test_n = build_dataframe(split.test_normal, **kwargs)
    df_test_a = build_dataframe(split.test_abnormal, **kwargs)

    out_dir = Path(args.out_root) / "logmel" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(out_dir / "train_normal.csv", index=False)
    df_test_n.to_csv(out_dir / "test_normal.csv", index=False)
    df_test_a.to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"\n✅ Saved to: {out_dir}")


if __name__ == "__main__":
    main()