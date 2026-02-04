#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combo feature extractor: MFCC (mean/std) + handcrafted + optional raw_max.

Supports:
- Running as module: python -m src.extract_combo ...
- Running as script: python src/extract_combo.py ...

Config:
- CLI args override defaults
- If --config is provided, values are loaded from JSON and applied (then CLI can still override if explicitly passed)

Outputs:
features/{feature}/{machine}/
  train_normal.csv
  test_normal.csv
  test_abnormal.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---- Robust imports (works for both "python -m src.xxx" and "python src/xxx.py") ----
try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
    from src.dsp.dsp_mfcc import compute_mfcc
    from src.handcraft import extract_all_ml_features
except Exception:
    # Fallback for running when src is on PYTHONPATH or running inside src/
    from load_data import collect_all
    from audio_utils import load_audio
    from dsp.dsp_mfcc import compute_mfcc
    from handcraft import extract_all_ml_features


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coalesce(*vals):
    """Return the first value that is not None."""
    for v in vals:
        if v is not None:
            return v
    return None


def compute_raw_max(filepath: str) -> float:
    """
    Compute raw max(abs(y)) BEFORE any normalization.
    We intentionally bypass audio_utils.load_audio() because it normalizes audio.
    """
    try:
        import soundfile as sf
        y, _sr = sf.read(filepath)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # stereo -> mono
        return float(np.max(np.abs(y))) if y.size else 0.0
    except Exception:
        # If anything goes wrong, fall back to NaN (still keeps vector length consistent).
        return float("nan")


def get_combo_vector(
    filepath: str,
    sr: int,
    n_mfcc: int,
    frame_size: int,
    hop_size: int,
    n_fft: int,
    roll_percent: float,
    include_raw_max: bool,
) -> np.ndarray:
    # Load normalized audio for feature computation (as your pipeline does)
    y, sr = load_audio(filepath, sr_target=sr)

    # MFCC -> mean/std over time
    mfcc = compute_mfcc(
        y,
        sr,
        n_mfcc=n_mfcc,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft,
    )
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    # Handcrafted
    hc = extract_all_ml_features(
        y,
        sr,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft,
        roll_percent=roll_percent,
    )

    parts = [mfcc_mean, mfcc_std, hc]

    # Optional raw_max (computed from raw waveform pre-normalization)
    if include_raw_max:
        raw_max = compute_raw_max(filepath)
        parts.append(np.array([raw_max], dtype=np.float32))

    return np.concatenate(parts, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract combo features (MFCC+handcrafted+raw_max).")

    # Common
    parser.add_argument("--machine", type=str, default=None, help="Machine type (e.g., fan, pump, slider, valve).")
    parser.add_argument("--feature", type=str, default=None, help="Feature folder name under features/ (default: combo_final).")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file.")

    # Audio / MFCC params
    parser.add_argument("--sr", type=int, default=None)
    parser.add_argument("--n_mfcc", type=int, default=None)
    parser.add_argument("--frame_size", type=int, default=None)
    parser.add_argument("--hop_size", type=int, default=None)
    parser.add_argument("--n_fft", type=int, default=None)

    # Handcrafted params
    parser.add_argument("--roll_percent", type=float, default=None)

    # Combo options
    parser.add_argument("--include_raw_max", action="store_true", help="Append raw max(abs(y)) before normalization.")
    parser.add_argument("--no_raw_max", action="store_true", help="Force disable raw_max even if config enables it.")

    # Scaling in extract
    parser.add_argument("--scale_in_extract", action="store_true", help="Apply StandardScaler to extracted vectors before saving.")

    args = parser.parse_args()

    # ---- Defaults ----
    defaults = {
        "feature": "combo_final",
        "sr": 16000,
        "n_mfcc": 13,
        "frame_size": 1024,
        "hop_size": 512,
        "n_fft": 1024,
        "roll_percent": 0.85,
        "include_raw_max": True,  # your original code included it
        "scale_in_extract": False,  # avoid double-scaling; prefer scaling in SVM
    }

    cfg = {}
    if args.config:
        cfg = _read_json(args.config)

    # Resolve machine (must exist)
    machine = _coalesce(args.machine, cfg.get("machine"), cfg.get("machine_type"))
    if not machine:
        raise SystemExit("Missing --machine (or machine in config.json).")

    # Resolve feature folder
    feature = _coalesce(args.feature, cfg.get("feature_combo"), cfg.get("feature"), defaults["feature"])

    # Pull MFCC params from config
    mfcc_cfg = cfg.get("mfcc", {})
    hc_cfg = cfg.get("handcrafted", {})
    combo_cfg = cfg.get("combo", {})

    sr = int(_coalesce(args.sr, mfcc_cfg.get("sr"), hc_cfg.get("sr"), defaults["sr"]))
    n_mfcc = int(_coalesce(args.n_mfcc, mfcc_cfg.get("n_mfcc"), defaults["n_mfcc"]))
    frame_size = int(_coalesce(args.frame_size, mfcc_cfg.get("frame_size"), hc_cfg.get("frame_size"), defaults["frame_size"]))
    hop_size = int(_coalesce(args.hop_size, mfcc_cfg.get("hop_size"), hc_cfg.get("hop_size"), defaults["hop_size"]))
    n_fft = int(_coalesce(args.n_fft, mfcc_cfg.get("n_fft"), hc_cfg.get("n_fft"), defaults["n_fft"]))
    roll_percent = float(_coalesce(args.roll_percent, hc_cfg.get("roll_percent"), defaults["roll_percent"]))

    # include_raw_max: config can set combo.include_raw_max, CLI flags can override
    include_raw_max = bool(_coalesce(
        True if args.include_raw_max else None,
        False if args.no_raw_max else None,
        combo_cfg.get("include_raw_max"),
        defaults["include_raw_max"],
    ))

    # scale_in_extract: config can set combo.scale_in_extract
    scale_in_extract = bool(_coalesce(
        True if args.scale_in_extract else None,
        combo_cfg.get("scale_in_extract"),
        defaults["scale_in_extract"],
    ))

    # ---- Load file lists ----
    train_normal, test_normal, test_abnormal = collect_all(machine)

    # ---- Extract ----
    def _extract(paths):
        feats = []
        for p in paths:
            feats.append(
                get_combo_vector(
                    p,
                    sr=sr,
                    n_mfcc=n_mfcc,
                    frame_size=frame_size,
                    hop_size=hop_size,
                    n_fft=n_fft,
                    roll_percent=roll_percent,
                    include_raw_max=include_raw_max,
                )
            )
        return np.vstack(feats) if feats else np.zeros((0, 0), dtype=np.float32)

    train_feat = _extract(train_normal)
    test_norm_feat = _extract(test_normal)
    test_abn_feat = _extract(test_abnormal)

    # ---- Optional scaling (NOT recommended if you also scale in SVM) ----
    if scale_in_extract and train_feat.size:
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_feat)
        test_norm_feat = scaler.transform(test_norm_feat) if test_norm_feat.size else test_norm_feat
        test_abn_feat = scaler.transform(test_abn_feat) if test_abn_feat.size else test_abn_feat

    # ---- Save ----
    out_dir = Path("features") / feature / machine
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_feat).to_csv(out_dir / "train_normal.csv", index=False)
    pd.DataFrame(test_norm_feat).to_csv(out_dir / "test_normal.csv", index=False)
    pd.DataFrame(test_abn_feat).to_csv(out_dir / "test_abnormal.csv", index=False)

    print(f"[OK] Saved combo features to: {out_dir}")
    print(f"  train_normal:  {train_feat.shape}")
    print(f"  test_normal:   {test_norm_feat.shape}")
    print(f"  test_abnormal: {test_abn_feat.shape}")
    print(f"  include_raw_max={include_raw_max} (raw_max computed pre-normalization)")
    print(f"  mfcc: sr={sr}, n_mfcc={n_mfcc}, frame={frame_size}, hop={hop_size}, n_fft={n_fft}")
    print(f"  handcrafted: roll_percent={roll_percent}")
    print(f"  scale_in_extract={scale_in_extract}")


if __name__ == "__main__":
    main()
