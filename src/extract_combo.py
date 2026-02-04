#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combo Feature Extractor
======================

Extract feature vector gồm:

    MFCC(mean,std) + Handcrafted + optional raw_max

Chạy được:

    python -m src.extract_combo --config config.json
    python -m src.extract_combo --machine fan

Output:

    features/{feature}/{machine}/
        train_normal.csv
        test_normal.csv
        test_abnormal.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ✅ Import đúng project structure src/
try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
    from src.dsp.dsp_mfcc import compute_mfcc
    from src.handcraft import extract_all_ml_features
except:
    from load_data import collect_all
    from audio_utils import load_audio
    from dsp.dsp_mfcc import compute_mfcc
    from handcraft import extract_all_ml_features


# ============================================================
# ✅ Helper
# ============================================================

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_raw_max(filepath):
    """
    raw_max lấy từ waveform gốc BEFORE normalize.
    Vì load_audio() normalize max=1 nên phải đọc thẳng file wav.
    """
    import soundfile as sf

    y, _ = sf.read(filepath)
    y = np.asarray(y, dtype=np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    return float(np.max(np.abs(y)))


# ============================================================
# ✅ Combo Feature Vector
# ============================================================

def get_combo_vector(
    filepath,
    sr,
    n_mfcc,
    frame_size,
    hop_size,
    n_fft,
    roll_percent,
    include_raw_max=True
):
    # Load normalized audio
    y, sr = load_audio(filepath, sr_target=sr)

    # ---- MFCC ----
    mfcc = compute_mfcc(
        y,
        sr,
        n_mfcc=n_mfcc,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft
    )

    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    # ---- Handcrafted ----
    hc = extract_all_ml_features(
        y,
        sr,
        frame_size=frame_size,
        hop_size=hop_size,
        n_fft=n_fft,
        roll_percent=roll_percent
    )

    parts = [mfcc_mean, mfcc_std, hc]

    # ---- raw_max ----
    if include_raw_max:
        raw_max = compute_raw_max(filepath)
        parts.append(np.array([raw_max], dtype=np.float32))

    return np.concatenate(parts, axis=0)


# ============================================================
# ✅ Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Extract Combo Features")

    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--feature", type=str, default="combo_final")

    parser.add_argument("--config", type=str, default=None)

    # MFCC params
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--frame_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=512)
    parser.add_argument("--n_fft", type=int, default=2048)

    # Handcrafted params
    parser.add_argument("--roll_percent", type=float, default=0.85)

    # Combo options
    parser.add_argument("--no_raw_max", action="store_true")

    # Scaling
    parser.add_argument("--scale_in_extract", action="store_true")

    args = parser.parse_args()

    # ==================================================
    # ✅ Load config.json nếu có
    # ==================================================

    if args.config is not None:
        cfg = read_json(args.config)

        args.machine = cfg.get("machine", args.machine)

        mfcc_cfg = cfg.get("mfcc", {})
        hc_cfg = cfg.get("handcrafted", {})
        combo_cfg = cfg.get("combo", {})

        args.sr = mfcc_cfg.get("sr", args.sr)
        args.n_mfcc = mfcc_cfg.get("n_mfcc", args.n_mfcc)
        args.frame_size = mfcc_cfg.get("frame_size", args.frame_size)
        args.hop_size = mfcc_cfg.get("hop_size", args.hop_size)
        args.n_fft = mfcc_cfg.get("n_fft", args.n_fft)

        args.roll_percent = hc_cfg.get("roll_percent", args.roll_percent)

        if combo_cfg.get("include_raw_max") is False:
            args.no_raw_max = True

    if args.machine is None:
        raise ValueError("❌ Bạn phải truyền --machine hoặc machine trong config.json")

    include_raw_max = not args.no_raw_max

    # ==================================================
    # ✅ Load dataset file list
    # collect_all() returns dict
    # ==================================================

    data = collect_all(args.machine)

    train_files = data["train_normal"]
    test_norm_files = data["test_normal"]
    test_abn_files = data["test_abnormal"]

    print("✅ Loaded file lists:")
    print("   train:", len(train_files))
    print("   test normal:", len(test_norm_files))
    print("   test abnormal:", len(test_abn_files))

    # ==================================================
    # ✅ Extract function
    # ==================================================

    def extract(paths):
        feats = []
        for p in paths:
            vec = get_combo_vector(
                p,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                frame_size=args.frame_size,
                hop_size=args.hop_size,
                n_fft=args.n_fft,
                roll_percent=args.roll_percent,
                include_raw_max=include_raw_max
            )
            feats.append(vec)

        return np.vstack(feats)

    train_feat = extract(train_files)
    test_norm_feat = extract(test_norm_files)
    test_abn_feat = extract(test_abn_files)

    # ==================================================
    # ✅ Optional scaling
    # ==================================================

    if args.scale_in_extract:
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_feat)
        test_norm_feat = scaler.transform(test_norm_feat)
        test_abn_feat = scaler.transform(test_abn_feat)

    # ==================================================
    # ✅ Save CSV
    # ==================================================

    out_dir = Path("features") / args.feature / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_feat).to_csv(out_dir / "train_normal.csv", index=False)
    pd.DataFrame(test_norm_feat).to_csv(out_dir / "test_normal.csv", index=False)
    pd.DataFrame(test_abn_feat).to_csv(out_dir / "test_abnormal.csv", index=False)

    print("\n✅ DONE! Saved combo features:")
    print(" Folder:", out_dir)
    print(" Shape train:", train_feat.shape)
    print(" Shape test normal:", test_norm_feat.shape)
    print(" Shape test abnormal:", test_abn_feat.shape)
    print(" include_raw_max =", include_raw_max)


# ============================================================
if __name__ == "__main__":
    main()
