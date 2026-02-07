
"""
Paper-style Feature Extractor
============================

Extract feature kiểu chuẩn trong ASD paper:

    Log-Mel Spectrogram + mean/std pooling

Pipeline:
    waveform
      → STFT
      → Mel filterbank (n_mels)
      → log compression
      → mean + std over time
      → feature vector

Output:
    features/logmel/{machine}/train_normal.csv
    features/logmel/{machine}/test_normal.csv
    features/logmel/{machine}/test_abnormal.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
except:
    from load_data import collect_all
    from audio_utils import load_audio



# Log-Mel computation


def compute_logmel(y, sr, n_fft=1024, hop=512, n_mels=128):

    import librosa

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2


    mel = librosa.feature.melspectrogram(
        S=S, sr=sr, n_mels=n_mels
    )

    logmel = librosa.power_to_db(mel)

    return logmel.T  


def extract_logmel_vector(filepath, sr, n_fft, hop, n_mels):
    y, sr = load_audio(filepath, sr_target=sr)

    logmel = compute_logmel(
        y, sr,
        n_fft=n_fft,
        hop=hop,
        n_mels=n_mels
    )

    mean_vec = np.mean(logmel, axis=0)
    std_vec  = np.std(logmel, axis=0)

    return np.concatenate([mean_vec, std_vec])

#  Main


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--machine", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)

    parser.add_argument("--scale_in_extract", action="store_true")

    args = parser.parse_args()

    # Load config if exists
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        args.machine = cfg.get("machine", args.machine)

        logmel_cfg = cfg.get("logmel", {})
        args.sr = logmel_cfg.get("sr", args.sr)
        args.n_fft = logmel_cfg.get("n_fft", args.n_fft)
        args.hop = logmel_cfg.get("hop", args.hop)
        args.n_mels = logmel_cfg.get("n_mels", args.n_mels)

    # Load data
    data = collect_all(args.machine)

    train_files = data["train_normal"]
    test_norm   = data["test_normal"]
    test_abn    = data["test_abnormal"]

    print("✅ Extracting Log-Mel...")

    def extract(paths):
        feats = []
        for p in paths:
            feats.append(
                extract_logmel_vector(
                    p,
                    sr=args.sr,
                    n_fft=args.n_fft,
                    hop=args.hop,
                    n_mels=args.n_mels
                )
            )
        return np.vstack(feats)

    train_feat = extract(train_files)
    test_norm_feat = extract(test_norm)
    test_abn_feat = extract(test_abn)

    # Optional scaling
    if args.scale_in_extract:
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_feat)
        test_norm_feat = scaler.transform(test_norm_feat)
        test_abn_feat = scaler.transform(test_abn_feat)

    # Save
    out_dir = Path("features") / "logmel" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_feat).to_csv(out_dir/"train_normal.csv", index=False)
    pd.DataFrame(test_norm_feat).to_csv(out_dir/"test_normal.csv", index=False)
    pd.DataFrame(test_abn_feat).to_csv(out_dir/"test_abnormal.csv", index=False)

    print("Saved Log-Mel features to:", out_dir)
    print("Train shape:", train_feat.shape)


if __name__ == "__main__":
    main()
