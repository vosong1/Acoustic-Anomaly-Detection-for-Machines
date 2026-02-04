import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from src.load_data import collect_all
    from src.audio_utils import load_audio
    from src.handcraft import extract_all_ml_features
except Exception:
    from load_data import collect_all
    from audio_utils import load_audio
    from handcraft import extract_all_ml_features

FEATURE_NAMES = [
    "rms_mean", "rms_std",
    "zcr_mean", "zcr_std",
    "centroid_mean", "centroid_std",
    "rolloff_mean", "rolloff_std",
    "flatness_mean", "flatness_std"
]

def process_group(files, label, sr, frame_size, hop_size, n_fft, roll_percent, show_example=False):
    X, y = [], []
    for i, f in enumerate(files):
        try:
            audio, _sr = load_audio(f, sr)
            vec = extract_all_ml_features(
                audio, _sr,
                frame_size=frame_size,
                hop_size=hop_size,
                n_fft=n_fft,
                roll_percent=roll_percent
            )

            X.append(vec)
            y.append(label)

            if show_example and i == 0:
                print(f"\nĐặc trưng mẫu của file đầu tiên:")
                for name, value in zip(FEATURE_NAMES, vec):
                    print(f"  {name}: {value:.4f}")

        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    return np.array(X), np.array(y)

def save_split(X, y, out_csv):
    cols = FEATURE_NAMES + ["label"]
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

def apply_config(args):
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.machine = cfg.get("machine", args.machine)
    args.feature = cfg.get("feature_handcrafted", args.feature)

    hc_cfg = cfg.get("handcrafted", {}) if isinstance(cfg.get("handcrafted", {}), dict) else {}
    args.sr = hc_cfg.get("sr", args.sr)
    args.frame_size = hc_cfg.get("frame_size", args.frame_size)
    args.hop_size = hc_cfg.get("hop_size", args.hop_size)
    args.n_fft = hc_cfg.get("n_fft", args.n_fft)
    args.roll_percent = hc_cfg.get("roll_percent", args.roll_percent)

    if "scale_in_extract" in hc_cfg:
        args.scale_in_extract = bool(hc_cfg["scale_in_extract"])

    return args

def main(args):
    args = apply_config(args)

    data = collect_all(args.machine)

    out_dir = os.path.join("features", args.feature, args.machine)
    os.makedirs(out_dir, exist_ok=True)

    scaler = StandardScaler()

    X_train, y_train = process_group(
        data["train_normal"], 0,
        sr=args.sr,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        roll_percent=args.roll_percent,
        show_example=True
    )
    print(f"X_train shape: {X_train.shape}")

    if len(X_train) == 0:
        print("[WARN] No training data found.")
        return

    if args.scale_in_extract:
        X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"))

    X_test_n, y_test_n = process_group(
        data["test_normal"], 0,
        sr=args.sr,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        roll_percent=args.roll_percent
    )
    if len(X_test_n) > 0:
        if args.scale_in_extract:
            X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"))
    else:
        print("[WARN] No test_normal data")

    X_test_a, y_test_a = process_group(
        data["test_abnormal"], 1,
        sr=args.sr,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_fft=args.n_fft,
        roll_percent=args.roll_percent
    )
    if len(X_test_a) > 0:
        if args.scale_in_extract:
            X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"))
    else:
        print("[WARN] No test_abnormal data")

    print(f"Done handcrafted for: {args.machine} | feature={args.feature}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file JSON config")
    parser.add_argument("--machine", type=str, default="fan")
    parser.add_argument("--feature", type=str, default="handcrafted",
                        help="Tên folder feature (ví dụ: handcrafted_fs400_h160_fft512_r0.85)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--frame_size", type=int, default=400)
    parser.add_argument("--hop_size", type=int, default=160)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--roll_percent", type=float, default=0.85)
    parser.add_argument("--scale_in_extract", action="store_true",
                        help="Nếu bật, sẽ StandardScaler ngay ở bước extract (mặc định: tắt).")
    args = parser.parse_args()
    main(args)
