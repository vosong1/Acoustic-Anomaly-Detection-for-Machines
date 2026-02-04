import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file: {path}")

    df = pd.read_csv(path)

    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
        y = df["label"].values
    else:
        X = df.values
        y = None

    return X, y

def apply_config(args):
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args.machine = cfg.get("machine", args.machine)

    svm_cfg = cfg.get("svm", {}) if isinstance(cfg.get("svm", {}), dict) else {}
    args.nu = svm_cfg.get("nu", args.nu)
    args.gamma = svm_cfg.get("gamma", args.gamma)
    args.kernel = svm_cfg.get("kernel", args.kernel)
    args.pca = svm_cfg.get("pca", args.pca)
    args.scale = svm_cfg.get("scale", args.scale)

    # Cho phép config chọn feature mặc định
    if args.feature is None:
        args.feature = cfg.get("feature", None)

    return args

def main(args):
    args = apply_config(args)
    if not args.feature:
        raise ValueError("Bạn phải truyền --feature (hoặc set 'feature' trong config.json)")

    base_dir = os.path.join("features", args.feature, args.machine)
    train_path = os.path.join(base_dir, "train_normal.csv")
    test_n_path = os.path.join(base_dir, "test_normal.csv")
    test_a_path = os.path.join(base_dir, "test_abnormal.csv")

    result_dir = os.path.join("results", "OneClassSVM", args.machine, args.feature)
    model_dir = os.path.join("models", "OneClassSVM", args.machine, args.feature)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[START] Machine: {args.machine} | Feature: {args.feature}")

    try:
        X_train, _ = load_csv(train_path)
        X_test_n, _ = load_csv(test_n_path)
        X_test_a, _ = load_csv(test_a_path)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"[INFO] Original Train shape: {X_train.shape}")
    original_dim = X_train.shape[1]

    scaler = None
    if args.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_n = scaler.transform(X_test_n)
        X_test_a = scaler.transform(X_test_a)

    pca = None
    if args.pca is not None:
        pca = PCA(n_components=float(args.pca))
        X_train = pca.fit_transform(X_train)
        X_test_n = pca.transform(X_test_n)
        X_test_a = pca.transform(X_test_a)
        print(f"[INFO] PCA: giảm số chiều từ {original_dim} xuống {X_train.shape[1]}")
    else:
        print("[INFO] PCA: tắt")

    model = OneClassSVM(kernel=args.kernel, gamma=args.gamma, nu=args.nu)

    print("[INFO] Training One-Class SVM...")
    model.fit(X_train)

    score_n = model.decision_function(X_test_n)
    score_a = model.decision_function(X_test_a)

    scores = np.concatenate([score_n, score_a])
    labels = np.concatenate([np.ones(len(score_n)), np.zeros(len(score_a))])

    auc = roc_auc_score(labels, scores)
    print(f"[RESULT] AUC Score: {auc:.4f}")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Model: One-Class SVM\n")
        f.write(f"Machine: {args.machine}\n")
        f.write(f"Feature: {args.feature}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Original Dim: {original_dim}\n")
        f.write(f"Dim: {X_train.shape[1]}\n")

    with open(os.path.join(result_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(f"kernel: {args.kernel}\n")
        f.write(f"gamma: {args.gamma}\n")
        f.write(f"nu: {args.nu}\n")
        f.write(f"scale: {args.scale}\n")
        f.write(f"pca: {args.pca}\n")

    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    if scaler is not None:
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
    if pca is not None:
        with open(os.path.join(model_dir, "pca.pkl"), "wb") as f:
            pickle.dump(pca, f)

    print(f"[DONE] Saved to: {result_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file JSON config")
    parser.add_argument("--machine", type=str, default="fan", help="Tên loại máy (ví dụ: fan, pump)")
    parser.add_argument("--feature", type=str, default=None, help="Tên bộ feature (ví dụ: mfcc_n11_fs1024_h512_fft1024)")
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--gamma", default="scale", help="scale/auto hoặc số (vd 0.01)")
    parser.add_argument("--nu", type=float, default=0.15)
    parser.add_argument("--pca", type=float, default=0.95, help="tỉ lệ PCA (vd 0.95). Muốn tắt: dùng --no_pca", nargs='?')
    parser.add_argument("--no_pca", action="store_true", help="Tắt PCA")
    parser.add_argument("--scale", action="store_true", help="Bật StandardScaler trong SVM (khuyến nghị bật nếu extract không scale)")

    args = parser.parse_args()
    if args.no_pca:
        args.pca = None
    main(args)
