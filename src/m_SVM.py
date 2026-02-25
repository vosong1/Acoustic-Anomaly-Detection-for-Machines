import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA


# =========================
# Utils
# =========================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
    else:
        X = df.values
    return X


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pick_threshold(
    anomaly_train: np.ndarray,
    anomaly_test: np.ndarray,
    y_true: np.ndarray,
    mode: str,
    nu: float,
    percentile: float,
    force_threshold: float | None,
):
    """
    anomaly_score: càng lớn càng bất thường
    y_true: 0=normal, 1=abnormal
    """
    if force_threshold is not None:
        return float(force_threshold), "force_threshold"

    mode = (mode or "nu").lower().strip()

    if mode == "nu":
        # ~nu fraction train_normal bị coi là outlier
        thr = float(np.percentile(anomaly_train, 100.0 * (1.0 - float(nu))))
        return thr, f"nu ({nu})"

    if mode == "percentile":
        thr = float(np.percentile(anomaly_train, float(percentile)))
        return thr, f"percentile ({percentile})"

    if mode == "best_f1":
        # WARNING: dùng test để chọn threshold (debug)
        candidates = np.unique(np.quantile(anomaly_test, np.linspace(0.0, 1.0, 200)))
        best_f1 = -1.0
        best_thr = None
        for t in candidates:
            y_pred = (anomaly_test >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
        if best_thr is None:
            best_thr = float(np.percentile(anomaly_train, float(percentile)))
        return best_thr, "best_f1 (on test)"

    thr = float(np.percentile(anomaly_train, float(percentile)))
    return thr, f"percentile ({percentile})"


def auto_detect_features(machine: str):
    """
    Tự dò tất cả feature có đủ 3 file:
      train_normal.csv, test_normal.csv, test_abnormal.csv
    trong features/{feature}/{machine}/
    """
    feats = []
    root = "features"
    if not os.path.isdir(root):
        return feats

    for feat in sorted(os.listdir(root)):
        base = os.path.join(root, feat, machine)
        if not os.path.isdir(base):
            continue
        ok = all(
            os.path.exists(os.path.join(base, fn))
            for fn in ["train_normal.csv", "test_normal.csv", "test_abnormal.csv"]
        )
        if ok:
            feats.append(feat)
    return feats


# =========================
# Core runner for 1 feature
# =========================
def run_one_feature(args, machine: str, feature: str):
    base_dir = os.path.join("features", feature, machine)
    train_path = os.path.join(base_dir, "train_normal.csv")
    test_n_path = os.path.join(base_dir, "test_normal.csv")
    test_a_path = os.path.join(base_dir, "test_abnormal.csv")

    X_train = load_csv(train_path)
    X_test_n = load_csv(test_n_path)
    X_test_a = load_csv(test_a_path)

    original_dim = X_train.shape[1]

    # ---- Scaling ----
    scaler = None
    if args.scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_n = scaler.transform(X_test_n)
        X_test_a = scaler.transform(X_test_a)

    # ---- PCA ----
    pca = None
    if args.pca is not None:
        pca = PCA(n_components=float(args.pca))
        X_train = pca.fit_transform(X_train)
        X_test_n = pca.transform(X_test_n)
        X_test_a = pca.transform(X_test_a)

    # ---- Train ----
    model = OneClassSVM(kernel=args.kernel, gamma=args.gamma, nu=args.nu)
    model.fit(X_train)

    # ---- Scores ----
    # decision_function: + => normal, - => abnormal
    # anomaly_score = -decision_function => lớn => bất thường
    train_dec = model.decision_function(X_train)
    testn_dec = model.decision_function(X_test_n)
    testa_dec = model.decision_function(X_test_a)

    anomaly_train = -train_dec
    anomaly_n = -testn_dec
    anomaly_a = -testa_dec

    anomaly_test = np.concatenate([anomaly_n, anomaly_a])
    y_true = np.concatenate(
        [np.zeros(len(anomaly_n), dtype=int), np.ones(len(anomaly_a), dtype=int)]
    )

    auc = roc_auc_score(y_true, anomaly_test)

    thr, thr_note = pick_threshold(
        anomaly_train=anomaly_train,
        anomaly_test=anomaly_test,
        y_true=y_true,
        mode=args.threshold_mode,
        nu=float(args.nu),
        percentile=float(args.percentile),
        force_threshold=args.force_threshold,
    )
    y_pred = (anomaly_test >= thr).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]]

    # ---- Save per-feature artifacts (optional) ----
    if args.save_artifacts:
        result_dir = os.path.join("results", "OneClassSVM", machine, feature)
        model_dir = os.path.join("models", "OneClassSVM", machine, feature)
        ensure_dir(result_dir)
        ensure_dir(model_dir)

        with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Machine: {machine}\n")
            f.write(f"Feature: {feature}\n")
            f.write(f"AUC: {auc:.6f}\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1: {f1:.6f}\n")
            f.write(f"Threshold_mode: {thr_note}\n")
            f.write(f"Threshold: {thr:.10f}\n")
            f.write(f"Original Dim: {original_dim}\n")
            f.write(f"Dim After PCA: {X_train.shape[1]}\n")
            f.write(f"ConfusionMatrix_TN_FP_FN_TP: {cm[0,0]} {cm[0,1]} {cm[1,0]} {cm[1,1]}\n")

        with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)
        if scaler is not None:
            with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        if pca is not None:
            with open(os.path.join(model_dir, "pca.pkl"), "wb") as f:
                pickle.dump(pca, f)

    return {
        "machine": machine,
        "feature": feature,
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(thr),
        "threshold_mode": thr_note,
        "dim": int(X_train.shape[1]),
        "original_dim": int(original_dim),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--machine", type=str, default=None)

    # feature control
    parser.add_argument("--feature", type=str, default=None, help="Chạy 1 feature cụ thể")
    parser.add_argument("--features", type=str, default=None, help="Comma list: mfcc,logmel,combo_final")
    parser.add_argument("--all_features", action="store_true", help="Tự dò và chạy tất cả feature có sẵn")

    # SVM params
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--nu", type=float, default=0.15)

    # preprocess
    parser.add_argument("--scale", action="store_true", help="Bật StandardScaler trong SVM")
    parser.add_argument("--pca", type=float, default=0.95, help="tỉ lệ PCA (vd 0.95). Tắt PCA dùng --no_pca")
    parser.add_argument("--no_pca", action="store_true")

    # metrics threshold
    parser.add_argument("--threshold_mode", type=str, default="nu", help="nu | percentile | best_f1")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--force_threshold", type=float, default=None)

    # output
    parser.add_argument("--save_artifacts", action="store_true",
                        help="Lưu model/metrics cho từng feature vào results/ & models/")

    args = parser.parse_args()

    # ---- load config ----
    cfg = {}
    if args.config:
        cfg = read_json(args.config)
        if args.machine is None:
            args.machine = cfg.get("machine", None)

        svm_cfg = cfg.get("svm", {}) if isinstance(cfg.get("svm", {}), dict) else {}
        args.kernel = svm_cfg.get("kernel", args.kernel)
        args.gamma = svm_cfg.get("gamma", args.gamma)
        args.nu = svm_cfg.get("nu", args.nu)
        args.scale = svm_cfg.get("scale", args.scale)
        if not args.no_pca:
            args.pca = svm_cfg.get("pca", args.pca)

        metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {}
        args.threshold_mode = metrics_cfg.get("threshold_mode", args.threshold_mode)
        args.percentile = metrics_cfg.get("percentile", args.percentile)
        args.force_threshold = metrics_cfg.get("force_threshold", args.force_threshold)

        # config có thể set danh sách features
        # ví dụ: "features": ["mfcc","logmel","combo_final"]
        if args.features is None and isinstance(cfg.get("features", None), list):
            args.features = ",".join(cfg["features"])

        # config có thể set feature đơn
        if args.feature is None and isinstance(cfg.get("feature", None), str):
            args.feature = cfg.get("feature")

    if args.no_pca:
        args.pca = None

    if not args.machine:
        raise SystemExit("❌ Thiếu --machine (hoặc 'machine' trong config.json)")

    # ---- decide feature list ----
    feature_list = []

    if args.all_features:
        feature_list = auto_detect_features(args.machine)

    elif args.features:
        feature_list = [x.strip() for x in args.features.split(",") if x.strip()]

    elif args.feature:
        feature_list = [args.feature.strip()]

    else:
        # fallback: tự dò luôn cho tiện
        feature_list = auto_detect_features(args.machine)

    if not feature_list:
        raise SystemExit("❌ Không tìm thấy feature nào để chạy. Hãy kiểm tra thư mục features/")

    print(f"\n[RUN] machine={args.machine}")
    print(f"[RUN] features={feature_list}")
    print(f"[RUN] svm: kernel={args.kernel}, gamma={args.gamma}, nu={args.nu}, scale={args.scale}, pca={args.pca}")
    print(f"[RUN] threshold: mode={args.threshold_mode}, percentile={args.percentile}, force={args.force_threshold}")
    print("")

    # ---- run loop ----
    rows = []
    for feat in feature_list:
        try:
            r = run_one_feature(args, args.machine, feat)
            rows.append(r)
            print(f"[OK] {feat:12s} | AUC={r['auc']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}  dim={r['dim']}")
        except FileNotFoundError as e:
            print(f"[SKIP] {feat}: thiếu file {e}")
        except Exception as e:
            print(f"[FAIL] {feat}: {e}")

    if not rows:
        raise SystemExit("❌ Không feature nào chạy thành công.")

    # ---- summary table (sorted by AUC) ----
    df = pd.DataFrame(rows).sort_values(["auc", "f1"], ascending=False)

    print("\n========== SUMMARY (sorted by AUC) ==========")
    print(df[["feature", "auc", "precision", "recall", "f1", "dim", "threshold_mode", "threshold"]].to_string(index=False))

    # optional: save summary csv (khi bạn muốn)
    if args.save_artifacts:
        out_dir = os.path.join("results", "OneClassSVM", args.machine)
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "summary_all_features.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[DONE] Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
