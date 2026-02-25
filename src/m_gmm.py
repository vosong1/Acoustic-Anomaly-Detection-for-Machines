import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# Utils
# =========================================================
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_csv_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "label" in df.columns:
        return df.drop(columns=["label"]).values
    return df.values


def auto_detect_features(machine: str):
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


# =========================================================
# Threshold for anomaly_score
# anomaly_score: càng lớn càng bất thường
# y_true: 0=normal, 1=abnormal
# =========================================================
def pick_threshold(
    anomaly_train: np.ndarray,
    anomaly_test: np.ndarray,
    y_true: np.ndarray,
    mode: str,
    nu: float,
    percentile: float,
    force_threshold: float | None,
):
    if force_threshold is not None:
        return float(force_threshold), "force_threshold"

    mode = (mode or "nu_like").lower().strip()

    if mode == "nu_like":
        # giống SVM: cho phép ~nu % train_normal bị coi là outlier
        thr = float(np.percentile(anomaly_train, 100.0 * (1.0 - float(nu))))
        return thr, f"nu_like ({nu})"

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


# =========================================================
# GMM scoring
# log-likelihood: cao => giống normal
# anomaly_score = -log_likelihood: cao => bất thường
# =========================================================
def run_gmm_once(X_train, X_test_n, X_test_a, n_components, cov_type, max_iter, seed):
    gmm = GaussianMixture(
        n_components=int(n_components),
        covariance_type=str(cov_type),
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    gmm.fit(X_train)

    ll_train = gmm.score_samples(X_train)
    ll_n = gmm.score_samples(X_test_n)
    ll_a = gmm.score_samples(X_test_a)

    anomaly_train = -ll_train
    anomaly_n = -ll_n
    anomaly_a = -ll_a

    anomaly_test = np.concatenate([anomaly_n, anomaly_a])
    y_true = np.concatenate(
        [np.zeros(len(anomaly_n), dtype=int), np.ones(len(anomaly_a), dtype=int)]
    )

    auc = roc_auc_score(y_true, anomaly_test)
    return gmm, anomaly_train, anomaly_test, y_true, auc


# =========================================================
# Run one feature with grid search
# =========================================================
def run_one_feature(args, machine: str, feature: str):
    base_dir = os.path.join("features", feature, machine)
    train_path = os.path.join(base_dir, "train_normal.csv")
    test_n_path = os.path.join(base_dir, "test_normal.csv")
    test_a_path = os.path.join(base_dir, "test_abnormal.csv")

    X_train = load_csv_any(train_path)
    X_test_n = load_csv_any(test_n_path)
    X_test_a = load_csv_any(test_a_path)

    original_dim = int(X_train.shape[1])

    # ---- preprocess
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

    dim_after = int(X_train.shape[1])

    # ---- grid search
    best = None
    grid_rows = []

    for n_comp in args.n_components_list:
        for cov in args.cov_types:
            try:
                model, anomaly_train, anomaly_test, y_true, auc = run_gmm_once(
                    X_train, X_test_n, X_test_a,
                    n_components=n_comp,
                    cov_type=cov,
                    max_iter=args.max_iter,
                    seed=args.seed,
                )

                thr, thr_note = pick_threshold(
                    anomaly_train=anomaly_train,
                    anomaly_test=anomaly_test,
                    y_true=y_true,
                    mode=args.threshold_mode,
                    nu=args.nu,
                    percentile=args.percentile,
                    force_threshold=args.force_threshold,
                )

                y_pred = (anomaly_test >= thr).astype(int)

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)

                row = {
                    "machine": machine,
                    "feature": feature,
                    "n_components": int(n_comp),
                    "covariance_type": str(cov),
                    "auc": float(auc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "threshold": float(thr),
                    "threshold_mode": thr_note,
                    "dim": dim_after,
                    "original_dim": original_dim,
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1]),
                }
                grid_rows.append(row)

                if (best is None) or (row["auc"] > best["auc"]) or (row["auc"] == best["auc"] and row["f1"] > best["f1"]):
                    best = dict(row)
                    best["_model"] = model
                    best["_scaler"] = scaler
                    best["_pca"] = pca

            except Exception as e:
                # vẫn ghi lại để debug grid nếu cần
                grid_rows.append({
                    "machine": machine,
                    "feature": feature,
                    "n_components": int(n_comp),
                    "covariance_type": str(cov),
                    "auc": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "threshold": np.nan,
                    "threshold_mode": "ERROR",
                    "dim": dim_after,
                    "original_dim": original_dim,
                    "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
                })

    if best is None:
        raise RuntimeError(f"Không tìm được cấu hình GMM hợp lệ cho feature={feature}")

    # save artifacts if needed
    if args.save_artifacts:
        result_dir = os.path.join("results", "GMM", machine, feature)
        model_dir = os.path.join("models", "GMM", machine, feature)
        ensure_dir(result_dir)
        ensure_dir(model_dir)

        pd.DataFrame(grid_rows).to_csv(os.path.join(result_dir, "grid_results.csv"), index=False)

        with open(os.path.join(result_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Machine: {machine}\n")
            f.write(f"Feature: {feature}\n")
            f.write(f"BEST AUC: {best['auc']:.6f}\n")
            f.write(f"Precision: {best['precision']:.6f}\n")
            f.write(f"Recall: {best['recall']:.6f}\n")
            f.write(f"F1: {best['f1']:.6f}\n")
            f.write(f"Params: n_components={best['n_components']}, cov={best['covariance_type']}\n")
            f.write(f"Threshold_mode: {best['threshold_mode']}\n")
            f.write(f"Threshold: {best['threshold']:.10f}\n")
            f.write(f"Original Dim: {best['original_dim']}\n")
            f.write(f"Dim After PCA: {best['dim']}\n")
            f.write(f"ConfusionMatrix_TN_FP_FN_TP: {best['tn']} {best['fp']} {best['fn']} {best['tp']}\n")

        with open(os.path.join(model_dir, "gmm.pkl"), "wb") as f:
            pickle.dump(best["_model"], f)
        if best["_scaler"] is not None:
            with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
                pickle.dump(best["_scaler"], f)
        if best["_pca"] is not None:
            with open(os.path.join(model_dir, "pca.pkl"), "wb") as f:
                pickle.dump(best["_pca"], f)

    # remove heavy objects
    best.pop("_model", None)
    best.pop("_scaler", None)
    best.pop("_pca", None)

    return best, grid_rows


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Train/Tune GMM for anomaly detection (multi-feature like SVM)")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--machine", type=str, default=None)

    # feature control
    parser.add_argument("--feature", type=str, default=None)
    parser.add_argument("--features", type=str, default=None, help="Comma list: mfcc,logmel,combo_final")
    parser.add_argument("--all_features", action="store_true", help="Auto detect and run all features")

    # preprocess
    parser.add_argument("--scale", action="store_true", help="StandardScaler before GMM")
    parser.add_argument("--pca", type=float, default=None, help="PCA ratio (vd 0.95). Tắt: None")
    parser.add_argument("--no_pca", action="store_true")

    # GMM grid
    parser.add_argument("--n_components", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--cov_types", type=str, default="full,diag,tied")
    parser.add_argument("--max_iter", type=int, default=100)

    # threshold/metrics
    parser.add_argument("--threshold_mode", type=str, default="nu_like",
                        help="nu_like | percentile | best_f1")
    parser.add_argument("--nu", type=float, default=0.15, help="dùng cho threshold_mode=nu_like")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--force_threshold", type=float, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_artifacts", action="store_true")

    args = parser.parse_args()

    # ---------- load config ----------
    if args.config:
        cfg = read_json(args.config)

        if args.machine is None:
            args.machine = cfg.get("machine", None)

        # preprocess section (optional)
        prep_cfg = cfg.get("preprocess", {}) if isinstance(cfg.get("preprocess", {}), dict) else {}
        args.scale = prep_cfg.get("scale", args.scale)
        if not args.no_pca:
            args.pca = prep_cfg.get("pca", args.pca)

        # gmm section (optional)
        gmm_cfg = cfg.get("gmm", {}) if isinstance(cfg.get("gmm", {}), dict) else {}
        if "n_components" in gmm_cfg:
            if isinstance(gmm_cfg["n_components"], list):
                args.n_components = ",".join(str(x) for x in gmm_cfg["n_components"])
            else:
                args.n_components = str(gmm_cfg["n_components"])
        if "cov_types" in gmm_cfg:
            if isinstance(gmm_cfg["cov_types"], list):
                args.cov_types = ",".join(gmm_cfg["cov_types"])
            else:
                args.cov_types = str(gmm_cfg["cov_types"])
        args.max_iter = gmm_cfg.get("max_iter", args.max_iter)

        # metrics section (optional)
        metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), dict) else {}
        args.threshold_mode = metrics_cfg.get("threshold_mode", args.threshold_mode)
        args.percentile = metrics_cfg.get("percentile", args.percentile)
        args.force_threshold = metrics_cfg.get("force_threshold", args.force_threshold)
        # allow nu in config
        args.nu = metrics_cfg.get("nu", args.nu)

        # features list from config (optional)
        if args.features is None and isinstance(cfg.get("features", None), list):
            args.features = ",".join(cfg["features"])
        if args.feature is None and isinstance(cfg.get("feature", None), str):
            args.feature = cfg.get("feature")

    if args.no_pca:
        args.pca = None

    if not args.machine:
        raise SystemExit("❌ Thiếu --machine (hoặc 'machine' trong config.json)")

    # parse grids
    args.n_components_list = [int(x.strip()) for x in args.n_components.split(",") if x.strip()]
    args.cov_types = [x.strip() for x in args.cov_types.split(",") if x.strip()]

    # decide features
    feature_list = []
    if args.all_features:
        feature_list = auto_detect_features(args.machine)
    elif args.features:
        feature_list = [x.strip() for x in args.features.split(",") if x.strip()]
    elif args.feature:
        feature_list = [args.feature.strip()]
    else:
        feature_list = auto_detect_features(args.machine)

    if not feature_list:
        raise SystemExit("❌ Không tìm thấy feature nào để chạy. Hãy kiểm tra thư mục features/")

    print(f"\n[RUN] machine={args.machine}")
    print(f"[RUN] features={feature_list}")
    print(f"[RUN] preprocess: scale={args.scale}, pca={args.pca}")
    print(f"[RUN] GMM grid: n_components={args.n_components_list}, cov_types={args.cov_types}, max_iter={args.max_iter}")
    print(f"[RUN] threshold: mode={args.threshold_mode}, nu={args.nu}, percentile={args.percentile}, force={args.force_threshold}\n")

    best_rows = []
    for feat in feature_list:
        try:
            best, _ = run_one_feature(args, args.machine, feat)
            best_rows.append(best)
            print(
                f"[OK] {feat:12s} | "
                f"AUC={best['auc']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}  "
                f"n={best['n_components']:<3d} cov={best['covariance_type']:<4s} dim={best['dim']}"
            )
        except FileNotFoundError as e:
            print(f"[SKIP] {feat}: thiếu file {e}")
        except Exception as e:
            print(f"[FAIL] {feat}: {e}")

    if not best_rows:
        raise SystemExit("❌ Không feature nào chạy thành công.")

    df = pd.DataFrame(best_rows).sort_values(["auc", "f1"], ascending=False)

    print("\n========== BEST SUMMARY (sorted by AUC) ==========")
    print(df[["feature", "auc", "precision", "recall", "f1", "n_components", "covariance_type", "dim", "threshold_mode", "threshold"]].to_string(index=False))

    if args.save_artifacts:
        out_dir = os.path.join("results", "GMM", args.machine)
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "summary_all_features_best.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[DONE] Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
