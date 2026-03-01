# src/m_gmm.py
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score


# -----------------------
# Utils
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_csv_X(csv_path: Path) -> np.ndarray:
    """Load CSV -> numeric feature matrix. Drops non-feature columns."""
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)

    # drop non-feature columns
    df = df.drop(columns=["label", "path"], errors="ignore")

    # ensure numeric
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(X).any():
        bad = int(np.isnan(X).any(axis=1).sum())
        raise ValueError(f"{csv_path}: có {bad} dòng bị NaN (feature lỗi hoặc cột không phải số).")
    return X


def detect_features(feature_root: Path, machine: str):
    """Detect feature folders that contain required CSVs for the given machine."""
    feats = []
    if not feature_root.exists():
        return feats

    for feature_dir in sorted([p for p in feature_root.iterdir() if p.is_dir()]):
        base = feature_dir / machine
        if not base.exists():
            continue

        ok = (base / "train_normal.csv").exists() and (base / "test_normal.csv").exists() and (base / "test_abnormal.csv").exists()
        if ok:
            feats.append(feature_dir.name)

    return feats


def preprocess_fit_transform(X_train, X_test_n, X_test_a, scale=True, pca=None):
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_n = scaler.transform(X_test_n)
        X_test_a = scaler.transform(X_test_a)

    pca_model = None
    if pca is not None:
        pca_model = PCA(n_components=pca)
        X_train = pca_model.fit_transform(X_train)
        X_test_n = pca_model.transform(X_test_n)
        X_test_a = pca_model.transform(X_test_a)

    return X_train, X_test_n, X_test_a, scaler, pca_model


def eval_scores(anomaly_scores_n, anomaly_scores_a, threshold):
    y_true = np.concatenate([np.zeros(len(anomaly_scores_n), dtype=int),
                             np.ones(len(anomaly_scores_a), dtype=int)])
    scores = np.concatenate([anomaly_scores_n, anomaly_scores_a])
    y_pred = (scores >= threshold).astype(int)

    auc = roc_auc_score(y_true, scores)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auc": float(auc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "n_test_normal": int(len(anomaly_scores_n)),
        "n_test_abnormal": int(len(anomaly_scores_a)),
    }


# -----------------------
# Config
# -----------------------
def apply_config_to_args(args):
    """Map config.json -> argparse args. CLI vẫn override nếu bạn truyền sau đó."""
    if not args.config:
        return args

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # machine
    if cfg.get("machine"):
        args.machine = cfg["machine"]

    # feature_root: ưu tiên out_root (extractors)
    if cfg.get("out_root"):
        args.feature_root = cfg["out_root"]
    elif cfg.get("feature_root"):
        args.feature_root = cfg["feature_root"]

    # gmm params
    gmm_cfg = cfg.get("gmm", {})
    if isinstance(gmm_cfg, dict):
        if "n_components" in gmm_cfg:
            args.n_components = int(gmm_cfg["n_components"])
        if "covariance_type" in gmm_cfg:
            args.covariance_type = gmm_cfg["covariance_type"]
        if "reg_covar" in gmm_cfg:
            args.reg_covar = float(gmm_cfg["reg_covar"])
        if "seed" in gmm_cfg:
            args.seed = int(gmm_cfg["seed"])
        if "scale" in gmm_cfg:
            args.scale = bool(gmm_cfg["scale"])
        if "pca" in gmm_cfg:
            args.pca = gmm_cfg["pca"]

    # threshold mode (tùy bạn mở rộng)
    metrics_cfg = cfg.get("metrics", {})
    if isinstance(metrics_cfg, dict) and "threshold_quantile" in metrics_cfg:
        args.threshold_quantile = float(metrics_cfg["threshold_quantile"])

    return args


# -----------------------
# Core
# -----------------------
def run_one_feature(args, feature: str):
    base = Path(args.feature_root) / feature / args.machine
    train_csv = base / "train_normal.csv"
    test_n_csv = base / "test_normal.csv"
    test_a_csv = base / "test_abnormal.csv"

    X_train = load_csv_X(train_csv)
    X_test_n = load_csv_X(test_n_csv)
    X_test_a = load_csv_X(test_a_csv)

    pca_param = None if args.no_pca else args.pca
    X_train, X_test_n, X_test_a, scaler, pca_model = preprocess_fit_transform(
        X_train, X_test_n, X_test_a,
        scale=args.scale,
        pca=pca_param
    )

    model = GaussianMixture(
        n_components=int(args.n_components),
        covariance_type=args.covariance_type,
        reg_covar=float(args.reg_covar),
        random_state=int(args.seed),
    )
    model.fit(X_train)

    # score_samples: log-likelihood lớn => normal, nhỏ => abnormal
    ll_train = model.score_samples(X_train)
    ll_n = model.score_samples(X_test_n)
    ll_a = model.score_samples(X_test_a)

    # anomaly score lớn => abnormal
    anom_train = -ll_train
    anom_n = -ll_n
    anom_a = -ll_a

    threshold = float(np.quantile(anom_train, float(args.threshold_quantile)))
    metrics = eval_scores(anom_n, anom_a, threshold)

    metrics.update({
        "model": "gmm",
        "machine": args.machine,
        "feature": feature,
        "feature_root": str(args.feature_root),
        "n_components": int(args.n_components),
        "covariance_type": args.covariance_type,
        "reg_covar": float(args.reg_covar),
        "seed": int(args.seed),
        "scale": bool(args.scale),
        "pca": None if args.no_pca else args.pca,
        "train_dim": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
    })

    if args.save_artifacts:
        out_res = Path("results") / "gmm" / args.machine / feature
        out_mod = Path("models") / "gmm" / args.machine / feature
        ensure_dir(out_res)
        ensure_dir(out_mod)

        (out_res / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        with open(out_mod / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        if scaler is not None:
            with open(out_mod / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
        if pca_model is not None:
            with open(out_mod / "pca.pkl", "wb") as f:
                pickle.dump(pca_model, f)

    return metrics


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default=None)

    ap.add_argument("--machine", default="fan")
    ap.add_argument("--feature", default=None, help="vd: mfcc/logmel/stft/hc/chroma ...")
    ap.add_argument("--all_features", action="store_true")

    ap.add_argument("--feature_root", default="extract/features",
                    help='vd: "extract/features" hoặc "features"')

    # preprocess
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--pca", type=float, default=0.95)
    ap.add_argument("--no_pca", action="store_true")

    # gmm params
    ap.add_argument("--n_components", type=int, default=4)
    ap.add_argument("--covariance_type", default="full", choices=["full", "tied", "diag", "spherical"])
    ap.add_argument("--reg_covar", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)

    # threshold
    ap.add_argument("--threshold_quantile", type=float, default=0.95)

    ap.add_argument("--save_artifacts", action="store_true")

    args = ap.parse_args()
    args = apply_config_to_args(args)

    feature_root = Path(args.feature_root)

    if args.all_features:
        features = detect_features(feature_root, args.machine)
        if not features:
            raise RuntimeError(f"Không tìm thấy feature nào hợp lệ trong {feature_root} (machine={args.machine})")
    else:
        if not args.feature:
            raise RuntimeError("Bạn phải truyền --feature hoặc dùng --all_features")
        features = [args.feature]

    results = []
    for feat in features:
        m = run_one_feature(args, feat)
        results.append(m)

    df = pd.DataFrame(results)

    cols = [c for c in ["feature", "auc", "f1", "precision", "recall", "threshold", "train_size", "train_dim"] if c in df.columns]
    df_show = df[cols].sort_values("auc", ascending=False)

    print("\n===== SUMMARY (GMM) =====")
    print(df_show.to_string(index=False))

    out_summary_dir = Path("results") / "gmm" / args.machine
    ensure_dir(out_summary_dir)
    df.to_csv(out_summary_dir / "summary.csv", index=False)
    print(f"\nSaved summary to: {out_summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()