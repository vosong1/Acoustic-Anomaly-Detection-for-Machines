from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from src.config_utils import load_config, ensure_dir


def get_feature_paths(cfg: dict, feature_name: str):
    dataset_cfg = cfg["dataset"]
    feature_root = Path(dataset_cfg["feature_dir"]) / feature_name
    machines = dataset_cfg["machines"]
    snr_levels = dataset_cfg["snr_levels"]

    items = []

    for machine in machines:
        for snr in snr_levels:
            db_dir = feature_root / machine / snr
            items.append(
                {
                    "machine": machine,
                    "snr": snr,
                    "mode": "db",
                    "dir": db_dir,
                }
            )

        all_db_dir = feature_root / machine / "all_db"
        items.append(
            {
                "machine": machine,
                "snr": "all_db",
                "mode": "all_db",
                "dir": all_db_dir,
            }
        )

    return items


def load_feature_csv(csv_path: Path):
    if not csv_path.exists():
        return None, None

    df = pd.read_csv(csv_path)
    if df.empty:
        return df, None

    meta_cols = [c for c in ["file", "machine", "snr", "split"] if c in df.columns]
    feature_cols = [c for c in df.columns if c.startswith("f")]

    if not feature_cols:
        raise ValueError(f"No feature columns found in {csv_path}")

    X = df[feature_cols].values.astype(float)
    return df, X


def make_output_dirs(cfg: dict, feature_name: str, model_name: str, machine: str, snr: str):
    model_root = Path(cfg["model"]["model_dir"]) / model_name / feature_name / machine / snr
    result_root = Path(cfg["model"]["result_dir"]) / model_name / feature_name / machine / snr

    ensure_dir(model_root)
    ensure_dir(result_root)

    return model_root, result_root


def save_predictions(df: pd.DataFrame, scores, preds, out_file: Path):
    out_df = df.copy()
    out_df["anomaly_score"] = scores
    out_df["pred"] = preds
    out_df.to_csv(out_file, index=False)


def save_summary(summary: dict, out_file: Path):
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_model(model, out_file: Path):
    joblib.dump(model, out_file)


def percentile_threshold(train_scores: np.ndarray, percentile: float):
    return float(np.percentile(train_scores, percentile))


def evaluate_from_scores(scores: np.ndarray, threshold: float, abnormal_when_lower: bool = True):
    """
    Return binary prediction:
    1 = normal
    -1 = abnormal
    """
    if abnormal_when_lower:
        preds = np.where(scores < threshold, -1, 1)
    else:
        preds = np.where(scores > threshold, -1, 1)
    return preds


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true: 1 normal, -1 abnormal
    y_pred: 1 normal, -1 abnormal
    """
    tp = int(np.sum((y_true == -1) & (y_pred == -1)))
    tn = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 1) & (y_pred == -1)))
    fn = int(np.sum((y_true == -1) & (y_pred == 1)))

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "f1_abnormal": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_model_pipeline(
    model_name: str,
    feature_name: str,
    build_model_fn,
    fit_model_fn,
    score_fn,
    threshold_percentile_key: str,
    abnormal_when_lower: bool = True,
    config_path: str = "config.json",
):
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    model_params = cfg[model_name]
    feature_items = get_feature_paths(cfg, feature_name)

    for item in feature_items:
        feature_dir = item["dir"]
        machine = item["machine"]
        snr = item["snr"]

        train_csv = feature_dir / "normal.csv"
        abnormal_csv = feature_dir / "abnormal.csv"

        if not train_csv.exists():
            print(f"[SKIP] Missing train file: {train_csv}")
            continue

        train_df, X_train = load_feature_csv(train_csv)
        if X_train is None or len(X_train) == 0:
            print(f"[SKIP] Empty train file: {train_csv}")
            continue

        abnormal_df, X_abnormal = load_feature_csv(abnormal_csv)
        if abnormal_df is None or X_abnormal is None:
            print(f"[WARN] Missing abnormal file: {abnormal_csv}")
            abnormal_df = None
            X_abnormal = None

        model = build_model_fn(model_params)
        fit_model_fn(model, X_train)

        train_scores = score_fn(model, X_train)
        threshold = percentile_threshold(
            train_scores,
            model_params.get(threshold_percentile_key, 5.0),
        )

        train_preds = evaluate_from_scores(
            train_scores,
            threshold=threshold,
            abnormal_when_lower=abnormal_when_lower,
        )

        model_dir, result_dir = make_output_dirs(
            cfg=cfg,
            feature_name=feature_name,
            model_name=model_name,
            machine=machine,
            snr=snr,
        )

        save_model(model, model_dir / "model.pkl")
        save_predictions(train_df, train_scores, train_preds, result_dir / "normal_pred.csv")

        summary = {
            "model": model_name,
            "feature": feature_name,
            "machine": machine,
            "snr": snr,
            "threshold": threshold,
            "train_size": int(len(X_train)),
            "abnormal_test_size": int(len(X_abnormal)) if X_abnormal is not None else 0,
            "train_score_mean": float(np.mean(train_scores)),
            "train_score_std": float(np.std(train_scores)),
        }

        if X_abnormal is not None and len(X_abnormal) > 0:
            abnormal_scores = score_fn(model, X_abnormal)
            abnormal_preds = evaluate_from_scores(
                abnormal_scores,
                threshold=threshold,
                abnormal_when_lower=abnormal_when_lower,
            )

            save_predictions(
                abnormal_df,
                abnormal_scores,
                abnormal_preds,
                result_dir / "abnormal_pred.csv",
            )

            y_true = np.concatenate([
                np.ones(len(train_preds), dtype=int),
                -np.ones(len(abnormal_preds), dtype=int),
            ])
            y_pred = np.concatenate([train_preds, abnormal_preds])

            summary["abnormal_score_mean"] = float(np.mean(abnormal_scores))
            summary["abnormal_score_std"] = float(np.std(abnormal_scores))
            summary["metrics"] = binary_metrics(y_true, y_pred)

        save_summary(summary, result_dir / "summary.json")
        print(f"[DONE] {model_name} | {feature_name} | {machine} | {snr}")