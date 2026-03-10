import argparse
from sklearn.ensemble import IsolationForest

from model_utils import run_model_pipeline


def build_iso(cfg):
    return IsolationForest(
        n_estimators=cfg.get("n_estimators", 100),
        contamination=cfg.get("contamination", "auto"),
        random_state=cfg.get("random_state", 42),
        max_samples=cfg.get("max_samples", "auto"),
    )


def fit_iso(model, X):
    model.fit(X)


def score_iso(model, X):
    """
    Higher = more normal
    """
    return model.decision_function(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True, choices=["mfcc", "logmel", "stft", "chroma"])
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    run_model_pipeline(
        model_name="isolation_forest",
        feature_name=args.feature,
        build_model_fn=build_iso,
        fit_model_fn=fit_iso,
        score_fn=score_iso,
        threshold_percentile_key="threshold_percentile",
        abnormal_when_lower=True,
        config_path=args.config,
    )