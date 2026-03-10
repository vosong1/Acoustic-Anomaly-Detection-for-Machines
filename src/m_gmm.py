import argparse
import numpy as np
from sklearn.mixture import GaussianMixture

from model_utils import run_model_pipeline


def build_gmm(cfg):
    return GaussianMixture(
        n_components=cfg.get("n_components", 2),
        covariance_type=cfg.get("covariance_type", "full"),
        random_state=cfg.get("random_state", 42),
        max_iter=cfg.get("max_iter", 200),
    )


def fit_gmm(model, X):
    model.fit(X)


def score_gmm(model, X):
    """
    Higher = more normal
    """
    return model.score_samples(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True, choices=["mfcc", "logmel", "stft", "chroma"])
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    run_model_pipeline(
        model_name="gmm",
        feature_name=args.feature,
        build_model_fn=build_gmm,
        fit_model_fn=fit_gmm,
        score_fn=score_gmm,
        threshold_percentile_key="threshold_percentile",
        abnormal_when_lower=True,
        config_path=args.config,
    )