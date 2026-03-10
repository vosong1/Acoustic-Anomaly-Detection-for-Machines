import argparse
from sklearn.svm import OneClassSVM

from model_utils import run_model_pipeline


def build_svm(cfg):
    return OneClassSVM(
        kernel=cfg.get("kernel", "rbf"),
        gamma=cfg.get("gamma", "scale"),
        nu=cfg.get("nu", 0.05),
    )


def fit_svm(model, X):
    model.fit(X)


def score_svm(model, X):
    """
    Higher = more normal
    """
    return model.decision_function(X).reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True, choices=["mfcc", "logmel", "stft", "chroma"])
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    run_model_pipeline(
        model_name="oneclass_svm",
        feature_name=args.feature,
        build_model_fn=build_svm,
        fit_model_fn=fit_svm,
        score_fn=score_svm,
        threshold_percentile_key="threshold_percentile",
        abnormal_when_lower=True,
        config_path=args.config,
    )