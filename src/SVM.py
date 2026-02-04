import argparse
import os
import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score


def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y



def main(machine, feature):
    base_dir = f"features/{feature}/{machine}"

    train_path = os.path.join(base_dir, "train_normal.csv")
    test_n_path = os.path.join(base_dir, "test_normal.csv")
    test_a_path = os.path.join(base_dir, "test_abnormal.csv")

    out_dir = f"results/OneClassSVM/{machine}/{feature}"
    os.makedirs(out_dir, exist_ok=True)

    X_train, _ = load_csv(train_path)
    X_test_n, _ = load_csv(test_n_path)
    X_test_a, _ = load_csv(test_a_path)

    print(f"[INFO] Train samples: {X_train.shape}")
    print(f"[INFO] Test normal: {X_test_n.shape}")
    print(f"[INFO] Test abnormal: {X_test_a.shape}")

    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )

    model.fit(X_train)

    score_n = model.decision_function(X_test_n)
    score_a = model.decision_function(X_test_a)

    scores = np.concatenate([score_n, score_a])
    labels = np.concatenate([
        np.ones(len(score_n)),   
        np.zeros(len(score_a))  
    ])

    auc = roc_auc_score(labels, scores)

    print(f"[RESULT] AUC = {auc:.4f}")

    df_score = pd.DataFrame({
        "score": scores,
        "label": labels
    })
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("Model: One-Class SVM\n")
        f.write(f"Machine: {machine}\n")
        f.write(f"Feature: {feature}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test normal: {len(X_test_n)}\n")
        f.write(f"Test abnormal: {len(X_test_a)}\n")

    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        f.write("kernel = rbf\n")
        f.write("gamma = scale\n")
        f.write("nu = 0.05\n")

    print(f"[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", required=True)
    parser.add_argument("--feature", required=True)

    args = parser.parse_args()
    main(args.machine, args.feature)
