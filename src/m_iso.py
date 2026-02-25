import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MACHINE_TYPE = "fan"
DATA_DIR = os.path.join(parent_dir, "features/stft_pca", MACHINE_TYPE) 
RESULT_DIR = os.path.join(parent_dir, "results/isolation_forest", MACHINE_TYPE)
os.makedirs(RESULT_DIR, exist_ok=True)

def _to_X(df: pd.DataFrame):
    return df.drop(columns=["label"], errors="ignore").values

def load_data():
    try:
        df_train = pd.read_csv(os.path.join(DATA_DIR, "train_normal.csv"))
        df_test_n = pd.read_csv(os.path.join(DATA_DIR, "test_normal.csv"))
        df_test_a = pd.read_csv(os.path.join(DATA_DIR, "test_abnormal.csv"))

        X_train = _to_X(df_train)

        X_test_n = _to_X(df_test_n)
        y_test_n = np.zeros(len(X_test_n))
        X_test_a = _to_X(df_test_a)
        y_test_a = np.ones(len(X_test_a))

        X_test = np.concatenate([X_test_n, X_test_a])
        y_test = np.concatenate([y_test_n, y_test_a])
        return X_train, X_test, y_test
    except Exception as e:
        print("Lỗi load_data:", e)
        print("Hãy kiểm tra DATA_DIR và đảm bảo đã có train_normal/test_normal/test_abnormal CSV.")
        sys.exit(1)
def train_and_eval():
    X_train, X_test, y_test = load_data()
    print(f"Training Isolation Forest for {MACHINE_TYPE}...")

    model = IsolationForest(n_estimators=100, 
                            contamination=0.01, 
                            random_state=42, 
                            n_jobs=-1)
    
    model.fit(X_train)

    scores = -model.decision_function(X_test)

    auc = roc_auc_score(y_test, scores)
    print(f"Creating results...")
    print(f"Model: Isolation Forest | AUC: {auc:.4f} ({auc*100:.2f}%)")

    plt.figure(figsize=(10, 5))
    sns.histplot(scores[y_test==0], color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(scores[y_test==1], color='red', label='Abnormal', kde=True, stat="density")
    plt.title(f'Isolation Forest Scores (AUC={auc:.2f})')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "if_dist.png"))

if __name__ == "__main__":
    train_and_eval()