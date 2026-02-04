import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, roc_curve

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MACHINE_TYPE = "fan"
DATA_DIR = os.path.join(parent_dir, "features/stft_pca", MACHINE_TYPE)
RESULT_DIR = os.path.join(parent_dir, "results/gmm", MACHINE_TYPE)

os.makedirs(RESULT_DIR, exist_ok=True)

def load_data():
    print(f"Loading data from: {DATA_DIR}")
    try:
        df_train = pd.read_csv(os.path.join(DATA_DIR, "train_normal.csv"))
        df_test_n = pd.read_csv(os.path.join(DATA_DIR, "test_normal.csv"))
        df_test_a = pd.read_csv(os.path.join(DATA_DIR, "test_abnormal.csv"))
        
        X_train = df_train.drop(columns=["label"]).values
        
        X_test_n = df_test_n.drop(columns=["label"]).values
        y_test_n = np.zeros(len(X_test_n))
        
        X_test_a = df_test_a.drop(columns=["label"]).values
        y_test_a = np.ones(len(X_test_a))
        
        X_test = np.concatenate([X_test_n, X_test_a])
        y_test = np.concatenate([y_test_n, y_test_a])
        
        return X_train, X_test, y_test
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy file CSV PCA. Hãy chạy 'dsp_stft_pca.py' trước!")
        sys.exit(1)

def train_and_eval():
    X_train, X_test, y_test = load_data()
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    N_COMPONENTS = 4 
    
    print(f"\nTraining GMM (n_components={N_COMPONENTS})...")
    gmm = GaussianMixture(n_components=N_COMPONENTS, 
                          covariance_type='full', 
                          max_iter=100, 
                          random_state=42)
    
    gmm.fit(X_train)
    print("Training finished.")

    log_likelihood = gmm.score_samples(X_test)

    anomaly_scores = -log_likelihood

    auc = roc_auc_score(y_test, anomaly_scores)
    print(f"\n================ RESULT ================")
    print(f"Model: GMM (Gaussian Mixture Model)")
    print(f"Machine: {MACHINE_TYPE}")
    print(f"AUC Score: {auc:.4f} ({auc*100:.2f}%)")
    print(f"========================================")

    plt.figure(figsize=(10, 5))
    sns.histplot(anomaly_scores[y_test==0], color='blue', label='Normal', kde=True, stat="density")
    sns.histplot(anomaly_scores[y_test==1], color='red', label='Abnormal', kde=True, stat="density")
    plt.title(f'GMM Anomaly Score Distribution (AUC={auc:.2f})')
    plt.xlabel('Anomaly Score (Negative Log-Likelihood)')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "score_dist.png"))
    print(f"Đã lưu biểu đồ phân phối tại: {RESULT_DIR}/score_dist.png")

if __name__ == "__main__":
    train_and_eval()