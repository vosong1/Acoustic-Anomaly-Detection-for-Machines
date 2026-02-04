import sys
import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

# --- CẤU HÌNH ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
MACHINE_TYPE = "fan"
DATA_DIR = os.path.join(parent_dir, "features/stft_pca", MACHINE_TYPE)

def load_data():
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
    except Exception as e:
        print(f"Lỗi load data: {e}")
        sys.exit(1)

def main():
    X_train, X_test, y_test = load_data()
    print(f"--- AUTO TUNING GMM FOR {MACHINE_TYPE} ---")
    print(f"Data shape: {X_train.shape}")
    
    # DANH SÁCH THAM SỐ CẦN THỬ
    # n_components: Số lượng cụm (thử từ đơn giản đến phức tạp)
    n_components_list = [2, 4, 8, 16, 32, 64]
    
    # covariance_type: Cách tính hình dáng cụm
    # 'full': Hình dạng tự do (Mạnh nhất nhưng dễ overfitting)
    # 'diag': Hình bầu dục dọc theo trục (Nhanh, nhẹ)
    # 'tied': Các cụm có hình dáng giống nhau
    cov_types = ['full', 'diag', 'tied']

    best_auc = 0
    best_params = {}

    print(f"{'COMPONENTS':<12} | {'COVARIANCE':<12} | {'AUC SCORE':<10}")
    print("-" * 40)

    for n_comp in n_components_list:
        for cov in cov_types:
            try:
                # Train model
                gmm = GaussianMixture(n_components=n_comp, 
                                      covariance_type=cov, 
                                      max_iter=100, 
                                      random_state=42)
                gmm.fit(X_train)
                
                # Evaluate
                log_likelihood = gmm.score_samples(X_test)
                scores = -log_likelihood
                auc = roc_auc_score(y_test, scores)
                
                print(f"{n_comp:<12} | {cov:<12} | {auc:.4f}")
                
                # Lưu lại nếu kết quả tốt hơn
                if auc > best_auc:
                    best_auc = auc
                    best_params = {'n_components': n_comp, 'covariance_type': cov}
            except Exception as e:
                print(f"{n_comp:<12} | {cov:<12} | ERROR")

    print("\n" + "="*40)
    print(f"🏆 BEST RESULT: AUC = {best_auc:.4f} ({best_auc*100:.2f}%)")
    print(f"⚙️  PARAMS: {best_params}")
    print("="*40)

if __name__ == "__main__":
    main()