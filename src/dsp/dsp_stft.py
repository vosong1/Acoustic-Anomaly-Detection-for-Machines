import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from load_data import collect_all
from audio_utils import load_audio

SR = 16000          
N_FFT = 1024        
HOP_LENGTH = 512    

PCA_COMPONENTS = 0.95 

def plot_spectrogram(S_db, sr, title, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, 
                             x_axis="time", y_axis="linear", 
                             cmap='magma') 
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_stft_features(y):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def feature_vector_from_spec(S_db):
    mean = np.mean(S_db, axis=1) 
    std = np.std(S_db, axis=1)
    return np.concatenate([mean, std])

def process_group(files, label, plot_dir, prefix):
    X, y = [], []
    os.makedirs(plot_dir, exist_ok=True)
    MAX_PLOT = 5 

    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(f, SR)
            S_db = compute_stft_features(audio)
            vec = feature_vector_from_spec(S_db)
            X.append(vec)
            y.append(label)

            if i < MAX_PLOT:
                base = f"{prefix}_{i:04d}"
                plot_spectrogram(S_db, sr, f"{base} STFT", 
                                 os.path.join(plot_dir, f"{base}_stft.png"))
        except Exception as e:
            print(f"[ERROR] File: {f} - {e}")

    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

def save_split_pca(X, y, out_csv):
    if len(X) == 0: return
    
    # Vì dùng PCA, tên cột không còn là tần số nữa mà là Component
    n_features = X.shape[1]
    cols = [f"pca_component_{i}" for i in range(n_features)] + ["label"]
    
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} | Shape: {df.shape}")

def main(machine_type):
    # 1. Load Data & Split 60/40
    data = collect_all(machine_type)
    all_normal_files = data["train_normal"] + data["test_normal"]
    abnormal_files = data["test_abnormal"]

    print(f"--- FEATURE REDUCTION WITH PCA ---")
    train_files, test_normal_files = train_test_split(all_normal_files, test_size=0.4, random_state=42)

    # Setup đường dẫn
    out_dir = os.path.join(parent_dir, "../features/stft_pca", machine_type) # Lưu folder riêng stft_pca
    plot_base = os.path.join(parent_dir, "../results/stft", machine_type)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_base, exist_ok=True)

    scaler = StandardScaler()
    pca = PCA(n_components=PCA_COMPONENTS)

    # --- BƯỚC 1: TRAIN SET ---
    print("\n[1/3] Processing Train Normal...")
    X_train, y_train = process_group(train_files, 0, os.path.join(plot_base, "train_normal"), "normal")
    
    if len(X_train) > 0:
        print(f"Original Features: {X_train.shape[1]} columns")
        
        # 1. Chuẩn hóa (Fit Scaler)
        X_train = scaler.fit_transform(X_train)
        
        # 2. Giảm chiều (Fit PCA) -> QUAN TRỌNG
        X_train = pca.fit_transform(X_train)
        
        print(f"Reduced Features (PCA): {X_train.shape[1]} columns (Retained 95% variance)")
        save_split_pca(X_train, y_train, os.path.join(out_dir, "train_normal.csv"))
    
    # --- BƯỚC 2: TEST NORMAL ---
    print("\n[2/3] Processing Test Normal...")
    X_test_n, y_test_n = process_group(test_normal_files, 0, os.path.join(plot_base, "test_normal"), "normal")
    if len(X_test_n) > 0:
        X_test_n = scaler.transform(X_test_n) # Chỉ transform
        X_test_n = pca.transform(X_test_n)    # Chỉ transform
        save_split_pca(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"))

    # --- BƯỚC 3: TEST ABNORMAL ---
    print("\n[3/3] Processing Test Abnormal...")
    X_test_a, y_test_a = process_group(abnormal_files, 1, os.path.join(plot_base, "test_abnormal"), "abnormal")
    if len(X_test_a) > 0:
        X_test_a = scaler.transform(X_test_a) # Chỉ transform
        X_test_a = pca.transform(X_test_a)    # Chỉ transform
        save_split_pca(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"))

    print(f"\n[DONE] Saved optimized features to: features/stft_pca/{machine_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)