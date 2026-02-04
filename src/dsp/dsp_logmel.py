import sys
import os
import argparse
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- THIẾT LẬP ĐƯỜNG DẪN ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from load_data import collect_all
from audio_utils import load_audio

# --- CẤU HÌNH TỐI ƯU CHO LOG-MEL ---
SR = 16000          # Tần số lấy mẫu chuẩn
N_FFT = 1024        # Cửa sổ phân tích
HOP_LENGTH = 512    # Độ chồng lấp 50%
N_MELS = 128        # 128 băng tần Mel (Con số vàng cho Machine Learning)

def compute_logmel(y, sr):
    """
    Chuyển đổi Audio -> Mel Spectrogram -> Log Scale (dB)
    """
    # 1. Tính Mel Spectrogram (Nén tần số)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, 
                                       hop_length=HOP_LENGTH, n_mels=N_MELS)
    
    # 2. Chuyển sang dB (Log scale) - Giúp phân biệt độ to nhỏ tốt hơn
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def extract_features(S_db):
    """
    Rút trích đặc trưng thống kê (Feature Engineering)
    Tối ưu: Thêm MAX để bắt các xung nhiễu bất thường
    """
    mean = np.mean(S_db, axis=1)    # Đặc trưng trung bình
    std = np.std(S_db, axis=1)      # Độ biến động
    max_val = np.max(S_db, axis=1)  # Giá trị lớn nhất (Quan trọng cho lỗi va đập)
    
    # Gộp lại thành 1 vector: 128 + 128 + 128 = 384 chiều
    return np.concatenate([mean, std, max_val])

def process_group(files, label):
    X, y = [], []
    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(f, SR)
            S_db = compute_logmel(audio, sr)
            vec = extract_features(S_db)
            X.append(vec)
            y.append(label)
        except Exception as e:
            print(f"Error {f}: {e}")
    return np.array(X), np.array(y)

def save_csv(X, y, out_path):
    # Đặt tên cột rõ ràng
    n_feats = X.shape[1] // 3 # Chia 3 vì có mean, std, max
    cols = ([f"mel_mean_{i}" for i in range(n_feats)] + 
            [f"mel_std_{i}" for i in range(n_feats)] + 
            [f"mel_max_{i}" for i in range(n_feats)] + 
            ["label"])
    
    df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | Shape: {df.shape}")

def main(machine_type):
    print(f"--- [OPTIMIZATION] GENERATING LOG-MEL FEATURES FOR: {machine_type} ---")
    
    # 1. Thu thập dữ liệu
    data = collect_all(machine_type)
    
    # 2. Chiến thuật chia 60-40 (Tự động chia chuẩn xác)
    all_normal = data["train_normal"] + data["test_normal"]
    abnormal_files = data["test_abnormal"]
    
    # random_state=42 để kết quả cố định, không bị nhảy lung tung
    train_files, test_normal_files = train_test_split(all_normal, test_size=0.4, random_state=42)

    # 3. Output Folder: features/logmel (Thay vì stft)
    out_dir = os.path.join(parent_dir, "../features/logmel", machine_type)
    os.makedirs(out_dir, exist_ok=True)

    # 4. Xử lý & Chuẩn hóa
    scaler = StandardScaler()

    print(f"Processing Train ({len(train_files)} files)...")
    X_train, y_train = process_group(train_files, 0)
    # Học scaler từ tập train
    X_train = scaler.fit_transform(X_train) 
    save_csv(X_train, y_train, os.path.join(out_dir, "train_normal.csv"))

    print(f"Processing Test Normal ({len(test_normal_files)} files)...")
    X_test_n, y_test_n = process_group(test_normal_files, 0)
    # Áp dụng scaler đã học
    X_test_n = scaler.transform(X_test_n)
    save_csv(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"))

    print(f"Processing Test Abnormal ({len(abnormal_files)} files)...")
    X_test_a, y_test_a = process_group(abnormal_files, 1)
    # Áp dụng scaler đã học
    X_test_a = scaler.transform(X_test_a)
    save_csv(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"))

    print("\n[DONE] Log-Mel Features generated successfully!")
    print(f"Feature Vector Size: {X_train.shape[1]} (Mean + Std + Max)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)