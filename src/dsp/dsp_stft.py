import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler
from load_data import collect_all
from audio_utils import load_audio

# CẤU HÌNH CHO STFT
SR = 16000
N_FFT = 1024       # Độ phân giải tần số
HOP_LENGTH = 512   # Bước nhảy thời gian
MAX_PLOT = 50 

def plot_waveform(y, sr, title, save_path):
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_spectrogram(S_db, sr, title, save_path):
    # S_db: Amplitude in dB
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_stft_magnitude(y):
    """
    Tính STFT và lấy độ lớn (Magnitude).
    Output shape: (Frequency_Bins, Time_Frames)
    Với n_fft=1024 -> có 513 frequency bins.
    """
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)
    return S_mag

def stft_to_vector(S_mag):
    """
    Chuyển ma trận STFT thành vector để train Machine Learning.
    S_mag shape: (n_freq, n_time)
    Chúng ta tính Mean và Std dọc theo trục thời gian (axis=1).
    """
    # Chuyển về dB để phân bố dữ liệu tốt hơn cho ML (giống log-mel)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    mean = np.mean(S_db, axis=1) # Trung bình năng lượng của từng tần số
    std = np.std(S_db, axis=1)   # Độ biến động năng lượng của từng tần số
    
    return np.concatenate([mean, std])

def process_group(files, label, plot_dir, prefix):
    X, y = [], []
    os.makedirs(plot_dir, exist_ok=True)

    for i, f in enumerate(files):
        try:
            audio, sr = load_audio(f, SR)
            
            # 1. Tính STFT
            S_mag = compute_stft_magnitude(audio)
            
            # 2. Chuyển thành Vector (Feature Extraction)
            vec = stft_to_vector(S_mag)
            X.append(vec)
            y.append(label)

            # 3. Vẽ hình (Chỉ vẽ 50 hình đầu)
            if i < MAX_PLOT:
                base = f"{prefix}_{i:04d}"
                # Vẽ sóng
                plot_waveform(audio, sr, f"{base} waveform", 
                              os.path.join(plot_dir, f"{base}_waveform.png"))
                
                # Vẽ Spectrogram (dùng dB để dễ nhìn)
                S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
                plot_spectrogram(S_db, sr, f"{base} spectrogram", 
                                 os.path.join(plot_dir, f"{base}_spectrogram.png"))

        except Exception as e:
            print(f"[ERROR] File: {f} - {e}")

    return np.array(X), np.array(y)

def save_split(X, y, out_csv, n_bins):
    # Tạo tên cột: freq_mean_0, freq_mean_1...
    cols = (
        [f"freq_mean_{i}" for i in range(n_bins)] +
        [f"freq_std_{i}" for i in range(n_bins)] +
        ["label"]
    )
    
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

def main(machine_type):
    data = collect_all(machine_type)

    # Lưu ý: Sửa đường dẫn output ra thư mục gốc để dễ tìm
    out_dir = f"features/stft/{machine_type}"
    plot_base = f"results/stft/{machine_type}"

    # Đảm bảo đường dẫn này trỏ về đúng thư mục gốc dự án (dùng parent_dir)
    out_dir = os.path.join(parent_dir, "../features/stft", machine_type)
    plot_base = os.path.join(parent_dir, "../results/stft", machine_type)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_base, exist_ok=True)

    plot_dirs = {
        "train_normal": os.path.join(plot_base, "train_normal"),
        "test_normal": os.path.join(plot_base, "test_normal"),
        "test_abnormal": os.path.join(plot_base, "test_abnormal"),
    }

    scaler = StandardScaler()

    # Tính số lượng bin tần số: n_fft/2 + 1
    n_bins = N_FFT // 2 + 1  # 1024/2 + 1 = 513 features mean + 513 features std

    # -------- TRAIN NORMAL --------
    print("Processing Train Normal...")
    X_train, y_train = process_group(data["train_normal"], 0, plot_dirs["train_normal"], "normal")
    
    if len(X_train) == 0:
        print("[WARN] No train_normal data")
        return

    # Fit scaler trên tập train
    X_train = scaler.fit_transform(X_train)
    save_split(X_train, y_train, os.path.join(out_dir, "train_normal.csv"), n_bins)

    # -------- TEST NORMAL --------
    print("Processing Test Normal...")
    X_test_n, y_test_n = process_group(data["test_normal"], 0, plot_dirs["test_normal"], "normal")
    if len(X_test_n) > 0:
        X_test_n = scaler.transform(X_test_n)
        save_split(X_test_n, y_test_n, os.path.join(out_dir, "test_normal.csv"), n_bins)

    # -------- TEST ABNORMAL --------
    print("Processing Test Abnormal...")
    X_test_a, y_test_a = process_group(data["test_abnormal"], 1, plot_dirs["test_abnormal"], "abnormal")
    if len(X_test_a) > 0:
        X_test_a = scaler.transform(X_test_a)
        save_split(X_test_a, y_test_a, os.path.join(out_dir, "test_abnormal.csv"), n_bins)

    print(f"Done STFT for: {machine_type}. Feature Vector Size: {X_train.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)