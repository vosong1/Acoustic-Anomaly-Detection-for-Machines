import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import collect_all
from audio_utils import load_audio
from dsp.dsp_mfcc import compute_mfcc
from dsp.handcraft import extract_all_ml_features

SR = 16000
N_MFCC = 13
FRAME_SIZE = 1024 
HOP_SIZE = 512     
N_FFT = 1024

def get_combo_vector(file_path):
    try:
        y, sr = load_audio(file_path, sr_target=SR)
        mfcc_matrix = compute_mfcc(
            y, sr, 
            n_mfcc=N_MFCC, 
            frame_size=FRAME_SIZE, 
            hop_size=HOP_SIZE, 
            n_fft=N_FFT
        )
        
        mfcc_mean = np.mean(mfcc_matrix, axis=0)  
        mfcc_std = np.std(mfcc_matrix, axis=0)    
        
        handcraft_vec = extract_all_ml_features(
            y, sr, 
            frame_size=FRAME_SIZE, 
            hop_size=HOP_SIZE, 
            n_fft=N_FFT
        )
    
        raw_max = np.max(np.abs(y))
        
        full_vector = np.concatenate([
            mfcc_mean, 
            mfcc_std, 
            handcraft_vec, 
            [raw_max]
        ])
        
        return full_vector

    except Exception as e:
        print(f"[ERROR] Lỗi khi xử lý file {file_path}: {e}")
        return None

def process_and_save(files, label, out_csv, scaler=None, fit_scaler=False):
    X, y = [], []
    
    print(f"--> Đang xử lý {len(files)} file cho: {out_csv}...")
    
    for idx, f in enumerate(files):
        vec = get_combo_vector(f)
        if vec is not None:
            X.append(vec)
            y.append(label)
        
        if idx % 100 == 0 and idx > 0:
            print(f"   ...Đã xong {idx} files")
            
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        print("[WARN] Không trích xuất được dữ liệu nào!")
        return None

    if fit_scaler:
        print("[INFO] Đang tính toán StandardScaler (Fit)...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            print("[ERROR] Thiếu Scaler cho tập Test!")
            return None
        X_scaled = scaler.transform(X)

    cols = [f"feat_{i}" for i in range(X_scaled.shape[1])] + ["label"]
    df = pd.DataFrame(np.column_stack([X_scaled, y]), columns=cols)
    df.to_csv(out_csv, index=False)
    
    print(f"[OK] Đã lưu: {out_csv} | Shape: {X_scaled.shape}")
    return scaler

def main(machine):

    data_dict = collect_all(machine)
    
    out_dir = f"features/combo_final/{machine}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== BẮT ĐẦU TRÍCH XUẤT COMBO (MFCC + HANDCRAFT) CHO MÁY: {machine} ===")
    
    scaler = process_and_save(
        data_dict["train_normal"], 
        label=0, 
        out_csv=os.path.join(out_dir, "train_normal.csv"),
        fit_scaler=True
    )
    
    if scaler is None:
        print("[STOP] Dừng chương trình vì lỗi ở tập Train.")
        return

    process_and_save(
        data_dict["test_normal"], 
        label=1, 
        out_csv=os.path.join(out_dir, "test_normal.csv"),
        scaler=scaler
    )
    
   
    process_and_save(
        data_dict["test_abnormal"], 
        label=0,
        out_csv=os.path.join(out_dir, "test_abnormal.csv"),
        scaler=scaler
    )
    
    print("\n[HOÀN TẤT] Bạn hãy chạy Isolation Forest với feature='combo_final'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)