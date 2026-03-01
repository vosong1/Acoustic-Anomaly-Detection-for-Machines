import os
import shutil

BASE_DIR = r"D:\fan"
OUTPUT_DIR = r"D:\Acoustic-Anomaly-Detection-for-Machines\data\raw\fan\6db\abnormal"

os.makedirs(OUTPUT_DIR, exist_ok=True)

counter = 0  

for id_name in sorted(os.listdir(BASE_DIR)):
    id_path = os.path.join(BASE_DIR, id_name)
    if not os.path.isdir(id_path):
        continue

    abnormal_path = os.path.join(id_path, "abnormal")
    if not os.path.exists(abnormal_path):
        print(f"⚠ Không có file trong {id_name}")
        continue

    wav_files = sorted(f for f in os.listdir(abnormal_path) if f.lower().endswith(".wav"))

    for fname in wav_files:
        src = os.path.join(abnormal_path, fname)
        new_name = f"{counter:07d}.wav"   
        dst = os.path.join(OUTPUT_DIR, new_name)

        shutil.copy2(src, dst)
        print(f"Đã copy: {src} -> {dst}")

        counter += 1

print("Hoàn tất gộp & đánh số lại!")
