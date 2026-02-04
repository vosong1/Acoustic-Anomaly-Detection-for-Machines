import os
import shutil

BASE_DIR = r"D:\Spring 2026\AIL303m\Acoustic-Anomaly-Detection-for-Machines\data\raw"
OUTPUT_DIR = r"D:\Spring 2026\AIL303m\Acoustic-Anomaly-Detection-for-Machines\normal"

os.makedirs(OUTPUT_DIR, exist_ok=True)

counter = 0  

for id_name in sorted(os.listdir(BASE_DIR)):
    id_path = os.path.join(BASE_DIR, id_name)
    if not os.path.isdir(id_path):
        continue

    normal_path = os.path.join(id_path, "normal")
    if not os.path.exists(normal_path):
        print(f"⚠ Không có file trong {id_name}")
        continue

    wav_files = sorted(f for f in os.listdir(normal_path) if f.lower().endswith(".wav"))

    for fname in wav_files:
        src = os.path.join(normal_path, fname)
        new_name = f"{counter:07d}.wav"   
        dst = os.path.join(OUTPUT_DIR, new_name)

        shutil.copy2(src, dst)
        print(f"Đã copy: {src} -> {dst}")

        counter += 1

print("Hoàn tất gộp & đánh số lại!")
