import os
import random

def collect_label(machine_type, label, base_dir="data/raw"):
    target_dir = os.path.join(base_dir, machine_type, label)
    files = []

    if not os.path.exists(target_dir):
        print(f"[WARN] Folder not found: {target_dir}")
        return files

    for root, _, filenames in os.walk(target_dir):
        for f in filenames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))

    return sorted(files)


def split_train_test(files, test_ratio=0.3, seed=42):
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    n_test = int(len(files) * test_ratio)
    test_files = files[:n_test]
    train_files = files[n_test:]

    return train_files, test_files


def collect_all(machine_type, base_dir="data/raw", test_ratio=0.3):
    normal_files = collect_label(machine_type, "normal", base_dir)
    abnormal_files = collect_label(machine_type, "abnormal", base_dir)

    train_normal, test_normal = split_train_test(normal_files, test_ratio)

    return {
        "train_normal": train_normal,
        "test_normal": test_normal,
        "test_abnormal": abnormal_files
    }
