# data_loader.py
import os

def collect_files(machine_type, split, label, base_dir="data/raw"):
    """
    data/raw/{machine_type}/{split}/{label}/
    """
    target_dir = os.path.join(base_dir, machine_type, split, label)
    files = []

    if not os.path.exists(target_dir):
        print(f"[WARN] Folder not found: {target_dir}")
        return files

    for root, _, filenames in os.walk(target_dir):
        for f in filenames:
            if f.lower().endswith(".wav"):
                files.append(os.path.join(root, f))

    return sorted(files)


def collect_all(machine_type, base_dir="data/raw"):
    return {
        "train_normal": collect_files(machine_type, "train", "normal", base_dir),
        "test_normal": collect_files(machine_type, "test", "normal", base_dir),
        "test_abnormal": collect_files(machine_type, "test", "abnormal", base_dir),
    }
