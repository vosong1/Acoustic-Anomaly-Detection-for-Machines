import argparse
import numpy as np
import logging
from load_data import collect_all
import soundfile as sf


SR_TARGET = 16000

logging.basicConfig(
    filename="logs/data_check.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def check_audio(file_path):
    try:
        y, sr = sf.read(file_path)
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr <= 0:
            return None, None, ["Invalid sample rate"]
        duration = len(y) / sr
        issues = []
        if sr != SR_TARGET:
            issues.append(f"SR={sr}")
        if duration < 1:
            issues.append("Too short")
        if np.max(np.abs(y)) < 1e-4:
            issues.append("Silent")
        return sr, duration, issues
    except Exception as e:
        return None, None, [str(e)]

def main(machine_type):
    data = collect_all(machine_type)
    print(f"\n=== CHECK DATA: {machine_type.upper()} ===")
    total_files = 0
    for group, files in data.items():
        n_files = len(files)
        total_files += n_files

        print(f"\n[{group}]")
        print(f"  Number of files: {n_files}")

        srs, durations = [], []
        issue_count = 0

        for f in files:
            sr, dur, issues = check_audio(f)

            if sr is None:
                issue_count += 1
                logging.info(f"{f} ERROR {issues}")
                continue

            srs.append(sr)
            durations.append(dur)

            if issues:
                issue_count += 1
                logging.info(f"{f} ISSUES {issues}")

        if srs:
            print("  SR range:", min(srs), "→", max(srs))
            print("  Duration (s):",
                  round(min(durations), 2),
                  round(np.mean(durations), 2),
                  round(max(durations), 2))
            print("  Files with issues:", issue_count)

    print("\n=== DATASET SUMMARY ===")
    print("Total files:", total_files)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="valve")
    args = parser.parse_args()
    main(args.machine)
