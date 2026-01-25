import argparse
import numpy as np
import logging
from load_data import collect_all
from audio_utils import load_audio

SR_TARGET = 16000

logging.basicConfig(
    filename="logs/data_check.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
def check_audio(file_path):
    try:
        y, sr = load_audio(file_path, sr_target=None)

        duration = len(y) / sr
        issues = []

        if sr != SR_TARGET:
            issues.append(f"SR={sr}")

        if duration < 1.0:
            issues.append("Too short")

        if np.max(np.abs(y)) < 1e-4:
            issues.append("Silent")

        return sr, duration, issues

    except Exception as e:
        return None, None, [f"Load error: {str(e)}"]

def main(machine_type):
    data = collect_all(machine_type)

    print(f"\n=== CHECK DATA: {machine_type.upper()} ===")

    total_files = 0
    total_issues = 0

    for group, files in data.items():
        print(f"\n[{group}] - {len(files)} files")
        total_files += len(files)

        srs, durations = [], []
        issue_count = 0

        for f in files:
            sr, dur, issues = check_audio(f)

            if sr is None:
                issue_count += 1
                total_issues += 1
                logging.info(f"{f} ERROR {issues}")
                continue

            srs.append(sr)
            durations.append(dur)

            if issues:
                issue_count += 1
                total_issues += 1
                logging.info(f"{f} ISSUES {issues}")

        if srs:
            print("  SR range:", min(srs), "→", max(srs))
            print("  Duration (s):",
                  "min =", round(min(durations), 2),
                  "| mean =", round(np.mean(durations), 2),
                  "| max =", round(max(durations), 2))
            print("  Files with issues:", issue_count)

    print("\n=== SUMMARY ===")
    print("Total files:", total_files)
    print("Total problematic files:", total_issues)
    print("Log saved to: logs/data_check.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="fan")
    args = parser.parse_args()
    main(args.machine)