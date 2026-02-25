# src/check_data.py
from __future__ import annotations

import argparse
from pathlib import Path
import soundfile as sf

from load_data import DEFAULT_SNR_LIST, list_wavs_multi_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--snr", nargs="*", default=None, help='vd: -6db 0db 6db (mặc định: tất cả)')
    ap.add_argument("--max_files", type=int, default=50, help="chỉ đọc thử N file đầu mỗi class")
    args = ap.parse_args()

    snr_list = args.snr if args.snr is not None else DEFAULT_SNR_LIST
    data_root = Path(args.data_root)

    normal_files = list_wavs_multi_snr(data_root, args.machine, "normal", snr_list)
    abnormal_files = list_wavs_multi_snr(data_root, args.machine, "abnormal", snr_list)

    print(f"[INFO] machine={args.machine} snr={snr_list}")
    print(f"[INFO] normal files:   {len(normal_files)}")
    print(f"[INFO] abnormal files: {len(abnormal_files)}")

    # đọc thử vài file để xem sample rate / duration có ổn không
    def probe(files, label):
        if not files:
            print(f"[WARN] No files for {label}")
            return
        print(f"\n=== Probe {label} (first {min(args.max_files, len(files))}) ===")
        for p in files[: args.max_files]:
            try:
                info = sf.info(str(p))
                dur = info.frames / float(info.samplerate) if info.samplerate else 0.0
                print(f"{p.name} | sr={info.samplerate} | ch={info.channels} | dur={dur:.2f}s")
            except Exception as e:
                print(f"[BAD] {p} | {e}")

    probe(normal_files, "normal")
    probe(abnormal_files, "abnormal")


if __name__ == "__main__":
    main()