# src/rename_arrange.py
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def arrange_from_id_structure(
    src_root: str,
    dest_root: str,
    machine: str = "fan",
    snr_folder: str = "0db",
    mode: str = "copy",  # "copy" hoặc "move"
):
    """
    Nhận dữ liệu kiểu:
      src_root/fan/id_00/normal/*.wav
      src_root/fan/id_00/abnormal/*.wav
    Và đưa về:
      dest_root/fan/<snr_folder>/normal/
      dest_root/fan/<snr_folder>/abnormal/

    Đổi tên để không trùng: <id>_<original>.wav
    """
    src_root = Path(src_root)
    dest_root = Path(dest_root) / machine / snr_folder
    dest_n = dest_root / "normal"
    dest_a = dest_root / "abnormal"
    dest_n.mkdir(parents=True, exist_ok=True)
    dest_a.mkdir(parents=True, exist_ok=True)

    copier = shutil.copy2 if mode == "copy" else shutil.move

    fan_root = src_root / machine
    if not fan_root.exists():
        raise FileNotFoundError(f"Không thấy folder machine tại: {fan_root}")

    total = 0
    for cls, out_dir in [("normal", dest_n), ("abnormal", dest_a)]:
        for wav in fan_root.glob(f"id_*/{cls}/*.wav"):
            id_part = wav.parts[-3]  # id_00
            new_name = f"{id_part}_{wav.name}"
            out_path = out_dir / new_name

            if out_path.exists():
                k = 1
                while True:
                    cand = out_dir / f"{id_part}_{wav.stem}__dup{k}{wav.suffix}"
                    if not cand.exists():
                        out_path = cand
                        break
                    k += 1

            copier(str(wav), str(out_path))
            total += 1

    print(f"✅ Done. {mode} {total} files -> {dest_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="vd: D:\\DATA\\0_dB_fan")
    ap.add_argument("--dest_data_root", default="data/raw", help="vd: data/raw (root chứa fan/)")
    ap.add_argument("--machine", default="fan")
    ap.add_argument("--snr_folder", default="0db", help="vd: -6db, 0db, 6db")
    ap.add_argument("--mode", default="copy", choices=["copy", "move"])
    args = ap.parse_args()

    arrange_from_id_structure(
        src_root=args.src_root,
        dest_root=args.dest_data_root,
        machine=args.machine,
        snr_folder=args.snr_folder,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()