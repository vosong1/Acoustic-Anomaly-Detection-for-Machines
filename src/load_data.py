# src/load_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_SNR_LIST = ["-6db", "0db", "6db"]


@dataclass
class AudioSplit:
    train_normal: List[Path]
    test_normal: List[Path]
    test_abnormal: List[Path]


def _list_wavs(data_root: Path, machine: str, snr: str, label: str) -> List[Path]:
    """
    data_root/machine/snr/label/*.wav
    label: "normal" hoặc "abnormal"
    """
    p = data_root / machine / snr / label
    if not p.exists():
        return []
    return sorted(p.glob("*.wav"))


def list_wavs_multi_snr(
    data_root: str | Path,
    machine: str,
    label: str,
    snr_list: Optional[Iterable[str]] = None,
) -> List[Path]:
    """
    Gom wav từ nhiều SNR vào chung 1 list (giữ thứ tự ổn định).
    """
    data_root = Path(data_root)
    snr_list = list(snr_list) if snr_list is not None else DEFAULT_SNR_LIST

    files: List[Path] = []
    for snr in snr_list:
        files.extend(_list_wavs(data_root, machine, snr, label))
    return files


def make_split(
    data_root: str | Path,
    machine: str,
    train_snr: Optional[Iterable[str]] = None,
    test_snr: Optional[Iterable[str]] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> AudioSplit:
    """
    Tạo split theo chuẩn unsupervised:
    - train: chỉ normal
    - test: normal + abnormal

    Nếu bạn muốn "train all normal" và "test all normal+abnormal" cùng snr_list
    thì truyền train_snr=test_snr=[...]
    """
    import random

    data_root = Path(data_root)
    train_snr = list(train_snr) if train_snr is not None else DEFAULT_SNR_LIST
    test_snr = list(test_snr) if test_snr is not None else DEFAULT_SNR_LIST

    # train normal
    all_train_normal = list_wavs_multi_snr(data_root, machine, "normal", train_snr)
    if not all_train_normal:
        raise FileNotFoundError(
            f"Không tìm thấy wav train normal tại: {data_root / machine} (train_snr={train_snr})"
        )

    # để reproducible
    rng = random.Random(seed)
    idx = list(range(len(all_train_normal)))
    rng.shuffle(idx)

    cut = int(len(idx) * train_ratio)
    train_idx = idx[:cut]
    test_norm_idx = idx[cut:]  # phần còn lại của normal dùng làm test_normal (cùng train_snr)

    train_normal = [all_train_normal[i] for i in train_idx]
    test_normal_from_train_snr = [all_train_normal[i] for i in test_norm_idx]

    # test abnormal lấy theo test_snr
    test_abnormal = list_wavs_multi_snr(data_root, machine, "abnormal", test_snr)
    if not test_abnormal:
        raise FileNotFoundError(
            f"Không tìm thấy wav test abnormal tại: {data_root / machine} (test_snr={test_snr})"
        )

    # test_normal tổng hợp:
    # - phần normal leftover từ train_snr (để có normal đối chiếu)
    # - + (tuỳ chọn) thêm normal của test_snr nếu khác train_snr
    if set(test_snr) == set(train_snr):
        test_normal = test_normal_from_train_snr
    else:
        extra_test_normal = list_wavs_multi_snr(data_root, machine, "normal", test_snr)
        test_normal = sorted(set(test_normal_from_train_snr + extra_test_normal))

    return AudioSplit(
        train_normal=train_normal,
        test_normal=test_normal,
        test_abnormal=test_abnormal,
    )