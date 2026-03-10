from pathlib import Path


def build_dataset_index(raw_data_dir: str, machines: list, snr_levels: list, splits: list) -> list:
    """
    Scan dataset structure:
    data/raw/{machine}/{snr}/{split}/*.wav
    """
    items = []
    raw_root = Path(raw_data_dir)

    for machine in machines:
        for snr in snr_levels:
            for split in splits:
                folder = raw_root / machine / snr / split
                if not folder.exists():
                    continue

                wav_files = sorted(folder.glob("*.wav"))
                for wav_path in wav_files:
                    items.append(
                        {
                            "machine": machine,
                            "snr": snr,
                            "split": split,
                            "wav_path": wav_path,
                        }
                    )

    return items