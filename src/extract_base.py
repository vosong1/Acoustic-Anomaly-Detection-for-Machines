from pathlib import Path
import pandas as pd
from tqdm import tqdm

from config_utils import load_config, get_feature_config, ensure_dir
from load_data import build_dataset_index
from audio_utils import load_audio, feature_to_1d


def run_feature_extraction(feature_name: str, feature_fn, config_path: str = "config.json"):
    cfg = load_config(config_path)

    dataset_cfg = cfg["dataset"]
    audio_cfg = cfg["audio"]
    feature_cfg = get_feature_config(cfg, feature_name)

    raw_data_dir = dataset_cfg["raw_data_dir"]
    feature_root = Path(dataset_cfg["feature_dir"]) / feature_name
    machines = dataset_cfg["machines"]
    snr_levels = dataset_cfg["snr_levels"]
    splits = dataset_cfg["splits"]

    items = build_dataset_index(
        raw_data_dir=raw_data_dir,
        machines=machines,
        snr_levels=snr_levels,
        splits=splits,
    )

    if not items:
        raise ValueError("No .wav files found. Please check dataset paths in config.json")

    grouped_by_db = {}
    grouped_all_db = {}

    for item in tqdm(items, desc=f"Extracting {feature_name}"):
        y, sr = load_audio(
            item["wav_path"],
            sr=audio_cfg.get("sr", 16000),
            mono=audio_cfg.get("mono", True),
            duration=audio_cfg.get("duration", None),
        )

        feat = feature_fn(y, sr, feature_cfg)
        feat = feature_to_1d(feat)

        row = {
            "file": item["wav_path"].name,
            "machine": item["machine"],
            "snr": item["snr"],
            "split": item["split"],
        }

        for i, v in enumerate(feat):
            row[f"f{i:03d}"] = float(v)

        key_db = (item["machine"], item["snr"], item["split"])
        grouped_by_db.setdefault(key_db, []).append(row)

        key_all = (item["machine"], item["split"])
        grouped_all_db.setdefault(key_all, []).append(row)

    db_order = {"-6db": 0, "0db": 1, "6db": 2}

    for key in grouped_all_db:
        grouped_all_db[key] = sorted(
            grouped_all_db[key],
            key=lambda x: (db_order.get(x["snr"], 999), x["file"]),
        )

    for (machine, snr, split), rows in grouped_by_db.items():
        out_dir = ensure_dir(feature_root / machine / snr)
        out_file = out_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"Saved db-specific: {out_file}")

    for (machine, split), rows in grouped_all_db.items():
        out_dir = ensure_dir(feature_root / machine / "all_db")
        out_file = out_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(out_file, index=False)
        print(f"Saved merged-db: {out_file}")