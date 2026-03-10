import json
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_feature_config(cfg: dict, feature_name: str) -> dict:
    if feature_name not in cfg:
        raise KeyError(f"Missing '{feature_name}' section in config.json")
    return cfg[feature_name]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path