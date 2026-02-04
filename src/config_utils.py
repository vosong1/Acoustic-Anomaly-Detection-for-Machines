import json
from typing import Any, Dict, Optional

def load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = cfg.get(key, {})
    return val if isinstance(val, dict) else {}

def cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)
