from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "config/config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    return cfg

def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
