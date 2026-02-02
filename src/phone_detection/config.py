from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

@dataclass
class Settings:
    
    camera: dict
    models: dict
    face: dict
    hands: dict
    phone: dict
    association: dict
    smoothing: dict
    tracking: dict

def repo_root() -> Path:
    # expect <repo>/src/phone_detection/config.py

    return Path(__file__).resolve().parents[2]

def load_settings(toml_path: str | Path = "config/settings.toml") -> Setting:

    """
    Load config/settings.toml relative to repo root unless absolute path is provided.
    """

    base = repo_root()
    toml_path = Path(toml_path)
    path = toml_path if toml_path.is_absolute() else (base / toml_path)

    with path.open("rb") as f:
        raw = tomllib.load(f)


    # Safe defaults for missing sections
    camera = raw.get("camera", {})
    models = raw.get("models", {})
    face = raw.get("face", {})
    hands = raw.get("hands", {})
    phone = raw.get("phone", {})
    association = raw.get("association", {})
    smoothing = raw.get("smoothing", {})
    tracking = raw.get("tracking", {})

    return Settings(
            camera = camera,
            models = models,
            face = face,
            hands = hands,
            phone = phone,
            association = association,
            smoothing = smoothing,
            tracking = tracking,
            )


def resolve_model_path(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.is_absolute() else (repo_root() / p))




