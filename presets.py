"""Small on-disk preset store for the dialect Gradio demo."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


PRESET_VERSION = "1.0"
PRESETS_DIRNAME = "presets"


def get_presets_dir() -> Path:
    configured = os.getenv("VOXCPM_PRESETS_DIR")
    if configured:
        path = Path(configured).expanduser()
    else:
        data_root = (os.getenv("DATA_ROOT") or "").strip()
        if data_root:
            path = Path(data_root.rstrip("/") or "/").expanduser() / PRESETS_DIRNAME
        else:
            path = Path(__file__).parent.resolve() / PRESETS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_preset_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name or "untitled"


def _preset_dir(name: str) -> Path:
    return get_presets_dir() / safe_preset_name(name)


def _preset_file(name: str) -> Path:
    return _preset_dir(name) / "preset.json"


def list_presets() -> List[str]:
    presets_dir = get_presets_dir()
    return sorted(
        item.name
        for item in presets_dir.iterdir()
        if item.is_dir() and (item / "preset.json").is_file()
    )


def _remove_managed_files(preset_dir: Path, *, include_preset_json: bool = True) -> None:
    for path in preset_dir.glob("reference_audio.*"):
        if path.is_file():
            path.unlink()
    preset_json = preset_dir / "preset.json"
    if include_preset_json and preset_json.is_file():
        preset_json.unlink()


def _copy_reference_audio(src: Optional[str], preset_dir: Path) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.is_file():
        return ""
    suffix = src_path.suffix or ".wav"
    dst_name = f"reference_audio{suffix}"
    shutil.copy2(src_path, preset_dir / dst_name)
    return dst_name


def save_preset(name: str, data: Dict[str, Any], reference_audio: Optional[str] = None) -> str:
    safe_name = safe_preset_name(name)
    preset_dir = _preset_dir(safe_name)
    preset_dir.mkdir(parents=True, exist_ok=True)
    _remove_managed_files(preset_dir, include_preset_json=False)

    reference_audio_name = _copy_reference_audio(reference_audio, preset_dir)
    payload = {
        "version": PRESET_VERSION,
        "display_name": (name or "").strip() or safe_name,
        **data,
        "reference_audio": reference_audio_name,
    }

    tmp_path = preset_dir / "preset.json.tmp"
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    tmp_path.replace(preset_dir / "preset.json")
    return safe_name


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    preset_dir = _preset_dir(name)
    preset_file = preset_dir / "preset.json"
    if not preset_file.is_file():
        return None
    with preset_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if data.get("reference_audio"):
        data["reference_audio"] = str(preset_dir / data["reference_audio"])
    return data


def delete_preset(name: str) -> bool:
    preset_file = _preset_file(name)
    if not preset_file.is_file():
        return False
    _remove_managed_files(preset_file.parent)
    return True


def preset_exists(name: str) -> bool:
    return _preset_file(name).is_file()
