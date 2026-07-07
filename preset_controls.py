"""Pure helpers for the dialect demo preset controls."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple


PRESET_APPLY_OUTPUT_COUNT = 15


def coerce_seed(seed_value) -> Optional[int]:
    if seed_value is None or seed_value == "":
        return None
    return int(seed_value)


def collect_preset_data(
    dialect_value: str,
    role_value: str,
    text_value: str,
    rewritten_value: str,
    use_prompt_text: bool,
    prompt_text_value: str,
    auto_rewrite: bool,
    cfg_value_input: float,
    do_normalize: bool,
    denoise: bool,
    dit_steps_value: int,
    seed_value,
    random_seed_value: bool,
) -> Dict[str, Any]:
    return {
        "dialect": dialect_value or "粤语",
        "role_description": role_value or "",
        "mandarin_text": text_value or "",
        "rewritten_text": rewritten_value or "",
        "use_prompt_text": bool(use_prompt_text),
        "prompt_text": prompt_text_value or "",
        "auto_rewrite": bool(auto_rewrite),
        "cfg_value": float(cfg_value_input) if cfg_value_input is not None else 2.0,
        "do_normalize": bool(do_normalize),
        "denoise": bool(denoise),
        "dit_steps": int(dit_steps_value) if dit_steps_value is not None else 10,
        "seed": coerce_seed(seed_value),
        "random_seed": bool(random_seed_value),
    }


def empty_preset_apply_update_kwargs() -> Tuple[Dict[str, Any], ...]:
    return tuple({} for _ in range(PRESET_APPLY_OUTPUT_COUNT))


def build_preset_apply_update_kwargs(
    data: Dict[str, Any],
    reference_exists: Callable[[str], bool] = os.path.exists,
) -> Tuple[Tuple[Dict[str, Any], ...], bool]:
    ref_path = data.get("reference_audio") or None
    reference_missing = False
    if ref_path and not reference_exists(ref_path):
        reference_missing = True
        ref_path = None

    use_prompt = bool(data.get("use_prompt_text", False))
    random_seed_value = bool(data.get("random_seed", True))
    updates = (
        {"value": ref_path},
        {"value": data.get("dialect", "粤语")},
        {"value": data.get("role_description", ""), "visible": not use_prompt},
        {"value": data.get("mandarin_text", "")},
        {"value": data.get("rewritten_text", ""), "interactive": True},
        {"value": use_prompt},
        {"value": data.get("prompt_text", ""), "visible": use_prompt},
        {"value": data.get("auto_rewrite", True)},
        {"value": data.get("cfg_value", 2.0)},
        {"value": data.get("do_normalize", False)},
        {"value": data.get("denoise", False)},
        {"value": data.get("dit_steps", 10)},
        {"value": data.get("seed"), "interactive": not random_seed_value},
        {"value": random_seed_value},
        {"value": "", "visible": False},
    )
    return updates, reference_missing
