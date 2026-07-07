"""Dependency-light readiness checks for the dialect Gradio demo."""

from __future__ import annotations

import html
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple


REQUIRED_MODEL_FILES = (
    "config.json",
    "audiovae.pth",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_voxcpm2.py",
)


def _escape_html(value: str) -> str:
    return html.escape(value or "", quote=True)


def _readiness_card(title: str, status: str, detail: str) -> str:
    status_label = {"ok": "正常", "warn": "待确认", "error": "异常"}.get(status, "待确认")
    return (
        f'<div class="service-status service-status--{status}">'
        '<div class="service-status__line">'
        '<span class="service-status__dot" aria-hidden="true"></span>'
        f"<span>{_escape_html(title)} · {status_label}</span>"
        "</div>"
        f'<div class="service-status__detail">{_escape_html(detail)}</div>'
        "</div>"
    )


def _check_kralapi_connectivity(
    *,
    api_key: Optional[str],
    base_url: str,
    rewrite_model: str,
    urlopen: Callable = urllib.request.urlopen,
) -> Tuple[str, str]:
    if not api_key:
        return "error", "未配置 KRALAPI_API_KEY，自动方言改写不可用。"

    request = urllib.request.Request(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=8) as response:
            if 200 <= response.status < 300:
                return "ok", f"已连通 {base_url}，改写模型：{rewrite_model}。"
            return "warn", f"接口返回 HTTP {response.status}，生成时可能需要进一步确认。"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "error", f"接口可达，但密钥认证失败：HTTP {exc.code}。"
        return "warn", f"接口可达但返回 HTTP {exc.code}，生成时可能失败。"
    except urllib.error.URLError as exc:
        return "error", f"无法连通 {base_url}：{exc.reason}"
    except TimeoutError:
        return "error", f"连接 {base_url} 超时。"


def _get_data_root_path(data_root_value: Optional[str] = None) -> Optional[Path]:
    if data_root_value is None:
        data_root_value = os.getenv("DATA_ROOT") or ""
    data_root_value = data_root_value.strip()
    if not data_root_value:
        return None
    data_root = data_root_value.rstrip("/") or "/"
    return Path(data_root).expanduser()


def _check_data_root_mount(data_root_value: Optional[str] = None) -> Tuple[str, str]:
    data_root = _get_data_root_path(data_root_value)
    if data_root is None:
        return "warn", "未设置 DATA_ROOT；远端数据盘挂载点未知。8808 demo 推荐指向云盘/数据盘，例如 /root/autodl-tmp。"
    if not data_root.exists():
        return "error", f"DATA_ROOT 挂载目录不存在：{data_root}。请先挂载云盘/数据盘，或修正 DATA_ROOT。"
    if not data_root.is_dir():
        return "error", f"DATA_ROOT 存在但不是目录：{data_root}。请改成云盘/数据盘挂载目录。"
    return "ok", f"DATA_ROOT 挂载目录可访问：{data_root}"


def _check_data_root_subdirs(data_root_value: Optional[str] = None) -> Tuple[str, str]:
    data_root = _get_data_root_path(data_root_value)
    if data_root is None:
        return "warn", "未设置 DATA_ROOT；无法检查 $DATA_ROOT/models 和 $DATA_ROOT/cache。"
    if not data_root.is_dir():
        return "error", "DATA_ROOT 挂载目录不可用；先修复挂载目录，再检查 models/cache。"

    required_subdirs = ("models", "cache")
    missing = [str(data_root / subdir) for subdir in required_subdirs if not (data_root / subdir).is_dir()]
    if missing:
        return (
            "error",
            "DATA_ROOT 可访问，但缺少 8808 demo 子目录："
            + "、".join(missing)
            + "。请在挂载盘创建 models/ 和 cache/，或修正 DATA_ROOT。",
        )
    return "ok", f"已找到 8808 demo 子目录：{data_root / 'models'}，{data_root / 'cache'}"


def _is_under_path(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _check_model_path(model_id: str, data_root_value: Optional[str] = None) -> Tuple[str, str]:
    model_value = (model_id or "").strip()
    if not model_value:
        return "error", "未传入模型路径或模型 ID。"

    model_path = Path(model_value).expanduser()
    if model_path.exists():
        if model_path.is_dir():
            missing = [file for file in REQUIRED_MODEL_FILES if not (model_path / file).is_file()]
            weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if not weight_files:
                missing.append("*.safetensors 或 *.bin 权重文件")
            if missing:
                return "error", f"模型目录存在但文件不全：{model_path}。缺少：{', '.join(missing)}。"
            return "ok", f"VoxCPM2 模型目录和关键文件已就绪：{model_path}"
        return "warn", f"路径存在但不是目录：{model_path}"

    looks_like_local_path = model_value.startswith(("/", "./", "../", "~"))
    if looks_like_local_path:
        data_root = _get_data_root_path(data_root_value)
        if data_root is not None and _is_under_path(model_path, data_root / "models"):
            if data_root.is_dir():
                data_root_hint = "DATA_ROOT 挂载可访问时，这通常表示模型文件还没放到 "
            else:
                data_root_hint = "DATA_ROOT 已配置但挂载目录不可用；挂载修复后，也请确认模型文件已放到 "
            return (
                "error",
                f"模型目录缺失：{model_path}。{data_root_hint}"
                "$DATA_ROOT/models/VoxCPM2，或 VOXCPM_MODEL_PATH 指错目录。",
            )
        return "error", f"本地模型路径不存在：{model_path}。如果模型在远端数据盘，请先确认 DATA_ROOT 已挂载。"
    return "warn", f"当前使用模型 ID：{model_value}；未检查本地权重目录。"


def build_service_readiness_html(
    model_id: str,
    *,
    api_key: Optional[str],
    base_url: str,
    rewrite_model: str,
    data_root_value: Optional[str] = None,
    urlopen: Callable = urllib.request.urlopen,
) -> str:
    api_key_status = "ok" if api_key else "error"
    api_key_detail = (
        "已读取 KRALAPI_API_KEY，页面可调用方言改写。"
        if api_key
        else "未读取 KRALAPI_API_KEY；如开启自动方言改写，生成会失败。"
    )
    rewrite_status, rewrite_detail = _check_kralapi_connectivity(
        api_key=api_key,
        base_url=base_url,
        rewrite_model=rewrite_model,
        urlopen=urlopen,
    )
    data_root_status, data_root_detail = _check_data_root_mount(data_root_value)
    subdirs_status, subdirs_detail = _check_data_root_subdirs(data_root_value)
    model_status, model_detail = _check_model_path(model_id, data_root_value)
    readiness_statuses = (api_key_status, rewrite_status, data_root_status, subdirs_status, model_status)
    overall = "服务就绪" if all(s == "ok" for s in readiness_statuses) else "需要处理配置项"
    return (
        '<section class="service-readiness">'
        '<div class="service-readiness__header">'
        '<div class="service-readiness__title">服务就绪状态</div>'
        f'<div class="service-readiness__summary">{_escape_html(overall)} · 不提前加载 VoxCPM 大模型</div>'
        "</div>"
        '<div class="service-readiness__grid">'
        + _readiness_card("改写密钥", api_key_status, api_key_detail)
        + _readiness_card("方言改写接口", rewrite_status, rewrite_detail)
        + _readiness_card("DATA_ROOT 挂载", data_root_status, data_root_detail)
        + _readiness_card("models/cache 目录", subdirs_status, subdirs_detail)
        + _readiness_card("模型路径", model_status, model_detail)
        + "</div>"
        "</section>"
    )
