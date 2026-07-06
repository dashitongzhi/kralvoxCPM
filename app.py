import os
import re
import sys
import json
import html
import logging
import random
import urllib.error
import urllib.request
import numpy as np
import gradio as gr
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
import presets as preset_store
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_ZH = (
    "**普通话转方言语音生成**\n\n"
    "输入普通话文本，选择目标方言，并写出角色画像或声音描述。系统会先把普通话改写成对应方言口语，"
    "再用这个方言文本生成语音。你也可以上传参考音频，让生成结果更接近参考声音。\n\n"
    "**使用建议**  \n"
    "角色画像越具体，结果越容易稳定。例如：广东中年男教练，暴躁，语速快，声音粗粝；"
    "或四川话女店主，热情，接地气，像本地小店老板。\n\n"
    "**极致克隆模式**  \n"
    "开启后会根据参考音频文本续写，适合追求参考音频的细节还原；该模式会停用角色画像/声音描述。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**声音描述示例**  \n\n"
    "**示例 1：深宫太后**  \n"
    '`声音/方言描述`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`要生成的文字`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2：暴躁驾校教练**  \n"
    '`声音/方言描述`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`要生成的文字`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**方言生成指南**  \n"
    "要生成地道的方言语音，请在 **要生成的文字** 中直接使用方言词汇和句式，"
    "并在 **声音/方言描述** 中写明方言和声音风格。  \n\n"
    "**示例：广东话**  \n"
    '`声音/方言描述`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例：河南话**  \n"
    '`声音/方言描述`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "**小技巧：** 不知道方言怎么写时，可以先用 AI 助手把普通话改写成方言文本，"
    "再粘贴到“要生成的文字”里。  \n\n"
)

_ZH_TRANSLATIONS = {
    "reference_audio_label": "参考音频（可选，上传后用于克隆）",
    "show_prompt_text_label": "极致克隆模式（基于参考音频文本续写）",
    "show_prompt_text_info": "自动识别参考音频文本，尽量还原音色、节奏、情感等声音细节。开启后会停用声音/方言描述。",
    "prompt_text_label": "参考音频内容文本（自动识别，可手动编辑）",
    "prompt_text_placeholder": "参考音频的文字内容会自动识别并显示在这里。",
    "dialect_label": "目标方言",
    "role_label": "角色画像/声音描述",
    "role_placeholder": "例如：暴躁的广东中年男教练，语速快，声音粗粝，充满无奈和愤怒",
    "target_text_label": "普通话文本",
    "rewritten_text_label": "已改写的方言文本",
    "rewritten_text_placeholder": "先点击“先改写”，这里会显示方言文本；你也可以手动调整后再生成语音。",
    "rewrite_btn": "先改写",
    "generate_btn": "生成语音",
    "generated_audio_label": "生成结果",
    "advanced_settings_title": "高级设置",
    "auto_rewrite_label": "生成时自动补改写",
    "auto_rewrite_info": "如果没有先填写“已改写的方言文本”，生成语音时会自动补一次方言改写。",
    "ref_denoise_label": "参考音频降噪增强",
    "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理。",
    "normalize_label": "文本规范化",
    "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）。",
    "cfg_label": "CFG 引导强度",
    "cfg_info": "数值越高越贴合提示/参考音色；数值越低生成风格更自由。",
    "dit_steps_label": "生成迭代步数",
    "dit_steps_info": "步数越多可能音质更好，但生成速度会变慢。",
    "seed_label": "随机种子",
    "seed_info": "用于复现生成结果；生成成功后会显示实际使用的种子。",
    "random_seed_label": "每次自动随机",
    "random_seed_info": "每次生成前自动换一个随机种子。",
    "preset_section_title": "预设配置",
    "preset_name_label": "预设名称",
    "preset_name_placeholder": "例如：粤语暴躁教练 / 四川女店主",
    "save_preset_btn": "保存",
    "load_preset_label": "选择预设",
    "apply_preset_btn": "应用",
    "delete_preset_btn": "删除",
    "refresh_preset_btn": "刷新",
    "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
    "examples_footer": _EXAMPLES_FOOTER_ZH,
}

_I18N_TRANSLATIONS = {
    "en": _ZH_TRANSLATIONS,
    "zh-CN": _ZH_TRANSLATIONS,
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

_PRESET_TOASTS = {
    "name_empty": "请先输入预设名称。",
    "save_done": "预设已保存。",
    "save_overwritten": "同名预设已覆盖。",
    "save_failed": "保存预设失败：{error}",
    "select_first": "请先选择一个预设。",
    "missing": "预设不存在或已被删除。",
    "delete_done": "预设已删除。",
    "reference_missing": "预设中的参考音频文件不存在，已跳过音频。",
}

DEFAULT_TARGET_TEXT = "踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"

DEFAULT_DIALECTS = [
    "粤语",
    "四川话",
    "河南话",
    "东北话",
    "陕西话",
    "山东话",
    "天津话",
    "闽南话",
    "吴语",
    "普通话",
]

KRALAPI_BASE_URL = os.getenv("KRALAPI_BASE_URL", "https://kralapi.kralai.tech").rstrip("/")
KRALAPI_MODEL = os.getenv("KRALAPI_MODEL", "gpt-5.5")
KRALAPI_API_KEY = os.getenv("KRALAPI_API_KEY") or os.getenv("OPENAI_API_KEY")
REQUIRED_MODEL_FILES = (
    "config.json",
    "audiovae.pth",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_voxcpm2.py",
)

_CUSTOM_CSS = """
.brand-header {
    text-align: center;
    margin: 0.75rem 0 1rem 0;
}
.brand-name {
    display: inline-block;
    font-size: clamp(2rem, 4vw, 3.25rem);
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: 0;
    color: var(--body-text-color);
}
.brand-subtitle {
    margin-top: 0.5rem;
    color: var(--body-text-color-subdued);
    font-size: 0.95rem;
}
.service-readiness {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    background: var(--block-background-fill);
    padding: 14px 16px;
    margin: 0 0 1rem 0;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.service-readiness__header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}
.service-readiness__title {
    font-weight: 700;
    color: var(--body-text-color);
}
.service-readiness__summary {
    color: var(--body-text-color-subdued);
    font-size: 0.86rem;
}
.service-readiness__grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 10px;
}
.service-status {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    background: var(--background-fill-primary);
    padding: 10px 12px;
    min-width: 0;
}
.service-status__line {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.92rem;
    font-weight: 650;
}
.service-status__dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    flex: 0 0 auto;
}
.service-status--ok .service-status__dot { background: #16a34a; }
.service-status--warn .service-status__dot { background: #d97706; }
.service-status--error .service-status__dot { background: #dc2626; }
.service-status__detail {
    margin-top: 6px;
    color: var(--body-text-color-subdued);
    font-size: 0.82rem;
    line-height: 1.45;
    word-break: break-word;
}
.generation-alert {
    border: 1px solid #fecaca;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    background: #fef2f2;
    color: #7f1d1d;
    padding: 10px 12px;
    margin-top: 10px;
    line-height: 1.5;
}
.generation-alert__title {
    font-weight: 700;
    margin-bottom: 4px;
}
.workflow-alert {
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    background: #eff6ff;
    color: #1e3a8a;
    padding: 10px 12px;
    margin-top: 10px;
    line-height: 1.5;
}
.workflow-alert--success {
    border-color: #bbf7d0;
    border-left-color: #16a34a;
    background: #f0fdf4;
    color: #14532d;
}
.workflow-alert__title {
    font-weight: 700;
    margin-bottom: 4px;
}
@media (max-width: 900px) {
    .service-readiness__grid {
        grid-template-columns: 1fr;
    }
    .service-readiness__header {
        align-items: flex-start;
        flex-direction: column;
    }
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Dialect rewrite ----------


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


def _build_generation_error_html(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    return (
        '<div class="generation-alert">'
        '<div class="generation-alert__title">生成失败</div>'
        f"<div>{_escape_html(message)}</div>"
        "</div>"
    )


def _build_workflow_message_html(title: str, message: str, kind: str = "info") -> str:
    class_name = "workflow-alert workflow-alert--success" if kind == "success" else "workflow-alert"
    return (
        f'<div class="{class_name}">'
        f'<div class="workflow-alert__title">{_escape_html(title)}</div>'
        f"<div>{_escape_html(message)}</div>"
        "</div>"
    )


def _check_kralapi_connectivity() -> Tuple[str, str]:
    if not KRALAPI_API_KEY:
        return "error", "未配置 KRALAPI_API_KEY，自动方言改写不可用。"

    request = urllib.request.Request(
        f"{KRALAPI_BASE_URL}/v1/models",
        headers={"Authorization": f"Bearer {KRALAPI_API_KEY}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            if 200 <= response.status < 300:
                return "ok", f"已连通 {KRALAPI_BASE_URL}，改写模型：{KRALAPI_MODEL}。"
            return "warn", f"接口返回 HTTP {response.status}，生成时可能需要进一步确认。"
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return "error", f"接口可达，但密钥认证失败：HTTP {exc.code}。"
        return "warn", f"接口可达但返回 HTTP {exc.code}，生成时可能失败。"
    except urllib.error.URLError as exc:
        return "error", f"无法连通 {KRALAPI_BASE_URL}：{exc.reason}"
    except TimeoutError:
        return "error", f"连接 {KRALAPI_BASE_URL} 超时。"


def _get_data_root_path() -> Optional[Path]:
    data_root_value = (os.getenv("DATA_ROOT") or "").strip()
    if not data_root_value:
        return None
    data_root = data_root_value.rstrip("/") or "/"
    return Path(data_root).expanduser()


def _check_data_root_mount() -> Tuple[str, str]:
    data_root = _get_data_root_path()
    if data_root is None:
        return "warn", "未设置 DATA_ROOT；远端数据盘挂载点未知。8808 demo 推荐指向云盘/数据盘，例如 /root/autodl-tmp。"
    if not data_root.exists():
        return "error", f"DATA_ROOT 挂载目录不存在：{data_root}。请先挂载云盘/数据盘，或修正 DATA_ROOT。"
    if not data_root.is_dir():
        return "error", f"DATA_ROOT 存在但不是目录：{data_root}。请改成云盘/数据盘挂载目录。"
    return "ok", f"DATA_ROOT 挂载目录可访问：{data_root}"


def _check_data_root_subdirs() -> Tuple[str, str]:
    data_root = _get_data_root_path()
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


def _check_model_path(model_id: str) -> Tuple[str, str]:
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
        data_root = _get_data_root_path()
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


def build_service_readiness_html(model_id: str) -> str:
    api_key_status = "ok" if KRALAPI_API_KEY else "error"
    api_key_detail = (
        "已读取 KRALAPI_API_KEY，页面可调用方言改写。"
        if KRALAPI_API_KEY
        else "未读取 KRALAPI_API_KEY；如开启自动方言改写，生成会失败。"
    )
    rewrite_status, rewrite_detail = _check_kralapi_connectivity()
    data_root_status, data_root_detail = _check_data_root_mount()
    subdirs_status, subdirs_detail = _check_data_root_subdirs()
    model_status, model_detail = _check_model_path(model_id)
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


def _clean_llm_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:text|markdown|json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r"^(改写结果|方言文本|输出|结果)[:：]\s*", "", cleaned).strip()
    return cleaned.strip("\"'“”‘’ \n\t")


def rewrite_mandarin_to_dialect(text: str, dialect: str, role_description: str) -> str:
    source_text = (text or "").strip()
    target_dialect = (dialect or "粤语").strip()
    role = (role_description or "").strip()

    if not source_text:
        raise ValueError("请先输入普通话文本。")
    if target_dialect == "普通话":
        return source_text
    if not KRALAPI_API_KEY:
        raise ValueError("未配置 KRALAPI_API_KEY，无法自动改写方言文本。")

    system_prompt = (
        "你是中文方言口语改写助手，专门把普通话台词改写成指定方言口语。"
        "只输出改写后的台词，不要解释，不要加标题，不要加括号说明。"
        "保留原句的情绪、语气、人物关系、信息量和可朗读性。"
        "如果用户给了角色画像，要让遣词、节奏和口气符合角色。"
        "不要把所有字硬翻成生僻字，优先使用普通用户看得懂、TTS 容易读的方言口语写法。"
    )
    user_prompt = (
        f"目标方言：{target_dialect}\n"
        f"角色画像/声音描述：{role or '自然口语'}\n"
        f"普通话文本：{source_text}\n\n"
        "请输出适合直接交给中文 TTS 朗读的方言文本。"
    )
    payload = {
        "model": KRALAPI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.45,
        "max_tokens": min(1200, max(240, len(source_text) * 3)),
    }
    request = urllib.request.Request(
        f"{KRALAPI_BASE_URL}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {KRALAPI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"方言改写接口返回错误：HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"方言改写接口连接失败：{exc}") from exc

    rewritten = _clean_llm_text(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
    if not rewritten:
        raise RuntimeError("方言改写接口没有返回有效文本。")
    return rewritten


# ---------- Model ----------


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        wav = current_model.generate(**generate_kwargs)
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _rewrite_text_only(
        text: str,
        dialect: str,
        role_description: str,
        use_prompt_text: bool,
    ):
        try:
            input_text = (text or "").strip()
            if use_prompt_text:
                return (
                    gr.update(value=input_text, interactive=True),
                    gr.update(
                        value=_build_workflow_message_html(
                            "极致克隆模式已开启",
                            "该模式保持原有续写逻辑，不使用声音/方言描述；如需方言文本，可关闭极致克隆后再改写。",
                        ),
                        visible=True,
                    ),
                )
            rewritten_text = rewrite_mandarin_to_dialect(input_text, dialect, role_description)
            return (
                gr.update(value=rewritten_text, interactive=True),
                gr.update(
                    value=_build_workflow_message_html(
                        "方言文本已生成",
                        "你可以直接生成语音，也可以先手动微调右侧文本。",
                        kind="success",
                    ),
                    visible=True,
                ),
            )
        except Exception as exc:
            logger.exception("Dialect rewrite failed")
            return (
                gr.update(),
                gr.update(value=_build_generation_error_html(exc), visible=True),
            )

    def _generate(
        text: str,
        rewritten_text_value: str,
        dialect: str,
        role_description: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        auto_rewrite: bool,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
    ):
        try:
            actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
            input_text = (text or "").strip()
            edited_rewritten_text = (rewritten_text_value or "").strip()
            if use_prompt_text:
                synthesis_text = input_text
                display_rewritten_text = edited_rewritten_text
            elif edited_rewritten_text:
                synthesis_text = edited_rewritten_text
                display_rewritten_text = edited_rewritten_text
            elif auto_rewrite:
                synthesis_text = rewrite_mandarin_to_dialect(input_text, dialect, role_description)
                display_rewritten_text = synthesis_text
            else:
                synthesis_text = input_text
                display_rewritten_text = edited_rewritten_text
            control_parts = [dialect.strip() if dialect else "", role_description.strip() if role_description else ""]
            actual_control = "" if use_prompt_text else "，".join(part for part in control_parts if part)
            seed = _coerce_seed(seed_value)
            sr, wav_np, last_successful_seed = demo.generate_tts_audio(
                text_input=synthesis_text,
                control_instruction=actual_control,
                reference_wav_path_input=ref_wav,
                prompt_text=actual_prompt_text,
                cfg_value_input=cfg_value,
                do_normalize=do_normalize,
                denoise=denoise,
                inference_timesteps=int(dit_steps),
                seed=seed,
            )
            return display_rewritten_text, (sr, wav_np), last_successful_seed, gr.update(value="", visible=False)
        except Exception as exc:
            logger.exception("TTS generation failed")
            return (
                gr.update(),
                None,
                seed_value,
                gr.update(value=_build_generation_error_html(exc), visible=True),
            )

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    _PRESET_APPLY_OUTPUT_COUNT = 15

    def _collect_preset_data(
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
    ) -> dict:
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
            "seed": _coerce_seed(seed_value),
            "random_seed": bool(random_seed_value),
        }

    def _refresh_preset_dropdown(selected: str = ""):
        choices = preset_store.list_presets()
        value = selected if selected in choices else None
        return gr.update(choices=choices, value=value, interactive=bool(choices))

    def _empty_preset_apply_updates():
        return tuple(gr.update() for _ in range(_PRESET_APPLY_OUTPUT_COUNT))

    def _on_preset_save(
        name,
        ref_wav,
        dialect_value,
        role_value,
        text_value,
        rewritten_value,
        use_prompt_text,
        prompt_text_value,
        auto_rewrite,
        cfg_value_input,
        do_normalize,
        denoise,
        dit_steps_value,
        seed_value,
        random_seed_value,
    ):
        name = (name or "").strip()
        if not name:
            gr.Warning(_PRESET_TOASTS["name_empty"])
            return gr.update(), gr.update()

        safe_name = preset_store.safe_preset_name(name)
        existed = preset_store.preset_exists(safe_name)
        data = _collect_preset_data(
            dialect_value,
            role_value,
            text_value,
            rewritten_value,
            use_prompt_text,
            prompt_text_value,
            auto_rewrite,
            cfg_value_input,
            do_normalize,
            denoise,
            dit_steps_value,
            seed_value,
            random_seed_value,
        )
        try:
            saved_name = preset_store.save_preset(name, data, reference_audio=ref_wav)
        except Exception as exc:
            logger.exception("Preset save failed")
            gr.Warning(_PRESET_TOASTS["save_failed"].format(error=exc))
            return gr.update(), gr.update()

        gr.Info(_PRESET_TOASTS["save_overwritten" if existed else "save_done"])
        return gr.update(value=""), _refresh_preset_dropdown(saved_name)

    def _on_preset_apply(name):
        if not name:
            gr.Warning(_PRESET_TOASTS["select_first"])
            return _empty_preset_apply_updates()

        data = preset_store.load_preset(name)
        if data is None:
            gr.Warning(_PRESET_TOASTS["missing"])
            return _empty_preset_apply_updates()

        ref_path = data.get("reference_audio") or None
        if ref_path and not os.path.exists(ref_path):
            gr.Warning(_PRESET_TOASTS["reference_missing"])
            ref_path = None

        use_prompt = bool(data.get("use_prompt_text", False))
        random_seed_value = bool(data.get("random_seed", True))
        return (
            gr.update(value=ref_path),
            gr.update(value=data.get("dialect", "粤语")),
            gr.update(value=data.get("role_description", ""), visible=not use_prompt),
            gr.update(value=data.get("mandarin_text", "")),
            gr.update(value=data.get("rewritten_text", ""), interactive=True),
            gr.update(value=use_prompt),
            gr.update(value=data.get("prompt_text", ""), visible=use_prompt),
            gr.update(value=data.get("auto_rewrite", True)),
            gr.update(value=data.get("cfg_value", 2.0)),
            gr.update(value=data.get("do_normalize", False)),
            gr.update(value=data.get("denoise", False)),
            gr.update(value=data.get("dit_steps", 10)),
            gr.update(value=data.get("seed"), interactive=not random_seed_value),
            gr.update(value=random_seed_value),
            gr.update(value="", visible=False),
        )

    def _on_preset_delete(name):
        if not name:
            gr.Warning(_PRESET_TOASTS["select_first"])
            return gr.update()
        if preset_store.delete_preset(name):
            gr.Info(_PRESET_TOASTS["delete_done"])
        else:
            gr.Warning(_PRESET_TOASTS["missing"])
        return _refresh_preset_dropdown()

    def _on_preset_refresh():
        return _refresh_preset_dropdown()

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="brand-header">'
            '<div class="brand-name">声创科技</div>'
            '<div class="brand-subtitle">中文方言与声音克隆生成工具</div>'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))
        gr.HTML(build_service_readiness_html(demo._model_id))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                dialect = gr.Dropdown(
                    choices=DEFAULT_DIALECTS,
                    value="粤语",
                    label=I18N("dialect_label"),
                    allow_custom_value=True,
                )
                role_description = gr.Textbox(
                    value="暴躁的广东中年男教练，语速快，声音粗粝，充满无奈和愤怒",
                    label=I18N("role_label"),
                    placeholder=I18N("role_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    AutoRewriteText = gr.Checkbox(
                        value=True,
                        label=I18N("auto_rewrite_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("auto_rewrite_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    with gr.Row():
                        seed_value = gr.Number(
                            value=random.randint(0, 2**32 - 1),
                            precision=0,
                            label=I18N("seed_label"),
                            info=I18N("seed_info"),
                            interactive=False,
                        )
                        random_seed = gr.Checkbox(
                            value=True,
                            label=I18N("random_seed_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("random_seed_info"),
                        )

                with gr.Accordion(I18N("preset_section_title"), open=False):
                    with gr.Row():
                        preset_name = gr.Textbox(
                            value="",
                            label=I18N("preset_name_label"),
                            placeholder=I18N("preset_name_placeholder"),
                            scale=3,
                        )
                        save_preset_btn = gr.Button(I18N("save_preset_btn"), scale=1)
                    with gr.Row():
                        preset_choices = preset_store.list_presets()
                        preset_dropdown = gr.Dropdown(
                            choices=preset_choices,
                            value=None,
                            label=I18N("load_preset_label"),
                            allow_custom_value=False,
                            interactive=bool(preset_choices),
                            scale=3,
                        )
                        apply_preset_btn = gr.Button(I18N("apply_preset_btn"), scale=1)
                        delete_preset_btn = gr.Button(I18N("delete_preset_btn"), scale=1)
                        refresh_preset_btn = gr.Button(I18N("refresh_preset_btn"), scale=1)

                with gr.Row():
                    rewrite_btn = gr.Button(I18N("rewrite_btn"), variant="secondary")
                    run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")
                generation_error = gr.HTML(value="", visible=False)

            with gr.Column():
                rewritten_text_output = gr.Textbox(
                    value="",
                    label=I18N("rewritten_text_label"),
                    placeholder=I18N("rewritten_text_placeholder"),
                    lines=5,
                    interactive=True,
                )
                audio_output = gr.Audio(label=I18N("generated_audio_label"))
                gr.Markdown(I18N("examples_footer"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, role_description],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        rewrite_btn.click(
            fn=_rewrite_text_only,
            inputs=[
                text,
                dialect,
                role_description,
                show_prompt_text,
            ],
            outputs=[rewritten_text_output, generation_error],
            show_progress=True,
        )

        run_btn.click(
            fn=_prepare_seed,
            inputs=[random_seed, seed_value],
            outputs=[seed_value],
            show_progress=False,
        ).then(
            fn=_generate,
            inputs=[
                text,
                rewritten_text_output,
                dialect,
                role_description,
                reference_wav,
                show_prompt_text,
                prompt_text,
                AutoRewriteText,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                seed_value,
            ],
            outputs=[rewritten_text_output, audio_output, seed_value, generation_error],
            show_progress=True,
            api_name="generate",
        )

        preset_save_inputs = [
            preset_name,
            reference_wav,
            dialect,
            role_description,
            text,
            rewritten_text_output,
            show_prompt_text,
            prompt_text,
            AutoRewriteText,
            cfg_value,
            DoNormalizeText,
            DoDenoisePromptAudio,
            dit_steps,
            seed_value,
            random_seed,
        ]
        preset_apply_outputs = [
            reference_wav,
            dialect,
            role_description,
            text,
            rewritten_text_output,
            show_prompt_text,
            prompt_text,
            AutoRewriteText,
            cfg_value,
            DoNormalizeText,
            DoDenoisePromptAudio,
            dit_steps,
            seed_value,
            random_seed,
            generation_error,
        ]
        save_preset_btn.click(
            fn=_on_preset_save,
            inputs=preset_save_inputs,
            outputs=[preset_name, preset_dropdown],
        )
        apply_preset_btn.click(
            fn=_on_preset_apply,
            inputs=[preset_dropdown],
            outputs=preset_apply_outputs,
        )
        delete_preset_btn.click(
            fn=_on_preset_delete,
            inputs=[preset_dropdown],
            outputs=[preset_dropdown],
        )
        refresh_preset_btn.click(
            fn=_on_preset_refresh,
            inputs=[],
            outputs=[preset_dropdown],
        )
        interface.load(
            fn=_on_preset_refresh,
            inputs=[],
            outputs=[preset_dropdown],
        )

    return interface


def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "the default 0.0.0.0 exposes the unauthenticated UI/API to the network (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
