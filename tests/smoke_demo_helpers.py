"""Stdlib smoke checks for dependency-light dialect demo helpers.

Run with:
    python3 tests/smoke_demo_helpers.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import preset_controls  # noqa: E402
import readiness  # noqa: E402


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ok_urlopen(request, timeout):
    assert request.full_url == "https://example.test/v1/models"
    assert timeout == 8
    return _Response()


def smoke_preset_control_state_round_trip(tmp_path: Path) -> None:
    ref_audio = tmp_path / "voice.wav"
    ref_audio.write_bytes(b"fake wav")
    data = preset_controls.collect_preset_data(
        "四川话",
        "热情店主",
        "多少钱？",
        "好多钱？",
        True,
        "参考音频文字",
        False,
        2.4,
        True,
        False,
        18,
        "123",
        False,
    )
    data["reference_audio"] = str(ref_audio)

    updates, reference_missing = preset_controls.build_preset_apply_update_kwargs(data)

    assert reference_missing is False
    assert len(updates) == preset_controls.PRESET_APPLY_OUTPUT_COUNT
    assert updates[0] == {"value": str(ref_audio)}
    assert updates[2] == {"value": "热情店主", "visible": False}
    assert updates[6] == {"value": "参考音频文字", "visible": True}
    assert updates[12] == {"value": 123, "interactive": True}
    assert updates[14] == {"value": "", "visible": False}

    data["reference_audio"] = str(tmp_path / "missing.wav")
    missing_updates, reference_missing = preset_controls.build_preset_apply_update_kwargs(data)
    assert reference_missing is True
    assert missing_updates[0] == {"value": None}


def smoke_readiness_reports_missing_and_ready_paths(tmp_path: Path) -> None:
    html = readiness.build_service_readiness_html(
        str(tmp_path / "models" / "VoxCPM2"),
        api_key=None,
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path),
        urlopen=_ok_urlopen,
    )
    assert "未读取 KRALAPI_API_KEY" in html
    assert "缺少 8808 demo 子目录" in html
    assert "模型目录缺失" in html

    (tmp_path / "models" / "VoxCPM2").mkdir(parents=True)
    (tmp_path / "cache").mkdir()
    for file_name in readiness.REQUIRED_MODEL_FILES:
        (tmp_path / "models" / "VoxCPM2" / file_name).write_text("stub", encoding="utf-8")
    (tmp_path / "models" / "VoxCPM2" / "model.safetensors").write_text("stub", encoding="utf-8")

    html = readiness.build_service_readiness_html(
        str(tmp_path / "models" / "VoxCPM2"),
        api_key="secret",
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path),
        urlopen=_ok_urlopen,
    )
    assert "服务就绪" in html
    assert "VoxCPM2 模型目录和关键文件已就绪" in html


def main() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        smoke_preset_control_state_round_trip(Path(raw_tmp))
    with tempfile.TemporaryDirectory() as raw_tmp:
        smoke_readiness_reports_missing_and_ready_paths(Path(raw_tmp))
    print("demo helper smoke tests passed")


if __name__ == "__main__":
    main()
