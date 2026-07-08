from pathlib import Path

import preset_controls
import readiness


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


def test_collect_preset_data_normalizes_ui_values():
    data = preset_controls.collect_preset_data(
        "",
        None,
        None,
        "好多钱？",
        1,
        None,
        0,
        None,
        "",
        "yes",
        None,
        "",
        "",
    )

    assert data == {
        "dialect": "粤语",
        "role_description": "",
        "mandarin_text": "",
        "rewritten_text": "好多钱？",
        "use_prompt_text": True,
        "prompt_text": "",
        "auto_rewrite": False,
        "cfg_value": 2.0,
        "do_normalize": False,
        "denoise": True,
        "dit_steps": 10,
        "seed": None,
        "random_seed": False,
    }


def test_preset_apply_updates_round_trip_and_skips_missing_reference(tmp_path):
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


def test_readiness_reports_missing_demo_prerequisites(tmp_path):
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


def test_readiness_reports_ready_without_loading_model(tmp_path):
    model_dir = tmp_path / "models" / "VoxCPM2"
    model_dir.mkdir(parents=True)
    (tmp_path / "cache").mkdir()
    (tmp_path / "presets").mkdir()
    for file_name in readiness.REQUIRED_MODEL_FILES:
        (model_dir / file_name).write_text("stub", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")

    html = readiness.build_service_readiness_html(
        str(model_dir),
        api_key="secret",
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path),
        urlopen=_ok_urlopen,
    )

    assert "服务就绪" in html
    assert "VoxCPM2 模型目录、关键文件和主权重已就绪" in html
    assert "预设存储目录可写" in html
