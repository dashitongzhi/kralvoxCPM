from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import readiness


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ok_urlopen(request, timeout):
    return _Response()


def _write_required_model_files(model_path: Path) -> None:
    model_path.mkdir(parents=True)
    for file_name in readiness.REQUIRED_MODEL_FILES:
        (model_path / file_name).write_text("stub", encoding="utf-8")


def test_service_readiness_requires_local_model_weight_file(tmp_path):
    model_path = tmp_path / "models" / "VoxCPM2"
    (tmp_path / "cache").mkdir()
    _write_required_model_files(model_path)

    html = readiness.build_service_readiness_html(
        str(model_path),
        api_key="secret",
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path),
        urlopen=_ok_urlopen,
    )

    assert "需要处理配置项" in html
    assert "model.safetensors 或其他 *.safetensors/*.bin 主权重文件" in html
    assert "服务就绪 ·" not in html


def test_service_readiness_accepts_alternate_bin_weight_file(tmp_path):
    model_path = tmp_path / "models" / "VoxCPM2"
    (tmp_path / "cache").mkdir()
    (tmp_path / "presets").mkdir()
    _write_required_model_files(model_path)
    (model_path / "pytorch_model.bin").write_text("stub", encoding="utf-8")

    html = readiness.build_service_readiness_html(
        str(model_path),
        api_key="secret",
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path),
        urlopen=_ok_urlopen,
    )

    assert "服务就绪" in html
    assert "VoxCPM2 模型目录、关键文件和主权重已就绪" in html
    assert "预设存储目录可写" in html


def test_service_readiness_reports_unwritable_preset_parent(tmp_path):
    model_path = tmp_path / "models" / "VoxCPM2"
    (tmp_path / "cache").mkdir()
    _write_required_model_files(model_path)
    (model_path / "model.safetensors").write_text("stub", encoding="utf-8")

    html = readiness.build_service_readiness_html(
        str(model_path),
        api_key="secret",
        base_url="https://example.test",
        rewrite_model="gpt-test",
        data_root_value=str(tmp_path / "missing-root"),
        urlopen=_ok_urlopen,
    )

    assert "预设存储目录不存在且父目录不可写" in html
