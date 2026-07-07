from pathlib import Path

import presets


def test_save_load_and_list_preset_with_reference_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRESETS_DIR", str(tmp_path / "presets"))
    reference = tmp_path / "voice.wav"
    reference.write_bytes(b"fake wav")

    saved_name = presets.save_preset(
        "粤语 暴躁/教练",
        {
            "dialect": "粤语",
            "role_description": "暴躁的广东中年男教练",
            "mandarin_text": "踩刹车。",
            "rewritten_text": "踩刹车啊！",
            "use_prompt_text": False,
            "prompt_text": "",
            "auto_rewrite": True,
            "cfg_value": 2.1,
            "do_normalize": False,
            "denoise": True,
            "dit_steps": 12,
            "seed": 42,
            "random_seed": False,
        },
        reference_audio=str(reference),
    )

    assert saved_name == "粤语_暴躁_教练"
    assert presets.list_presets() == [saved_name]

    loaded = presets.load_preset(saved_name)
    assert loaded is not None
    assert loaded["dialect"] == "粤语"
    assert loaded["role_description"] == "暴躁的广东中年男教练"
    assert loaded["reference_audio"].endswith("reference_audio.wav")
    assert Path(loaded["reference_audio"]).read_bytes() == b"fake wav"


def test_overwrite_and_delete_remove_managed_files(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCPM_PRESETS_DIR", str(tmp_path / "presets"))
    first = tmp_path / "first.wav"
    second = tmp_path / "second.mp3"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    presets.save_preset("demo", {"dialect": "粤语"}, reference_audio=str(first))
    presets.save_preset("demo", {"dialect": "四川话"}, reference_audio=str(second))

    preset_dir = tmp_path / "presets" / "demo"
    assert not (preset_dir / "reference_audio.wav").exists()
    assert (preset_dir / "reference_audio.mp3").exists()
    assert presets.load_preset("demo")["dialect"] == "四川话"

    assert presets.delete_preset("demo") is True
    assert presets.list_presets() == []
    assert preset_dir.exists()
    assert not (preset_dir / "preset.json").exists()
    assert not (preset_dir / "reference_audio.mp3").exists()
