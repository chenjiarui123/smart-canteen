from pathlib import Path

from app.active_strategy import active_config, active_status


def test_strategy_json_is_read_with_explicit_utf8_on_windows(monkeypatch):
    """Simulate Windows failing whenever an UTF-8 strategy file uses GBK default."""
    original_read_text = Path.read_text

    def windows_read_text(path, *args, **kwargs):
        if path.name in {"targeted_config.json", "targeted_manifest.json"}:
            assert kwargs.get("encoding") == "utf-8-sig"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", windows_read_text)

    config = active_config()
    status = active_status()

    assert config["targeted_enabled"] is True
    assert status["strategy_modified"] is True
