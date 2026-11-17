"""skills.usage_tracking and skills.verify_interval_seconds read through a real config.yaml."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import agent.skill_utils as su
    su._raw_config_cache_clear()
    import tools.skill_usage as mod
    importlib.reload(mod)
    return home


def _config(home, body):
    (home / "config.yaml").write_text(body, encoding="utf-8")


def test_tracking_on_by_default_records_views(home):
    from tools.skill_usage import bump_view, bump_use, get_record
    bump_view("s")
    bump_use("s")
    rec = get_record("s")
    assert rec["view_count"] == 1 and rec["use_count"] == 1


def test_tracking_off_writes_nothing_for_views_and_uses(home):
    _config(home, "skills:\n  usage_tracking: false\n")
    from tools.skill_usage import bump_view, bump_use
    bump_view("s")
    bump_use("s")
    assert not (home / "skills" / ".usage.json").exists()


def test_tracking_off_still_records_patches(home):
    _config(home, "skills:\n  usage_tracking: false\n")
    from tools.skill_usage import bump_patch, get_record
    bump_patch("s")
    assert get_record("s")["patch_count"] == 1


def test_unreadable_tracking_setting_never_fails_a_skill_load(home, monkeypatch):
    import agent.skill_utils as su
    from tools.skill_usage import bump_view, get_record

    def broken():
        raise RuntimeError("config backend unavailable")
    monkeypatch.setattr(su, "skills_usage_tracking_enabled", broken)
    bump_view("s")  # no exception reaches skill_view
    assert get_record("s")["view_count"] == 1  # default stays on


def test_verify_interval_read_from_config(home):
    from agent.skill_utils import skills_verify_interval
    assert skills_verify_interval() == 0.0
    _config(home, "skills:\n  verify_interval_seconds: 5\n")
    assert skills_verify_interval() == 5.0


def test_curator_refuses_while_tracking_off_even_without_curator_key(home):
    from agent import curator
    assert curator.is_enabled() is True  # no config: upstream default stays on
    _config(home, "skills:\n  usage_tracking: false\n")
    assert curator.is_enabled() is False
    _config(home, "skills:\n  usage_tracking: true\ncurator:\n  enabled: false\n")
    assert curator.is_enabled() is False


def test_defaults_registered():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["skills"]["verify_interval_seconds"] == 0
    assert DEFAULT_CONFIG["skills"]["usage_tracking"] is True
