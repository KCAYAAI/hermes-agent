"""security.redact_phones: phone masking is a separate switch from secret redaction."""
import importlib

import pytest


PHONE = "+18165550123"
TOKEN = "xo" + "xb-" + "123456789012-123456789012-abcdefghijklmnopqrstuvwx"  # built at runtime: keeps push protection quiet


def _reload(monkeypatch, phones):
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    if phones is None:
        monkeypatch.delenv("HERMES_REDACT_PHONES", raising=False)
    else:
        monkeypatch.setenv("HERMES_REDACT_PHONES", phones)
    import agent.redact as r
    return importlib.reload(r)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("HERMES_REDACT_PHONES", raising=False)
    import agent.redact as r
    importlib.reload(r)


def test_default_still_masks_phone(monkeypatch):
    r = _reload(monkeypatch, None)
    assert PHONE not in r.redact_sensitive_text(f"call {PHONE}", force=True)


def test_off_keeps_complete_phone_even_on_force_paths(monkeypatch):
    r = _reload(monkeypatch, "false")
    out = r.redact_sensitive_text(f"call {PHONE} now", force=True)
    assert PHONE in out
    assert PHONE in r.redact_terminal_output(f"phone={PHONE}", "echo", force=True)


def test_off_does_not_weaken_secret_redaction(monkeypatch):
    r = _reload(monkeypatch, "false")
    out = r.redact_sensitive_text(f"{PHONE} {TOKEN}", force=True)
    assert PHONE in out
    assert TOKEN not in out


def test_routed_profile_uses_its_own_redact_phones(tmp_path, monkeypatch):
    """Multiplexed gateway: each served profile reads its own security.redact_phones."""
    from pathlib import Path
    import hermes_yaml as yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    r = _reload(monkeypatch, None)
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    (a / "config.yaml").write_text(yaml.safe_dump({"security": {"redact_phones": True}}), encoding="utf-8")
    (b / "config.yaml").write_text(yaml.safe_dump({"security": {"redact_phones": False}}), encoding="utf-8")
    r._REDACT_PHONES_BY_HOME.clear()

    def under(home):
        tok = set_hermes_home_override(str(home))
        try:
            return r.redact_sensitive_text(f"call {PHONE}", force=True)
        finally:
            reset_hermes_home_override(tok)

    assert PHONE in under(b)
    assert PHONE not in under(a)
