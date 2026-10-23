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


def _two_homes(tmp_path, monkeypatch):
    from pathlib import Path
    import hermes_yaml as yaml
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    (a / "config.yaml").write_text(yaml.safe_dump({"security": {"redact_phones": True}}), encoding="utf-8")
    (b / "config.yaml").write_text(yaml.safe_dump({"security": {"redact_phones": False}}), encoding="utf-8")
    return a, b


def _under(home, fn):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    tok = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(tok)


def test_routed_profile_uses_its_own_redact_phones(tmp_path, monkeypatch):
    """Multiplexed gateway: each served profile follows its own security.redact_phones once that
    profile's .env and config are loaded (the normal profile path), like upstream's routed
    redact_secrets. Before its .env is recorded, a scope-less read masks (secure default)."""
    from hermes_cli.config import load_config_readonly, load_env

    r = _reload(monkeypatch, None)
    a, b = _two_homes(tmp_path, monkeypatch)
    _under(b, load_config_readonly)
    assert PHONE not in _under(b, lambda: r.redact_sensitive_text(f"call {PHONE}", force=True))
    for home in (a, b):
        _under(home, load_env)
        _under(home, load_config_readonly)

    def red():
        return r.redact_sensitive_text(f"call {PHONE}", force=True)

    assert PHONE in _under(b, red)
    assert PHONE not in _under(a, red)


def test_routed_profile_policy_from_log_routing_setup(tmp_path, monkeypatch):
    """resolve_profile_policy (log-routing setup) records the phone switch too."""
    r = _reload(monkeypatch, None)
    a, b = _two_homes(tmp_path, monkeypatch)
    r.resolve_profile_policy(str(b))
    r.resolve_profile_policy(str(a))
    red = lambda: r.redact_sensitive_text(f"call {PHONE}", force=True)  # noqa: E731
    assert PHONE in _under(b, red)
    assert PHONE not in _under(a, red)


def test_routed_phone_switch_reads_no_config(tmp_path, monkeypatch):
    """The formatter path never reads config or .env: an unrecorded routed profile masks (the
    upstream default) without touching the loaders, and never raises."""
    import hermes_cli.config as cfg

    r = _reload(monkeypatch, None)
    _, b = _two_homes(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("redactor read config")

    monkeypatch.setattr(cfg, "load_config_readonly", boom)
    monkeypatch.setattr(cfg, "load_env", boom)
    out = _under(b, lambda: r.redact_sensitive_text(f"call {PHONE}", force=True))
    assert PHONE not in out


def test_routed_turn_with_secret_scope_follows_config(tmp_path, monkeypatch):
    """A multiplex turn runs under the profile's secret scope (no HERMES_REDACT_PHONES in .env):
    the recorded config value decides, and an env value in the scope overrides it."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli.config import load_config_readonly

    r = _reload(monkeypatch, None)
    a, b = _two_homes(tmp_path, monkeypatch)
    for home in (a, b):
        _under(home, load_config_readonly)

    def red_in_scope(secrets, home):
        def fn():
            tok = set_secret_scope(secrets, profile_home=str(home))
            try:
                return r.redact_sensitive_text(f"call {PHONE}", force=True)
            finally:
                reset_secret_scope(tok)
        return _under(home, fn)

    assert PHONE in red_in_scope({"OTHER": "x"}, b)
    assert PHONE not in red_in_scope({"OTHER": "x"}, a)
    assert PHONE not in red_in_scope({"HERMES_REDACT_PHONES": "true"}, b)
