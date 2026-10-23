"""KC /model + /reasoning access for Slack users (gateway/kc_model_access.py).

Every allowed Slack user may change the model and reasoning for their own conversation; only a
listed admin may change the agent for everyone (``--global``, reasoning display), and a non-admin
may only switch to a row of the platform's curated ``model_picker``.
"""

from types import SimpleNamespace

import pytest

from gateway import kc_model_access as access
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource

ADMIN = "UADMIN"
USER = "UUSER"

PICKER = [
    {"label": "Opus 5.5", "provider": "claude-subscription-directsdk-experimental",
     "model": "claude-opus-5-5[1m]"},
    {"label": "Sonnet 5.5", "provider": "claude-subscription-directsdk-experimental",
     "model": "claude-sonnet-5-5[1m]"},
    {"label": "GPT 6.1 Sol", "provider": "openai-codex", "model": "gpt-6.1-sol"},
    {"label": "GPT 6 Luna", "provider": "openai-codex", "model": "gpt-6-luna"},
    {"label": "GPT 6 Astra", "provider": "openai-codex", "model": "gpt-6-astra"},
]


def _extra(**over):
    extra = {
        "model_picker": PICKER,
        "allow_admin_from": [ADMIN],
        "group_allow_admin_from": [ADMIN],
        "user_allowed_commands": ["new", "queue", "steer", "stop", "model", "reasoning"],
        "group_user_allowed_commands": ["new", "queue", "steer", "stop", "model", "reasoning"],
    }
    extra.update(over)
    return {k: v for k, v in extra.items() if v is not None}


def _runner(extra=None, platform=Platform.SLACK):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={
        platform: PlatformConfig(enabled=True, token="***", extra=_extra() if extra is None else extra)})
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._session_db = None
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._evict_cached_agent = lambda _key: None
    runner.session_store = None
    return runner


def _source(user=USER, platform=Platform.SLACK, chat_type="group"):
    return SessionSource(platform=platform, chat_id="C1", chat_type=chat_type, user_id=user,
                         thread_id="1.2")


def _event(text, **kw):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source(**kw))


# --------------------------------------------------------------------------- module


def test_admin_is_only_an_explicitly_listed_user():
    runner = _runner()
    assert access.is_admin(runner, _source(ADMIN))
    assert not access.is_admin(runner, _source(USER))
    # Ungated profile (no admin list): nobody is an admin for agent-wide changes.
    ungated = _runner(_extra(allow_admin_from=None, group_allow_admin_from=None))
    assert not access.is_admin(ungated, _source(ADMIN))
    # model_admin_from names admins without turning on slash gating.
    listed = _runner(_extra(allow_admin_from=None, group_allow_admin_from=None, model_admin_from=[ADMIN]))
    assert access.is_admin(listed, _source(ADMIN))
    assert not access.is_admin(listed, _source(USER))


def test_restriction_is_slack_only():
    runner = _runner(platform=Platform.TELEGRAM)
    assert not access.restricted(runner, _source(USER, platform=Platform.TELEGRAM))
    assert access.restricted(_runner(), _source(USER))
    assert not access.restricted(_runner(), _source(ADMIN))


def test_curated_entries_and_labels():
    entries = access.curated_entries(_runner(), _source())
    assert [e["label"] for e in entries] == [p["label"] for p in PICKER]
    assert access.entry_for_label(entries, "  sonnet   5.5 ")["model"] == "claude-sonnet-5-5[1m]"
    assert access.entry_for_label(entries, "gpt-6.1-sol") is None
    assert access.is_curated_route(entries, "openai-codex", "gpt-6-luna")
    assert not access.is_curated_route(entries, "openai-codex", "gpt-6-luna-900k")
    assert access.curated_entries(_runner(_extra(model_picker=None)), _source()) == []


# --------------------------------------------------------------------------- /model


def _fake_switch(monkeypatch, tmp_path, *, new_model, provider, seen=None):
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  default: claude-opus-5-5[1m]\n  provider: claude-subscription-directsdk-experimental\n",
        encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    def switch(**kw):
        if seen is not None:
            seen.append(kw)
        return ModelSwitchResult(success=True, new_model=new_model, target_provider=provider,
                                 provider_changed=True, api_key="k", base_url="", api_mode="",
                                 provider_label=provider, is_global=kw.get("is_global", False))

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch)
    return home


@pytest.mark.asyncio
async def test_non_admin_global_model_is_refused(monkeypatch, tmp_path):
    seen = []
    _fake_switch(monkeypatch, tmp_path, new_model="gpt-6-luna", provider="openai-codex", seen=seen)
    reply = await _runner()._handle_model_command(_event("/model gpt-6-luna --global"))
    assert reply == access.GLOBAL_DENIED
    assert seen == []


@pytest.mark.asyncio
async def test_non_admin_switches_own_conversation_to_curated_model(monkeypatch, tmp_path):
    seen = []
    home = _fake_switch(monkeypatch, tmp_path, new_model="gpt-6-luna", provider="openai-codex", seen=seen)
    before = (home / "config.yaml").read_text()
    runner = _runner()
    reply = await runner._handle_model_command(_event("/model gpt-6-luna --provider openai-codex"))
    assert "⛔" not in (reply or "")
    assert seen and seen[0]["is_global"] is False
    assert any(v.get("model") == "gpt-6-luna" for v in runner._session_model_overrides.values())
    assert (home / "config.yaml").read_text() == before


@pytest.mark.asyncio
async def test_non_admin_can_pick_by_label(monkeypatch, tmp_path):
    seen = []
    _fake_switch(monkeypatch, tmp_path, new_model="claude-sonnet-5-5[1m]",
                 provider="claude-subscription-directsdk-experimental", seen=seen)
    reply = await _runner()._handle_model_command(_event("/model Sonnet 5.5"))
    assert "⛔" not in (reply or "")
    assert seen[0]["raw_input"] == "claude-sonnet-5-5[1m]"
    assert seen[0]["explicit_provider"] == "claude-subscription-directsdk-experimental"


@pytest.mark.asyncio
async def test_non_admin_off_list_model_is_refused_without_commit(monkeypatch, tmp_path):
    _fake_switch(monkeypatch, tmp_path, new_model="gpt-6-luna-900k", provider="openai-codex")
    runner = _runner()
    reply = await runner._handle_model_command(_event("/model gpt-6-luna-900k"))
    assert reply == access.not_curated_message(PICKER)
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_non_admin_bare_provider_is_refused(monkeypatch, tmp_path):
    seen = []
    _fake_switch(monkeypatch, tmp_path, new_model="x", provider="openrouter", seen=seen)
    reply = await _runner()._handle_model_command(_event("/model --provider openrouter"))
    assert reply == access.not_curated_message(PICKER)
    assert seen == []


@pytest.mark.asyncio
async def test_admin_keeps_global_and_off_list(monkeypatch, tmp_path):
    seen = []
    _fake_switch(monkeypatch, tmp_path, new_model="gpt-6-luna-900k", provider="openai-codex", seen=seen)
    reply = await _runner()._handle_model_command(_event("/model gpt-6-luna-900k --global", user=ADMIN))
    assert "⛔" not in (reply or "")
    assert seen[0]["is_global"] is True


@pytest.mark.asyncio
async def test_no_picker_configured_does_not_restrict_model_choice(monkeypatch, tmp_path):
    _fake_switch(monkeypatch, tmp_path, new_model="anything", provider="openrouter")
    runner = _runner(_extra(model_picker=None))
    reply = await runner._handle_model_command(_event("/model anything"))
    assert "⛔" not in (reply or "")


# --------------------------------------------------------------------------- /reasoning


@pytest.mark.asyncio
async def test_non_admin_reasoning_session_only(monkeypatch):
    runner = _runner()
    saved = []
    monkeypatch.setattr(runner, "_save_gateway_config_key", lambda *a: saved.append(a) or True)
    assert await runner._handle_reasoning_command(_event("/reasoning high --global")) == access.GLOBAL_DENIED
    for toggle in ("show", "hide", "on", "off"):
        assert await runner._handle_reasoning_command(_event(f"/reasoning {toggle}")) == access.DISPLAY_DENIED
    reply = await runner._handle_reasoning_command(_event("/reasoning high"))
    assert "⛔" not in reply
    assert saved == []
    assert any((v or {}).get("effort") == "high" for v in runner._session_reasoning_overrides.values())


@pytest.mark.asyncio
async def test_admin_reasoning_global_still_works(monkeypatch):
    runner = _runner()
    saved = []
    monkeypatch.setattr(runner, "_save_gateway_config_key", lambda *a: saved.append(a) or True)
    reply = await runner._handle_reasoning_command(_event("/reasoning high --global", user=ADMIN))
    assert "⛔" not in reply
    assert saved == [("agent.reasoning_effort", "high")]


@pytest.mark.asyncio
async def test_non_admin_reasoning_picker_hides_display_toggles(monkeypatch):
    runner = _runner()
    captured = {}

    async def picker(event, session_key, title, choices, on_choice_selected):
        captured["values"] = [c["value"] for c in choices]
        captured["cb"] = on_choice_selected
        return True

    monkeypatch.setattr(runner, "_try_send_choice_picker", picker)
    assert await runner._handle_reasoning_command(_event("/reasoning")) is None
    assert "show" not in captured["values"] and "hide" not in captured["values"]
    assert "reset" in captured["values"] and "high" in captured["values"]
    assert await captured["cb"]("C1", "show") == access.DISPLAY_DENIED

    assert await runner._handle_reasoning_command(_event("/reasoning", user=ADMIN)) is None
    assert "show" in captured["values"] and "hide" in captured["values"]


@pytest.mark.asyncio
async def test_telegram_unaffected(monkeypatch):
    runner = _runner(platform=Platform.TELEGRAM)
    saved = []
    monkeypatch.setattr(runner, "_save_gateway_config_key", lambda *a: saved.append(a) or True)
    reply = await runner._handle_reasoning_command(
        _event("/reasoning high --global", platform=Platform.TELEGRAM, chat_type="dm"))
    assert "⛔" not in reply
    assert saved == [("agent.reasoning_effort", "high")]


# ------------------------------------------------------------- usage-limit one-tap switch (KC gate)


def _usage_limit_runner(monkeypatch, runner):
    from gateway import slash_commands_model as scm
    from unittest.mock import AsyncMock
    runner._normalize_source_for_session_key = lambda source: source
    seen = {"perform": 0, "commit": 0, "tasks": []}

    async def _perform(ctx, target, provider, source):
        seen["perform"] += 1
        return SimpleNamespace(new_model=target, target_provider=provider), None

    async def _commit(result, ctx, *, source, picker):
        seen["commit"] += 1
        runner._session_model_overrides[ctx.session_key] = {"model": result.new_model, "provider": "nous"}
        return "✓ switched"

    runner._perform_model_switch = _perform
    runner._commit_model_switch = _commit
    runner._track_background_task = lambda coro: (seen["tasks"].append(coro), coro.close())
    monkeypatch.setattr(scm._ModelSwitchContext, "read_config", lambda self: None)
    monkeypatch.setattr(scm._ModelSwitchContext, "apply_override", lambda self, override: None)
    return seen


@pytest.mark.asyncio
async def test_non_admin_usage_limit_switch_off_list_is_refused(monkeypatch):
    runner = _runner()
    seen = _usage_limit_runner(monkeypatch, runner)
    reply = await runner._accept_usage_limit_switch(_event("x"), "example/model-free:free")
    assert reply == access.not_curated_message(PICKER)
    assert seen["commit"] == 0 and seen["tasks"] == []
    assert runner._session_model_overrides == {}


@pytest.mark.asyncio
async def test_admin_usage_limit_switch_still_works(monkeypatch):
    runner = _runner()
    seen = _usage_limit_runner(monkeypatch, runner)
    reply = await runner._accept_usage_limit_switch(_event("x", user=ADMIN), "example/model-free:free")
    assert reply == "✓ switched"
    assert seen["commit"] == 1 and len(seen["tasks"]) == 1
