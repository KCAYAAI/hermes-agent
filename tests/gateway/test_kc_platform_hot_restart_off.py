"""KC carry: ``gateway.platform_hot_restart: false`` keeps every running platform adapter when its
config changes. A hot reconcile still starts a platform that has no adapter yet, but never restarts or
stops a live one; the change applies at the next gateway restart (live r36 behavior). A Slack adapter
torn down mid-turn would cancel the reply, reaction and progress-card work KC's Slack carries still owe.

Live: a real ``hermes gateway run`` on a disposable root, the same harness as upstream's
``test_platform_hot_start.py``; "running" is a webhook socket that accepts.
"""
import asyncio
import os
import threading
from pathlib import Path

import pytest

from gateway.config import Platform
from gateway.run_platform_reconcile import (
    BuildPrint, GatewayPlatformReconcileMixin, PlatformPlan, _defer_held, hold_live_adapters, plan_platform_changes,
    read_platform_hot_restart)
from tests.gateway.test_multiplex_hot_serve_live import control, start_daemon, stop, tail
from tests.gateway.test_platform_hot_start import _accepts, _free_port
from tests.waiting import wait_until


def _home(tmp_path, text):
    home = tmp_path / "launch"
    home.mkdir(exist_ok=True)
    if text is not None:
        (home / "config.yaml").write_text(text, encoding="utf-8")
    return home


@pytest.mark.parametrize("text, expected", [
    ("gateway:\n  multiplex_profiles: true\n", True),          # key unset: upstream's behavior
    (None, True),                                             # no file: upstream's behavior
    ("model: x\n", True),
    ("gateway:\n  platform_hot_restart:\n", True),               # empty value: the default
    ("gateway:\n  platform_hot_restart: true\n", True),
    ("gateway:\n  platform_hot_restart: false\n", False),
    ("gateway:\n  platform_hot_restart: 'off'\n", False),
    ("gateway:\n  platform_hot_restart: '0'\n", False),
    ("gateway:\n  platform_hot_restart: 0\n", False),
    ("gateway: [unclosed\n", False),                          # unparseable: hold
    ("- a\n- b\n", False),                                    # not a mapping: hold
    ("", False),                                              # emptied file: hold
])
def test_the_switch_and_a_bad_read_holds(tmp_path, text, expected):
    assert read_platform_hot_restart(_home(tmp_path, text)) is expected


def test_a_config_path_that_cannot_be_read_holds(tmp_path):
    home = tmp_path / "launch"
    (home / "config.yaml").mkdir(parents=True)
    assert read_platform_hot_restart(home) is False


def test_the_read_takes_no_config_lock_or_cache_a_loop_reader_shares(tmp_path, monkeypatch):
    import hermes_cli.config as config_module
    home = _home(tmp_path, "gateway:\n  platform_hot_restart: false\n")
    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("shared loader used")))
    held, release = threading.Event(), threading.Event()

    def hold_lock():
        with config_module._CONFIG_LOCK:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert held.wait(5)
    result = []
    reader = threading.Thread(target=lambda: result.append(read_platform_hot_restart(home)), daemon=True)
    reader.start()
    reader.join(5)
    release.set()
    holder.join(5)
    assert result == [False], "the switch read waited on the shared config lock"


def test_a_profile_scope_or_home_override_does_not_redirect_the_owner_read(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    owner = _home(tmp_path, "gateway:\n  platform_hot_restart: false\n")
    other = tmp_path / "profiles" / "brand"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("gateway:\n  platform_hot_restart: true\n", encoding="utf-8")
    token = set_hermes_home_override(other)
    try:
        assert read_platform_hot_restart(owner) is False
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("error", [PermissionError(13, "denied"), OSError(5, "Input/output error"),
                                   IsADirectoryError(21, "is a directory"), MemoryError()])
def test_a_failed_open_of_an_existing_file_holds_and_only_not_found_is_upstreams_default(tmp_path, monkeypatch,
                                                                                         error):
    home = _home(tmp_path, "gateway:\n  platform_hot_restart: false\n")
    real_open = open

    def failing_open(path, *a, **k):
        if str(path).endswith("config.yaml"):
            raise error
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", failing_open)
    assert read_platform_hot_restart(home) is False
    monkeypatch.setattr("builtins.open", lambda path, *a, **k: (_ for _ in ()).throw(FileNotFoundError(2, "x")))
    assert read_platform_hot_restart(home) is True


def test_a_baseline_retry_from_the_watcher_never_reads_the_switch_again(tmp_path, monkeypatch):
    """Boot samples False; a later baseline step fails, so the watcher's start calls the baseline again
    while the owner file is briefly absent. The confirmed False must survive."""
    home = _home(tmp_path, "gateway:\n  platform_hot_restart: false\n")
    reads = []
    real = read_platform_hot_restart
    monkeypatch.setattr("gateway.run_platform_reconcile.read_platform_hot_restart",
                        lambda h: reads.append(h) or real(h))
    monkeypatch.setattr("gateway.run_platform_reconcile._credential_print",
                        lambda h: (_ for _ in ()).throw(MemoryError()))
    monkeypatch.setattr("gateway.run_profile_reconcile.profile_serve_signature", lambda h: ("sig",))
    runner = GatewayPlatformReconcileMixin()
    runner._primary_platform_home = lambda: home
    runner._primary_platform_signature = None
    asyncio.run(runner._note_boot_platform_signature())
    assert runner._kc_platform_hot_restart is False and runner._primary_platform_signature is None
    (home / "config.yaml").unlink()
    asyncio.run(runner._start_primary_platform_watch())
    assert runner._kc_platform_hot_restart is False and len(reads) == 1


def test_a_boot_sample_that_raises_holds(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run_platform_reconcile.read_platform_hot_restart",
                        lambda h: (_ for _ in ()).throw(RuntimeError("worker failed")))
    monkeypatch.setattr("gateway.run_profile_reconcile.profile_serve_signature", lambda h: ("sig",))
    runner = GatewayPlatformReconcileMixin()
    runner._primary_platform_home = lambda: _home(tmp_path, None)
    asyncio.run(runner._note_boot_platform_signature())
    assert runner._kc_platform_hot_restart is False


@pytest.mark.parametrize("text, expected", [("gateway:\n  platform_hot_restart: false\n", False),
                                            ("gateway:\n  multiplex_profiles: true\n", True)])
def test_boot_reads_the_switch_off_loop_before_any_adapter_is_built(tmp_path, monkeypatch, text, expected):
    home = _home(tmp_path, text)
    monkeypatch.setattr("gateway.run_profile_reconcile.profile_serve_signature", lambda h: ("sig",))
    threads = []
    real = read_platform_hot_restart

    def spy(h):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(h)

    monkeypatch.setattr("gateway.run_platform_reconcile.read_platform_hot_restart", spy)
    runner = GatewayPlatformReconcileMixin()
    runner._primary_platform_home = lambda: home
    asyncio.run(runner._note_boot_platform_signature())
    assert runner._kc_platform_hot_restart is expected and threads == [False]


def _plan():
    old, new = BuildPrint("a"), BuildPrint("b")
    return plan_platform_changes(
        wanted={Platform.SLACK: new, Platform.WEBHOOK: new},
        built={Platform.SLACK: old, Platform.TELEGRAM: old},
        live={Platform.SLACK, Platform.TELEGRAM})


def test_off_holds_restarts_and_stops_and_still_starts_a_new_platform():
    plan = _plan()
    assert plan.restart == (Platform.SLACK,) and plan.stop == (Platform.TELEGRAM,)  # upstream's plan
    assert hold_live_adapters(plan, "brand", False) == PlatformPlan(
        start=(Platform.WEBHOOK,), restart=(), stop=(), forget=(), held=(Platform.SLACK, Platform.TELEGRAM))


def test_on_leaves_upstreams_plan_untouched():
    plan = _plan()
    assert hold_live_adapters(plan, "brand", True) is plan


def test_a_receipt_with_a_held_platform_says_a_restart_is_needed():
    held = hold_live_adapters(_plan(), "brand", False)
    receipt = _defer_held({"applied": True, "changed": False, "reasons": {"x": "kept"}}, held)
    assert receipt["applied"] is False and receipt["reason"] == "restart_needed"
    assert set(receipt["reasons"]) == {"x", "slack", "telegram"}
    untouched = {"applied": True, "reasons": {}}
    assert _defer_held(dict(untouched), _plan()) == untouched


@pytest.mark.parametrize("switch, held", [(False, True), (True, False), (None, True)])
def test_a_reconcile_reads_nothing_and_uses_the_boot_value(monkeypatch, switch, held):
    monkeypatch.setattr("gateway.run_platform_reconcile.read_platform_hot_restart",
                        lambda h: (_ for _ in ()).throw(AssertionError("read during a reconcile")))
    runner = GatewayPlatformReconcileMixin()
    if switch is not None:          # None: a runner that never booted holds
        runner._kc_platform_hot_restart = switch
    plan = runner._kc_hold_live_adapters(_plan(), "brand")
    assert bool(plan.held) is held


@pytest.mark.parametrize("switch, retired", [(False, 0), (True, 1)])
def test_a_served_profiles_live_slack_adapter_is_not_retired_when_the_switch_is_off(monkeypatch, switch, retired):
    """The served-profile call site (six of KC's seven Slack adapters are secondaries)."""
    import contextlib
    import types
    monkeypatch.setattr("gateway.run._profile_runtime_scope", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr("gateway.run._platform_has_bot_credential", lambda *a, **k: True)
    adapter = object()
    runner = GatewayPlatformReconcileMixin()
    runner._profile_adapters = {"brand": {Platform.SLACK: adapter}}
    runner._profile_failed_platforms = {}
    runner._profile_build_prints = {"brand": {Platform.SLACK: BuildPrint("old")}}
    runner._secondary_build_print = lambda home, platform, config: BuildPrint("new")
    runner._kc_platform_hot_restart = switch
    calls = []

    async def retire(profile_name, profile_home, platform, queued, claimed):
        calls.append((profile_name, platform))
        runner._profile_adapters[profile_name].pop(platform, None)

    runner._retire_secondary_platform = retire
    current = types.SimpleNamespace(platforms={Platform.SLACK: types.SimpleNamespace(enabled=True)})
    asyncio.run(runner._retire_stale_profile_platforms("brand", Path("/brand"), current, {}))
    assert len(calls) == retired
    plan = runner._profile_platform_plans["brand"]
    if switch:
        assert plan.restart == (Platform.SLACK,) and not plan.held
    else:
        assert runner._profile_adapters["brand"][Platform.SLACK] is adapter
        assert plan.held == (Platform.SLACK,) and plan.restart == ()
        assert runner._profile_build_prints["brand"][Platform.SLACK] == BuildPrint("old")


def test_a_served_profiles_receipt_and_ledger_answer_say_a_restart_is_needed():
    """The named-profile receipt (``_close_profile_receipts``) reports the held platform before the
    ledger records it, so the ``reconcile-platforms`` answer drained from the ledger says it too."""
    runner = GatewayPlatformReconcileMixin()
    adapter = object()
    runner._profile_adapters = {"brand": {Platform.SLACK: adapter}}
    runner._profile_failed_platforms = {}
    runner._served_profile_homes = {"brand": Path("/brand")}
    runner._secondary_platform_states = lambda name: {"slack": "connected"}
    runner._profile_platform_plans = {"brand": hold_live_adapters(PlatformPlan(
        start=(), restart=(Platform.SLACK,), stop=()), "brand", False)}
    receipts = runner._close_profile_receipts({"brand": {Platform.SLACK}}, {})
    receipt = receipts["brand"]
    assert receipt["restarted"] == [] and receipt["applied"] is False and receipt["reason"] == "restart_needed"
    assert "slack" in receipt["reasons"]
    answer = runner.platform_change_ledger.drain("brand")
    assert answer["applied"] is False and answer["reason"] == "restart_needed" and "slack" in answer["reasons"]
    assert runner._profile_adapters["brand"][Platform.SLACK] is adapter


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_a_running_platform_survives_a_config_change_and_a_disable_when_hot_restart_is_off(tmp_path):
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: false\n  platform_hot_restart: false\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first, second = _free_port(), _free_port()
        dotenv = root / ".env"
        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={first}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["applied"] is True and answer["started"] == ["webhook"], (answer, tail(mux))
        wait_until(lambda: _accepts(first), timeout=10, interval=0.1, what="webhook listening")

        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["restarted"] == [] and answer["platforms"]["webhook"] == "connected", (answer, tail(mux))
        assert answer["applied"] is False and answer["reason"] == "restart_needed", answer
        assert _accepts(first) and not _accepts(second), "the running adapter was rebuilt"

        dotenv.write_text(f"WEBHOOK_ENABLED=false\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["stopped"] == [] and answer["applied"] is False, (answer, tail(mux))
        assert _accepts(first), "the running adapter was stopped"
        assert "gateway.platform_hot_restart is false, so the running adapter is kept" in tail(mux)
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)
