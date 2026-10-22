"""Gateway housekeeping: the hourly idle sweep for messaging sessions the gateway owns.

``SessionDB.end_idle_sessions`` closes the row with the reset reason ``idle``; the gateway chore
must then drop the in-memory route (and any route still pointing at a compression ancestor of the
ended tip) and evict the cached agent, so a late reply in that Slack thread starts a FRESH session
instead of resuming the row the DB already closed. Off by default (``sessions.messaging_idle_hours``
is ``0``); per served profile, against that profile's own store and config.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB

IDLE_HOURS = 6
STALE = 8 * 3600  # older than IDLE_HOURS
_ENABLED = {"sessions": {"messaging_idle_hours": IDLE_HOURS, "messaging_idle_interval_minutes": 60}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Runner:
    """The runner surface the chore reads: store, running turns, agent eviction."""

    def __init__(self, store: SessionStore, *, multiplex: bool = False):
        self.config = SimpleNamespace(multiplex_profiles=multiplex)
        self.session_store = store
        self.evicted: list[str] = []
        self.running: list[tuple[str, object]] = []

    def _running_agent_items(self):
        return list(self.running)

    def _running_agent_ids(self):
        return {id(agent) for _, agent in self.running}

    def _evict_cached_agent_at_boundary(self, session_key: str) -> None:
        self.evicted.append(session_key)


def _slack_source(thread: str = "1700000000.000100", *, profile: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK, chat_id="C1", chat_type="group", user_id="U1",
        thread_id=thread, scope_id="T1", profile=profile)


def _age(db: SessionDB, session_id: str, ts: float) -> None:
    """Make a row idle: old ``started_at``, an old message, old canonical activity."""
    db.append_message(session_id, role="user", content="hello")
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id))
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?", (ts, ts, session_id))
    db._conn.commit()


def _sweep(runner, **kwargs):
    from gateway.session_idle_sweep import sweep_idle_messaging_sessions
    return sweep_idle_messaging_sessions(runner, **kwargs)


def _enable(monkeypatch, cfg: dict = _ENABLED) -> None:
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: cfg)


@pytest.fixture
def store(tmp_path):
    handle = SessionStore(tmp_path / "sessions", GatewayConfig())
    yield handle
    handle.close_all_db_handles()


def _routing_rows(store: SessionStore) -> dict[str, str]:
    return store._db.load_gateway_routing_entries(scope=store._routing_scope())


def _mirror_keys(store: SessionStore) -> set[str]:
    path = Path(store.sessions_dir) / "sessions.json"
    if not path.exists():
        return set()
    return {k for k in json.loads(path.read_text(encoding="utf-8")) if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_defaults_are_off_and_slack_only():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    sessions = DEFAULT_CONFIG["sessions"]
    assert sessions["messaging_idle_hours"] == 0
    assert sessions["messaging_idle_sources"] == ["slack"]
    assert sessions["messaging_idle_interval_minutes"] == 60


# ---------------------------------------------------------------------------
# 1. Off by default
# ---------------------------------------------------------------------------

def test_off_by_default_never_calls_end_idle_sessions(store, monkeypatch):
    calls: list = []
    monkeypatch.setattr(SessionDB, "end_idle_sessions", lambda self, **kw: calls.append(kw) or [])
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)
    runner = _Runner(store)

    result = _sweep(runner)

    assert calls == []
    assert result["skipped"] == "disabled"
    assert store._db.get_session(entry.session_id)["ended_at"] is None
    assert entry.session_key in store._entries
    assert runner.evicted == []


# ---------------------------------------------------------------------------
# 2. Idle slack session with a route: ended, route gone (memory + disk), agent evicted
# ---------------------------------------------------------------------------

def test_idle_slack_session_is_ended_route_dropped_and_agent_evicted(store, monkeypatch, tmp_path):
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key, sid = entry.session_key, entry.session_id
    _age(store._db, sid, time.time() - STALE)
    assert key in _routing_rows(store) and key in _mirror_keys(store)
    runner = _Runner(store)

    result = _sweep(runner)

    row = store._db.get_session(sid)
    assert row["ended_at"] is not None and row["end_reason"] == "idle"
    assert result["ended"] == [sid]
    assert key not in store._entries
    # Persisted drop: routing table and legacy mirror both lost the key ...
    assert key not in _routing_rows(store)
    assert key not in _mirror_keys(store)
    # ... so a reloaded store never sees it either.
    reloaded = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        reloaded._ensure_loaded()
        assert key not in reloaded._entries
    finally:
        reloaded.close_all_db_handles()
    assert runner.evicted == [key]


def test_logs_one_info_line_per_profile_when_anything_ended(store, monkeypatch, caplog):
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)

    with caplog.at_level(logging.INFO, logger="gateway.session_idle_sweep"):
        _sweep(_Runner(store))

    lines = [r for r in caplog.records if r.levelno == logging.INFO and "idle" in r.getMessage()]
    assert len(lines) == 1
    text = lines[0].getMessage()
    assert "ended=1" in text and "routes_dropped=1" in text and "agents_evicted=1" in text


# ---------------------------------------------------------------------------
# 3. Busy sessions are spared: active-turn token on the entry / agent mid-turn
# ---------------------------------------------------------------------------

def test_entry_with_active_turn_token_is_not_ended_and_keeps_its_route(store, monkeypatch):
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)
    token = store.mark_turn_active(entry.session_key)
    assert token and store._entries[entry.session_key].active_turn_token == token
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["ended"] == []
    assert store._db.get_session(entry.session_id)["ended_at"] is None
    assert entry.session_key in store._entries
    assert runner.evicted == []


def test_session_with_a_running_agent_is_not_ended(store, monkeypatch):
    """``_running_agent_items`` names the busy session KEY; its id joins ``exclude_ids``."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)
    runner = _Runner(store)
    runner.running.append((entry.session_key, object()))

    result = _sweep(runner)

    assert result["ended"] == []
    assert store._db.get_session(entry.session_id)["ended_at"] is None
    assert entry.session_key in store._entries
    assert runner.evicted == []


# ---------------------------------------------------------------------------
# 4. Fresh session untouched; non-slack source untouched
# ---------------------------------------------------------------------------

def test_fresh_and_non_slack_sessions_are_untouched(store, monkeypatch):
    _enable(monkeypatch)
    fresh = store.get_or_create_session(_slack_source("1700000000.000200"))
    telegram = store.get_or_create_session(SessionSource(
        platform=Platform.TELEGRAM, chat_id="5140768830", chat_type="dm", user_id="5140768830"))
    _age(store._db, telegram.session_id, time.time() - STALE)
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["ended"] == []
    for entry in (fresh, telegram):
        assert store._db.get_session(entry.session_id)["ended_at"] is None
        assert entry.session_key in store._entries
    assert runner.evicted == []


# ---------------------------------------------------------------------------
# 5. Compression lineage: route on the PARENT, idle tip -> tip ended, parent route dropped
# ---------------------------------------------------------------------------

def test_route_on_compression_parent_is_dropped_when_idle_tip_ends(store, monkeypatch):
    """Route checks heal a parent-pointing route to its tip and then SKIP the stale check, so a
    route left on the compression parent would keep routing to the idle-ended tip."""
    _enable(monkeypatch)
    db = store._db
    entry = store.get_or_create_session(_slack_source())
    key, parent = entry.session_key, entry.session_id
    _age(db, parent, time.time() - STALE)
    assert db.try_acquire_compression_lock(parent, "compressor", ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id="tip", source="slack",
        messages=[{"role": "user", "content": "hello"}], compression_lock_holder="compressor")
    db.release_compression_lock(parent, "compressor")
    assert db.get_session(parent)["end_reason"] == "compression"
    _age(db, "tip", time.time() - STALE)
    # The route was never healed: it still names the parent.
    assert store._entries[key].session_id == parent
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["ended"] == ["tip"]
    assert db.get_session("tip")["end_reason"] == "idle"
    assert db.get_session(parent)["end_reason"] == "compression"  # ancestor untouched in the DB
    assert key not in store._entries
    assert key not in _routing_rows(store)
    assert runner.evicted == [key]
    # And a late reply does not land on either row of the ended lineage.
    fresh = store.get_or_create_session(_slack_source())
    assert fresh.session_id not in (parent, "tip")


# ---------------------------------------------------------------------------
# 6. Interval gate
# ---------------------------------------------------------------------------

def test_interval_gate_skips_within_interval_and_runs_again_after(store, monkeypatch):
    _enable(monkeypatch)
    from gateway.session_idle_sweep import LAST_RUN_META_KEY
    first = store.get_or_create_session(_slack_source("1700000000.000100"))
    _age(store._db, first.session_id, time.time() - STALE)
    runner = _Runner(store)

    assert _sweep(runner)["ended"] == [first.session_id]

    second = store.get_or_create_session(_slack_source("1700000000.000200"))
    _age(store._db, second.session_id, time.time() - STALE)
    again = _sweep(runner)
    assert again["skipped"] == "interval" and again["ended"] == []
    assert store._db.get_session(second.session_id)["ended_at"] is None
    assert second.session_key in store._entries

    # Past the interval the gate opens.
    store._db.set_meta(LAST_RUN_META_KEY, str(time.time() - 61 * 60))
    assert _sweep(runner)["ended"] == [second.session_id]
    assert second.session_key not in store._entries


# ---------------------------------------------------------------------------
# 8. Late reply: the key proof
# ---------------------------------------------------------------------------

def test_late_reply_gets_a_fresh_session_and_the_idle_row_stays_ended(store, monkeypatch):
    _enable(monkeypatch)
    source = _slack_source()
    old = store.get_or_create_session(source)
    _age(store._db, old.session_id, time.time() - STALE)

    _sweep(_Runner(store))

    late = store.get_or_create_session(source)
    assert late.session_key == old.session_key
    assert late.session_id != old.session_id
    ended = store._db.get_session(old.session_id)
    assert ended["ended_at"] is not None and ended["end_reason"] == "idle"
    assert store._db.get_session(late.session_id)["ended_at"] is None
    assert late.prev_session_id is None  # a fresh conversation, not a reset continuation


def test_slack_adapter_sees_no_active_thread_session_after_the_sweep(store, monkeypatch):
    """Slack reseeds up to 30 thread messages when ``_has_active_session_for_thread`` is False."""
    try:
        from plugins.platforms.slack.adapter import SlackAdapter
    except Exception as exc:  # pragma: no cover - environment without the Slack SDK
        pytest.skip(f"Slack adapter unavailable: {exc}")
    from gateway.config import PlatformConfig

    _enable(monkeypatch)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._session_store = store
    source = adapter._thread_session_source("C1", "1000.0", "U123", "", "group")
    old = store.get_or_create_session(source)
    assert adapter._has_active_session_for_thread("C1", "1000.0", "U123")
    _age(store._db, old.session_id, time.time() - STALE)

    _sweep(_Runner(store))

    assert not adapter._has_active_session_for_thread("C1", "1000.0", "U123")
    late = store.get_or_create_session(source)
    assert late.session_id != old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == "idle"


# ---------------------------------------------------------------------------
# 9. Race: message lands between the DB commit and the route drop
# ---------------------------------------------------------------------------

def test_message_between_db_end_and_route_drop_routes_to_a_fresh_session(store):
    source = _slack_source()
    old = store.get_or_create_session(source)
    _age(store._db, old.session_id, time.time() - STALE)
    # The DB half of the sweep has committed; the entry is still published.
    assert store._db.end_idle_sessions(
        sources=("slack",), max_idle_seconds=IDLE_HOURS * 3600, end_reason="idle") == [old.session_id]
    assert store._entries[old.session_key].session_id == old.session_id

    late = store.get_or_create_session(source)

    assert late.session_id != old.session_id
    assert store._db.get_session(old.session_id)["end_reason"] == "idle"
    assert store._entries[old.session_key].session_id == late.session_id


# ---------------------------------------------------------------------------
# 7. Multiplex: one sweep per served profile, each against its own store; isolation
# ---------------------------------------------------------------------------

class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


def _profile(home: Path, base_url: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: nous\n"
        "sessions:\n"
        f"  messaging_idle_hours: {IDLE_HOURS}\n"
        "  messaging_idle_interval_minutes: 60\n",
        encoding="utf-8")
    (home / ".env").write_text(f"NOUS_INFERENCE_BASE_URL={base_url}\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {
        "access_token": "x.y.z", "refresh_token": "r", "expires_at": 0,
        "portal_base_url": "https://portal.nousresearch.com", "client_id": "c"}}}), encoding="utf-8")


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (= multiplex ``default``) and served named profile B under ``A/profiles/b``."""
    fake_home = tmp_path / "home"
    a = fake_home / ".hermes"
    b = a / "profiles" / "b"
    _profile(a, "https://a.example/v1")
    _profile(b, "https://b.example/v1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    # The hermetic conftest pins ``hermes_state.DEFAULT_DB_PATH`` at one sandbox store; restore the
    # import-time sentinel so an argless ``acquire()`` resolves through the profile scope.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), f"unpinned store escaped the sandbox: {resolved}"
    return a, b


def _two_profile_sessions(a: Path, b: Path):
    """One idle Slack session per profile, each row in that profile's own state.db."""
    store = SessionStore(a / "sessions", GatewayConfig(multiplex_profiles=True))
    default_entry = store.get_or_create_session(_slack_source("1700000000.000100"))
    b_entry = store.get_or_create_session(_slack_source("1700000000.000200", profile="b"))
    assert default_entry.session_key.startswith("agent:main:")
    assert b_entry.session_key.startswith("agent:b:")
    db_a = store._db_for_key(default_entry.session_key)
    db_b = store._db_for_key(b_entry.session_key)
    assert Path(db_a.db_path) == a / "state.db"
    assert Path(db_b.db_path) == b / "state.db"
    _age(db_a, default_entry.session_id, time.time() - STALE)
    _age(db_b, b_entry.session_id, time.time() - STALE)
    return store, default_entry, b_entry, db_a, db_b


def test_multiplexed_sweep_visits_every_served_profiles_own_store(two_homes):
    from agent.secret_scope import set_multiplex_active

    a, b = two_homes
    store, default_entry, b_entry, db_a, db_b = _two_profile_sessions(a, b)
    runner = _Runner(store, multiplex=True)
    set_multiplex_active(True)
    try:
        gateway_run._start_gateway_housekeeping(_Ticks(1), interval=0, runner=runner)
    finally:
        set_multiplex_active(False)
        store.close_all_db_handles()

    assert db_a.get_session(default_entry.session_id)["end_reason"] == "idle"
    assert db_b.get_session(b_entry.session_id)["end_reason"] == "idle"
    assert default_entry.session_key not in store._entries
    assert b_entry.session_key not in store._entries
    assert sorted(runner.evicted) == sorted([default_entry.session_key, b_entry.session_key])


def test_one_profiles_failure_does_not_stop_the_next_profiles_sweep(two_homes, monkeypatch):
    import hermes_state_registry as registry
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import get_hermes_home

    a, b = two_homes
    store, default_entry, b_entry, db_a, db_b = _two_profile_sessions(a, b)
    real_acquire = registry.acquire

    def _acquire(*args, **kwargs):
        if get_hermes_home() == a:
            raise OSError("launch store unavailable")
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(registry, "acquire", _acquire)
    runner = _Runner(store, multiplex=True)
    set_multiplex_active(True)
    try:
        gateway_run._start_gateway_housekeeping(_Ticks(1), interval=0, runner=runner)
    finally:
        set_multiplex_active(False)
        store.close_all_db_handles()

    # A failed, untouched: row open, route kept, nothing evicted for it.
    assert db_a.get_session(default_entry.session_id)["ended_at"] is None
    assert default_entry.session_key in store._entries
    # B swept.
    assert db_b.get_session(b_entry.session_id)["end_reason"] == "idle"
    assert b_entry.session_key not in store._entries
    assert runner.evicted == [b_entry.session_key]
    assert get_hermes_home() == a
