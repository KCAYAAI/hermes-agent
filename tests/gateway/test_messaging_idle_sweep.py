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
from gateway.run_agent_cache import GatewayAgentCacheMixin
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

    def _evict_cached_agent_at_boundary(self, session_key: str, *, retired_session_ids=None) -> bool:
        self.evicted.append(session_key)
        return True


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


@pytest.fixture(autouse=True)
def _fresh_sweep_state(monkeypatch):
    """The sweep keeps per-profile module state (pending retries, window cursor, failure
    backoff); every test starts with none of it."""
    import gateway.session_idle_sweep as sweep_mod
    monkeypatch.setattr(sweep_mod, "_PENDING", {})
    monkeypatch.setattr(sweep_mod, "_CURSOR", {})
    monkeypatch.setattr(sweep_mod, "_FAILURES", {})
    monkeypatch.setattr(sweep_mod, "_WARNED_AT", {})


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
    # No skip: the adapter module imports without the Slack SDK (its SDK imports are guarded), and
    # this is the reseed proof, so any import failure must fail loudly.
    from plugins.platforms.slack.adapter import SlackAdapter
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


def test_slack_adapter_reseeds_when_the_routed_row_is_ended_but_the_entry_remains(store):
    """Between the DB commit and the route drop (or after a lost drop) the entry still names an
    ENDED row. Routing will mint a fresh session for that message, so the thread must reseed:
    the check consults the row, not just entry presence/suspension."""
    from plugins.platforms.slack.adapter import SlackAdapter
    from gateway.config import PlatformConfig

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="«redacted:xox…»"))
    adapter._session_store = store
    source = adapter._thread_session_source("C1", "1000.0", "U123", "", "group")
    old = store.get_or_create_session(source)
    # Entry present, row live -> active.
    assert adapter._has_active_session_for_thread("C1", "1000.0", "U123") is True

    store._db.end_session(old.session_id, "idle")
    assert store._entries[old.session_key].session_id == old.session_id  # route NOT dropped

    assert adapter._has_active_session_for_thread("C1", "1000.0", "U123") is False


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
# 10. Reconcile: route cleanup is state-based (idempotent, self-healing), not event-based
# ---------------------------------------------------------------------------

def test_route_left_by_a_failed_read_is_dropped_on_the_next_run(store, monkeypatch):
    """The DB commit succeeded but the routing read after it failed: the route survived that run.
    The next run ends nothing, yet still reconciles routes against the DB and drops it."""
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key, sid = entry.session_key, entry.session_id
    _age(store._db, sid, time.time() - STALE)
    runner = _Runner(store)
    real_reasons = sweep_mod._end_reasons
    monkeypatch.setattr(sweep_mod, "_end_reasons",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("database is locked")))

    with pytest.raises(OSError):
        _sweep(runner)
    assert store._db.get_session(sid)["end_reason"] == "idle"
    assert key in store._entries and runner.evicted == []  # cleanup lost in THIS run

    monkeypatch.setattr(sweep_mod, "_end_reasons", real_reasons)
    again = _sweep(runner)  # no gate to open: the failed run never stamped the interval

    assert again["ended"] == []
    assert again["routes_dropped"] == 1
    assert key not in store._entries and key not in _routing_rows(store)
    assert runner.evicted == [key]


def test_failed_save_still_evicts_the_dropped_routes_agent_and_propagates(store, monkeypatch, tmp_path):
    """``_save`` failed after the in-memory drop: the agent of the dropped key is still evicted
    (the key is gone from memory either way), the error propagates (interval not stamped), and
    the on-disk copy is repaired on restart by the startup prune; the next sweep is then clean."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key, sid = entry.session_key, entry.session_id
    _age(store._db, sid, time.time() - STALE)
    runner = _Runner(store)
    monkeypatch.setattr(store, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        _sweep(runner)
    assert store._db.get_session(sid)["end_reason"] == "idle"
    assert key not in store._entries and runner.evicted == [key]
    assert key in _routing_rows(store)  # the drop never reached disk

    reloaded = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        reloaded._ensure_loaded()
        assert key not in reloaded._entries  # startup prune: row ended, route gone
        again = _Runner(reloaded)
        result = _sweep(again)
        assert result["ended"] == [] and result["routes_dropped"] == 0 and again.evicted == []
        assert key not in _routing_rows(reloaded)
    finally:
        reloaded.close_all_db_handles()


def test_failed_save_is_retried_by_the_next_run_without_other_traffic(store, monkeypatch):
    """The in-memory drop happened but ``_save`` failed: the next run (no restart, nothing else
    to drop) first retries the save, so the durable index loses the route."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key, sid = entry.session_key, entry.session_id
    _age(store._db, sid, time.time() - STALE)
    runner = _Runner(store)
    real_save = store._save
    monkeypatch.setattr(store, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        _sweep(runner)
    assert key not in store._entries and key in _routing_rows(store)

    monkeypatch.setattr(store, "_save", real_save)
    again = _sweep(runner)

    assert again["ended"] == [] and again["routes_dropped"] == 0
    assert key not in _routing_rows(store) and key not in _mirror_keys(store)
    assert runner.evicted == [key]  # evicted once, by the first run


def test_failed_save_that_fails_again_keeps_the_run_failing(store, monkeypatch):
    """A pending save that still fails must not let the run stamp the interval."""
    _enable(monkeypatch)
    from gateway.session_idle_sweep import LAST_RUN_META_KEY
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)
    monkeypatch.setattr(store, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))
    runner = _Runner(store)
    with pytest.raises(OSError):
        _sweep(runner)
    with pytest.raises(OSError):
        _sweep(runner)
    assert store._db.get_meta(LAST_RUN_META_KEY) in (None, "")
    assert entry.session_key in _routing_rows(store)


def test_failed_eviction_is_retried_by_the_next_run(store, monkeypatch):
    """Eviction raised after the route was dropped: the key is remembered with its retired ids
    and evicted on the next run (the state-based reconcile no longer sees the route)."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key = entry.session_key
    _age(store._db, entry.session_id, time.time() - STALE)
    runner = _Runner(store)
    seen: list[tuple[str, frozenset]] = []
    fail = [True]

    def _evict(session_key, *, retired_session_ids=None):
        seen.append((session_key, frozenset(retired_session_ids or ())))
        if fail[0]:
            raise RuntimeError("release thread unavailable")
        runner.evicted.append(session_key)
        return True

    runner._evict_cached_agent_at_boundary = _evict
    first = _sweep(runner)
    assert first["routes_dropped"] == 1 and first["agents_evicted"] == 0
    assert key not in store._entries and runner.evicted == []

    fail[0] = False
    from gateway.session_idle_sweep import LAST_RUN_META_KEY
    store._db.set_meta(LAST_RUN_META_KEY, str(time.time() - 61 * 60))
    again = _sweep(runner)

    assert again["routes_dropped"] == 0 and again["agents_evicted"] == 1
    assert runner.evicted == [key]
    assert seen[-1] == (key, frozenset({entry.session_id}))  # same retired identity, not a bare key
    # Cleared on success: a third run does not evict again.
    store._db.set_meta(LAST_RUN_META_KEY, str(time.time() - 61 * 60))
    assert _sweep(runner)["agents_evicted"] == 0 and runner.evicted == [key]


def test_failed_lineage_lookup_leaves_the_key_for_the_next_run(store, monkeypatch):
    """A lineage read that fails would yield a partial id set; the key is left alone this run
    (its route still routes to the ended row, so the next run finds it again) rather than
    dropped with ancestors' cache identities missing."""
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    key = entry.session_key
    store._db.end_session(entry.session_id, "session_reset")
    runner = _Runner(store)
    real_lineage = SessionDB.get_compression_lineage
    broken = [True]

    def _lineage(self, sid):
        if broken[0]:
            raise OSError("database is locked")
        return real_lineage(self, sid)

    monkeypatch.setattr(SessionDB, "get_compression_lineage", _lineage)

    result = _sweep(runner)
    assert result["routes_dropped"] == 0
    assert key in store._entries and runner.evicted == []

    broken[0] = False
    store._db.set_meta(sweep_mod.LAST_RUN_META_KEY, str(time.time() - 61 * 60))
    again = _sweep(runner)
    assert again["routes_dropped"] == 1 and key not in store._entries and runner.evicted == [key]


def test_set_meta_runs_only_after_reconcile_succeeds(store, monkeypatch):
    """A failed reconcile must not stamp the interval, or the next tick waits an hour."""
    _enable(monkeypatch)
    from gateway.session_idle_sweep import LAST_RUN_META_KEY
    entry = store.get_or_create_session(_slack_source())
    _age(store._db, entry.session_id, time.time() - STALE)
    monkeypatch.setattr(store, "_save", lambda: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        _sweep(_Runner(store))

    assert store._db.get_meta(LAST_RUN_META_KEY) in (None, "")


def test_route_ended_elsewhere_is_reconciled_even_when_this_run_ends_nothing(store, monkeypatch):
    """Any non-recoverable end on the routed row (a CLI reset, a prune) retires the route."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    store._db.end_session(entry.session_id, "session_reset")
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["ended"] == [] and result["routes_dropped"] == 1
    assert entry.session_key not in store._entries
    assert runner.evicted == [entry.session_key]


def test_recoverable_end_keeps_the_route_for_the_runtime_self_heal(store, monkeypatch):
    """``agent_close`` rows are reopened by the next message (#54878 recovery): not ours to drop."""
    _enable(monkeypatch)
    entry = store.get_or_create_session(_slack_source())
    store._db.end_session(entry.session_id, "agent_close")
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["routes_dropped"] == 0
    assert entry.session_key in store._entries and runner.evicted == []


def test_parent_route_whose_tip_ended_earlier_is_reconciled(store, monkeypatch):
    """Route on a compression parent, tip ended by a PREVIOUS run whose lineage lookup failed:
    reconcile follows the parent's ``compression`` end to the tip and drops the route."""
    _enable(monkeypatch)
    db = store._db
    entry = store.get_or_create_session(_slack_source())
    key, parent = entry.session_key, entry.session_id
    assert db.try_acquire_compression_lock(parent, "compressor", ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id="tip", source="slack",
        messages=[{"role": "user", "content": "hello"}], compression_lock_holder="compressor")
    db.release_compression_lock(parent, "compressor")
    db.end_session("tip", "idle")
    assert store._entries[key].session_id == parent
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["ended"] == [] and result["routes_dropped"] == 1
    assert key not in store._entries and runner.evicted == [key]


def test_parent_route_with_a_live_tip_is_kept(store, monkeypatch):
    _enable(monkeypatch)
    db = store._db
    entry = store.get_or_create_session(_slack_source())
    key, parent = entry.session_key, entry.session_id
    assert db.try_acquire_compression_lock(parent, "compressor", ttl_seconds=60)
    db.publish_compression_child(
        parent_session_id=parent, child_session_id="tip", source="slack",
        messages=[{"role": "user", "content": "hello"}], compression_lock_holder="compressor")
    db.release_compression_lock(parent, "compressor")
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["routes_dropped"] == 0
    assert store._entries[key].session_id == parent and runner.evicted == []


def test_reconcile_examines_at_most_the_documented_bound(store, monkeypatch):
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    monkeypatch.setattr(sweep_mod, "_RECONCILE_LIMIT", 2)
    entries = [store.get_or_create_session(_slack_source(f"1700000000.00{i:04d}")) for i in range(3)]
    for e in entries:
        store._db.end_session(e.session_id, "session_reset")
    runner = _Runner(store)

    result = _sweep(runner)

    assert result["routes_dropped"] == 2
    assert sum(e.session_key in store._entries for e in entries) == 1


def test_bound_window_rotates_so_a_route_behind_2000_live_entries_is_reached(store, monkeypatch):
    """2,000 retained live entries sort ahead of one idle-ended route: a fixed prefix would starve
    it forever; the per-profile cursor reaches it on the second run."""
    from gateway.session import SessionEntry
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    assert sweep_mod._RECONCILE_LIMIT == 2000
    victim = store.get_or_create_session(_slack_source())
    key = victim.session_key
    store._db.end_session(victim.session_id, "idle")
    with store._lock:
        for i in range(2000):
            k = f"agent:main:slack:group:C1:0000{i:05d}"  # "0" < "1": sorts ahead of the victim
            assert k < key
            store._entries[k] = SessionEntry(
                session_key=k, session_id=f"live-{i}", created_at=victim.created_at,
                updated_at=victim.updated_at, platform=Platform.SLACK, chat_type="group")
    runner = _Runner(store)

    first = _sweep(runner)
    assert first["routes_dropped"] == 0 and key in store._entries
    store._db.set_meta(sweep_mod.LAST_RUN_META_KEY, str(time.time() - 61 * 60))
    second = _sweep(runner)

    assert second["routes_dropped"] == 1
    assert key not in store._entries and runner.evicted == [key]
    assert sum(1 for k in store._entries if k.startswith("agent:main:slack:group:C1:0000")) == 2000


# ---------------------------------------------------------------------------
# 10b. Compare-and-drop: a fresh route installed under the same key during the DB read survives
# ---------------------------------------------------------------------------

def test_fresh_route_installed_during_the_db_read_is_kept_and_persisted(store, monkeypatch):
    """Retirement snapshots ids, releases the lock for the DB reads, then drops. A fresh,
    active-token-bearing route installed under the same key meanwhile is a different
    conversation: the drop compares the routed id under the lock and leaves it, on disk too."""
    from gateway.session import SessionEntry
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    old = store.get_or_create_session(_slack_source())
    key = old.session_key
    _age(store._db, old.session_id, time.time() - STALE)
    runner = _Runner(store)
    real_reasons = sweep_mod._end_reasons
    fresh = SessionEntry(
        session_key=key, session_id="fresh-sid", created_at=old.created_at,
        updated_at=old.updated_at, platform=Platform.SLACK, chat_type="group",
        active_turn_token="tok-fresh")

    def _read_then_replace(*args, **kwargs):
        out = real_reasons(*args, **kwargs)
        with store._lock:
            store._entries[key] = fresh
            store._save()
        return out

    monkeypatch.setattr(sweep_mod, "_end_reasons", _read_then_replace)

    result = _sweep(runner)

    assert result["ended"] == [old.session_id]
    assert result["routes_dropped"] == 0 and runner.evicted == []
    assert store._entries[key] is fresh
    # Persisted unchanged: the durable row is the fresh route as it was written, not a removal.
    persisted = json.loads(_routing_rows(store)[key])
    assert persisted["session_id"] == "fresh-sid"
    assert persisted["active_turn_token"] == "tok-fresh"
    assert key in _mirror_keys(store)
    reloaded = SessionStore(store.sessions_dir, GatewayConfig())
    try:
        reloaded._ensure_loaded()
        assert reloaded._entries[key].session_id == "fresh-sid"
    finally:
        reloaded.close_all_db_handles()


def test_route_that_became_busy_during_the_db_read_is_kept(store, monkeypatch):
    """Same key, same id, but a turn started on it meanwhile: the busy recheck under the lock
    keeps the route (the turn's own unwind must still find its entry)."""
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    old = store.get_or_create_session(_slack_source())
    key = old.session_key
    store._db.end_session(old.session_id, "session_reset")
    runner = _Runner(store)
    real_reasons = sweep_mod._end_reasons

    def _read_then_start_turn(*args, **kwargs):
        out = real_reasons(*args, **kwargs)
        runner.running.append((key, object()))
        return out

    monkeypatch.setattr(sweep_mod, "_end_reasons", _read_then_start_turn)

    result = _sweep(runner)

    assert result["routes_dropped"] == 0 and runner.evicted == []
    assert store._entries[key].session_id == old.session_id


# ---------------------------------------------------------------------------
# 11. Profile scoping: the shared store holds every profile's keys, each profile its own DB
# ---------------------------------------------------------------------------

def test_identical_session_id_under_another_profiles_key_is_not_touched(store, monkeypatch):
    """Ids are timestamp + 8 hex chars, not globally unique. The row ended in THIS profile's DB
    retires only this profile's route; the other profile's entry with the same id stays."""
    from gateway.session import SessionEntry
    _enable(monkeypatch)
    mine = store.get_or_create_session(_slack_source())
    other_key = f"agent:b:{mine.session_key.split(':', 2)[2]}"
    with store._lock:
        store._entries[other_key] = SessionEntry(
            session_key=other_key, session_id=mine.session_id, created_at=mine.created_at,
            updated_at=mine.updated_at, platform=Platform.SLACK, chat_type="group")
    store._db.end_session(mine.session_id, "session_reset")
    runner = _Runner(store)

    result = _sweep(runner, profile="default")

    assert result["routes_dropped"] == 1
    assert mine.session_key not in store._entries
    assert store._entries[other_key].session_id == mine.session_id
    assert runner.evicted == [mine.session_key]


def test_busy_exclusion_only_counts_this_profiles_entries(store, monkeypatch):
    """A running turn under ANOTHER profile's key sharing the id must not shield this row."""
    from gateway.session import SessionEntry
    _enable(monkeypatch)
    mine = store.get_or_create_session(_slack_source())
    _age(store._db, mine.session_id, time.time() - STALE)
    other_key = f"agent:b:{mine.session_key.split(':', 2)[2]}"
    with store._lock:
        store._entries[other_key] = SessionEntry(
            session_key=other_key, session_id=mine.session_id, created_at=mine.created_at,
            updated_at=mine.updated_at, platform=Platform.SLACK, chat_type="group",
            active_turn_token="tok")
    runner = _Runner(store)
    runner.running.append((other_key, object()))

    result = _sweep(runner, profile="default")

    assert result["ended"] == [mine.session_id]
    assert mine.session_key not in store._entries
    assert other_key in store._entries


# ---------------------------------------------------------------------------
# 12. Identity-safe eviction: a fresh agent installed under the same key is left alone
# ---------------------------------------------------------------------------

class _CacheRunner(GatewayAgentCacheMixin, _Runner):
    """Runner driving the REAL ``GatewayAgentCacheMixin._evict_cached_agent`` against a cache of
    the production shape ``key -> (agent, sig, msg_count, session_id)``; the release thread is
    replaced by a recorder so ``evicted`` lists the keys whose agent was actually torn down."""

    def __init__(self, store):
        _Runner.__init__(self, store)
        import threading
        self._agent_cache: dict = {}
        self._agent_cache_lock = threading.Lock()

    def _peek_session_state(self, session_key):
        return None

    def _spawn_release_thread(self, target, args, name, *, inline_fallback, session_key=None):
        self.evicted.append(str(session_key))


def test_eviction_skips_a_replacement_agent_installed_before_the_evict(store, monkeypatch):
    _enable(monkeypatch)
    import gateway.session_idle_sweep as sweep_mod
    old = store.get_or_create_session(_slack_source())
    _age(store._db, old.session_id, time.time() - STALE)
    runner = _CacheRunner(store)
    fresh_agent = object()
    real_drop = sweep_mod._drop_routes

    def _drop_then_replace(*args, **kwargs):
        dropped = real_drop(*args, **kwargs)
        # Between the route drop and the evict a new conversation installed its own agent.
        with runner._agent_cache_lock:
            runner._agent_cache[old.session_key] = (fresh_agent, "sig", 0, "fresh-sid")
        return dropped

    monkeypatch.setattr(sweep_mod, "_drop_routes", _drop_then_replace)

    result = _sweep(runner)

    assert result["ended"] == [old.session_id] and result["routes_dropped"] == 1
    assert result["agents_evicted"] == 0 and runner.evicted == []
    assert runner._agent_cache[old.session_key][0] is fresh_agent


def test_eviction_hits_the_agent_still_bound_to_the_retired_session(store, monkeypatch):
    _enable(monkeypatch)
    old = store.get_or_create_session(_slack_source())
    _age(store._db, old.session_id, time.time() - STALE)
    runner = _CacheRunner(store)
    with runner._agent_cache_lock:
        runner._agent_cache[old.session_key] = (object(), "sig", 0, old.session_id)

    result = _sweep(runner)

    assert result["agents_evicted"] == 1 and runner.evicted == [old.session_key]


def test_eviction_hits_an_agent_bound_to_a_compression_ancestor(store, monkeypatch):
    """The cached agent may still record the PARENT id of the lineage whose tip ended."""
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
    _age(db, "tip", time.time() - STALE)
    runner = _CacheRunner(store)
    with runner._agent_cache_lock:
        runner._agent_cache[key] = (object(), "sig", 0, parent)

    result = _sweep(runner)

    assert result["ended"] == ["tip"]
    assert runner.evicted == [key]


# ---------------------------------------------------------------------------
# 13. A broken sweep is visible: one WARNING per interval per profile
# ---------------------------------------------------------------------------

def test_persistent_failure_warns_once_per_interval_per_profile(monkeypatch, caplog):
    from gateway.run import _housekeeping_messaging_idle_sweep
    import gateway.session_idle_sweep as sweep_mod

    def _boom(runner, *, profile=""):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(sweep_mod, "sweep_idle_messaging_sessions", _boom)
    monkeypatch.setattr(sweep_mod, "_WARNED_AT", {})
    clock = [1000.0]
    monkeypatch.setattr(sweep_mod.time, "monotonic", lambda: clock[0])
    runner = SimpleNamespace(session_store=None)

    with caplog.at_level(logging.DEBUG, logger="gateway.session_idle_sweep"):
        _housekeeping_messaging_idle_sweep(runner, "a")   # tick 1: WARNING
        clock[0] += 301                                   # past the first retry backoff (300 s)
        _housekeeping_messaging_idle_sweep(runner, "a")   # retry, same warn interval: debug
        _housekeeping_messaging_idle_sweep(runner, "b")   # other profile: its own WARNING
        clock[0] += sweep_mod._WARN_INTERVAL_S + 1
        _housekeeping_messaging_idle_sweep(runner, "a")   # next interval: WARNING again

    records = [r for r in caplog.records if "sweep exploded" in r.getMessage()]
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert [r.getMessage().split("(")[1].split(")")[0] for r in warnings] == ["a", "b", "a"]
    assert len(records) == 4 and records[1].levelno == logging.DEBUG


def test_failed_run_backs_off_exponentially_capped_at_the_interval_and_resets_on_success(monkeypatch):
    """After n consecutive failures the next attempt waits ``min(interval, 300 s * 2**(n-1))``;
    a success clears the count. Interval here: 60 minutes."""
    import gateway.session_idle_sweep as sweep_mod
    _enable(monkeypatch)
    calls: list[float] = []
    outcome = {"fail": True}

    def _sweep_stub(runner, *, profile=""):
        calls.append(clock[0])
        if outcome["fail"]:
            raise RuntimeError("sweep exploded")
        return {"skipped": None}

    monkeypatch.setattr(sweep_mod, "sweep_idle_messaging_sessions", _sweep_stub)
    clock = [10_000.0]
    monkeypatch.setattr(sweep_mod.time, "monotonic", lambda: clock[0])
    runner = SimpleNamespace(session_store=None)
    run = lambda: sweep_mod.run_idle_sweep_guarded(runner, profile="a")

    run()                                   # n=1 -> wait 300
    for dt in (60, 239):
        clock[0] += dt; run()
    assert len(calls) == 1
    clock[0] += 1; run()                    # 300 s later: n=2 -> wait 600
    assert len(calls) == 2
    clock[0] += 599; run(); assert len(calls) == 2
    clock[0] += 1; run()                    # n=3 -> 1200
    clock[0] += 1200; run()                 # n=4 -> 2400
    clock[0] += 2400; run()                 # n=5 -> min(3600, 4800) = 3600
    assert len(calls) == 5
    clock[0] += 3599; run(); assert len(calls) == 5
    clock[0] += 1; run(); assert len(calls) == 6   # n=6, still capped at 3600
    clock[0] += 3600
    outcome["fail"] = False
    run(); assert len(calls) == 7            # success resets the count ...
    outcome["fail"] = True
    clock[0] += 1; run(); assert len(calls) == 8   # ... so the very next tick runs again (n=1)
    clock[0] += 299; run(); assert len(calls) == 8
    clock[0] += 1; run(); assert len(calls) == 9
    # Another profile has its own count.
    sweep_mod.run_idle_sweep_guarded(runner, profile="b"); assert len(calls) == 10


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
