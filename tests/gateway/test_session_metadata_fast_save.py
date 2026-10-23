"""KC 2026-10-22: set_session_metadata (Slack thread watermark on every thread message, the
session-reset clock on every inbound message) rewrote the WHOLE routing index on the event
loop: ~0.6 s per call at 11.6k entries. It is metadata-only, so it must use the single-row
upsert like other metadata writers, and still survive a reload."""

from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def _db_store(tmp_path):
    from hermes_state import SessionDB
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = SessionDB(db_path=tmp_path / "state.db")
    return store


def _source():
    return SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="group",
                         user_id="U1", thread_id="1.000")


def test_metadata_write_is_a_single_row_upsert_and_survives_reload(tmp_path):
    store = _db_store(tmp_path)
    entry = store.get_or_create_session(_source())
    full = MagicMock(wraps=store._persist_routing_data)
    store._persist_routing_data = full
    fast = MagicMock(wraps=store._db.save_gateway_routing_entry)
    store._db.save_gateway_routing_entry = fast

    assert store.set_session_metadata(entry.session_key, "slack_thread_watermark:C1:1.000", "1.5")
    assert full.call_count == 0
    assert fast.call_count == 1
    store._db.close()

    reloaded = _db_store(tmp_path)
    reloaded.get_or_create_session(_source())
    assert reloaded.get_session_metadata(entry.session_key, "slack_thread_watermark:C1:1.000") == "1.5"
    reloaded._db.close()


def test_metadata_write_on_missing_entry_is_false_and_writes_nothing(tmp_path):
    store = _db_store(tmp_path)
    fast = MagicMock()
    store._db.save_gateway_routing_entry = fast
    assert store.set_session_metadata("agent:main:slack:group:C9:9.000", "k", "v") is False
    fast.assert_not_called()
    store._db.close()


def test_metadata_write_falls_back_to_full_rewrite_without_deadlock(tmp_path):
    """No single-row saver (or it fails): the full rewrite still persists, without deadlock."""
    import threading
    store = _db_store(tmp_path)
    entry = store.get_or_create_session(_source())
    store._db.save_gateway_routing_entry = MagicMock(side_effect=OSError("upsert unavailable"))
    done = threading.Event()
    result = []

    def run():
        result.append(store.set_session_metadata(entry.session_key, "k", "v"))
        done.set()
    threading.Thread(target=run, daemon=True).start()
    assert done.wait(10), "set_session_metadata deadlocked on the fallback path"
    assert result == [True]
    store._db.close()
    reloaded = _db_store(tmp_path)
    reloaded.get_or_create_session(_source())
    assert reloaded.get_session_metadata(entry.session_key, "k") == "v"
    reloaded._db.close()


def test_terminal_cwd_save_writes_with_the_store_lock_free_and_keeps_the_mirror(tmp_path, monkeypatch):
    """KC 2026-11-14: the turn-end terminal-cwd save went through ``_update_entry``, which holds
    the store lock across the whole state.db write. On 2026-11-13 the event loop waited on that
    lock behind queued turn ends until the liveness watchdog killed the gateway. The save is the
    same full snapshot (state.db and the sessions.json mirror), written with the lock free."""
    import json
    import threading
    from types import SimpleNamespace
    from gateway import session_terminal_cwd as cwd_mod
    from tools import terminal_tool

    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(cwd_mod, "_record_owner", {})
    monkeypatch.setenv("TERMINAL_ENV", "local")
    store = _db_store(tmp_path)
    entry = store.get_or_create_session(_source())
    runner = SimpleNamespace(session_store=store)
    token = cwd_mod.restore_turn_cwd(runner, entry.session_key, None)
    assert token is not None
    terminal_tool.record_session_cwd(entry.session_key, str(tmp_path))

    real_persist = store._persist_routing_data
    lock_free = []

    def persist(*args, **kwargs):
        def probe():
            got = store._lock.acquire(timeout=2)
            lock_free.append(got)
            if got:
                store._lock.release()
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(10)
        return real_persist(*args, **kwargs)
    store._persist_routing_data = persist

    cwd_mod.persist_turn_cwd(runner, entry.session_key, token)
    assert lock_free == [True], "another thread must get the store lock while the index is written"
    mirror = json.loads((tmp_path / "sessions" / "sessions.json").read_text())
    assert mirror[entry.session_key]["metadata"]["terminal_cwd"] == str(tmp_path)
    store._db.close()

    reloaded = _db_store(tmp_path)
    reloaded.get_or_create_session(_source())
    assert reloaded.get_session_metadata(entry.session_key, "terminal_cwd") == str(tmp_path)
    reloaded._db.close()
