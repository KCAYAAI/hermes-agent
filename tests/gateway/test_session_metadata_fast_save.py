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
