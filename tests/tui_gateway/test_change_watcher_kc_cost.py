"""KC: the change watcher must stay cheap on a busy multiplex gateway.

With no Desktop/TUI client attached, the watched-change pass is skipped entirely, and a change made
while nobody listened still broadcasts once a client attaches. (The session-table digest the pass
reads is pinned against the real store in
``tests/persistence_sqlite/housekeeping/test_session_projection_digest_kc.py``.)
"""

import sqlite3

import pytest

from persistence.sqlite.housekeeping import maintenance as maintenance_module
from tui_gateway import change_watcher, server


@pytest.fixture()
def watcher_home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("display: {}\n")
    (tmp_path / "cron").mkdir()
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_change_sigs", {})
    monkeypatch.setattr(server, "_change_checked_at", {})
    monkeypatch.setattr(server, "_change_broadcast_at", {})
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_watcher_home", lambda: tmp_path)
    events = []
    monkeypatch.setattr(
        server, "_broadcast_global_event", lambda ev, payload=None: events.append((ev, payload))
    )
    return tmp_path, events


def _store(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, message_count INTEGER)")
    conn.execute("DELETE FROM sessions")
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_no_listener_skips_the_watched_pass(watcher_home, monkeypatch):
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", False)
    assert server._watch_has_listener() is False
    monkeypatch.setattr(server, "_live_transports", {object()})
    assert server._watch_has_listener() is True
    monkeypatch.setattr(server, "_live_transports", set())
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", True)
    assert server._watch_has_listener() is True


def test_change_made_while_unwatched_fires_when_a_client_attaches(watcher_home, monkeypatch):
    home, events = watcher_home
    db = home / "state.db"
    _store(db, [("s1", "a", 1)])
    server._broadcast_watched_changes(now=0.0)   # seeded while a client was attached
    _store(db, [("s1", "renamed", 1)])           # nobody listening: the loop skips passes
    server._broadcast_watched_changes(now=10.0)  # first pass after a client attaches
    assert ("sessions.changed", {}) in events


@pytest.mark.parametrize("old_sqlite", [False, True], ids=["group_concat-order-by", "pre-3.44-subquery"])
def test_watcher_digests_an_invalid_utf8_title_without_the_mtime_fallback(tmp_path, monkeypatch, old_sqlite):
    """The watcher swallows a failed digest and falls back to mtime, so a broken digest branch would
    still broadcast. Read through the watcher's own entry point and require a real digest (not the
    fallback) that moves when only the undecodable title changes, on both SQL paths."""
    if old_sqlite:
        monkeypatch.setattr(maintenance_module.sqlite3, "sqlite_version_info", (3, 40, 0))
    db = tmp_path / "state.db"
    _store(db, [("s1", "a", 1)])

    def digest_with_title(hex_bytes):
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE sessions SET title = CAST(x'{hex_bytes}' AS TEXT)")
        conn.commit()
        conn.close()
        change_watcher._sessions_db_sig_cache.clear()
        return change_watcher._session_db_content_sig(db)

    first, second = digest_with_title("ff61"), digest_with_title("ff62")
    for sig in (first, second):
        assert sig is not None and sig[0] != change_watcher._MTIME_FALLBACK
    assert first != second
