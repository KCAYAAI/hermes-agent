"""KC: the change watcher must stay cheap on a busy multiplex gateway.

1. The session-table signature is computed by SQLite in one statement, and it still moves for
   every signature column (NULL vs empty string, field boundaries, row order by id).
2. With no Desktop/TUI client attached, the watched-change pass is skipped entirely, and a
   change made while nobody listened still broadcasts once a client attaches.
"""

import sqlite3

import pytest

from tui_gateway import server


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


def _sig(home, monkeypatch):
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})
    return server._session_db_content_sig(home / "state.db")


@pytest.mark.parametrize(
    "a, b",
    [
        ([("s1", None, 1)], [("s1", "", 1)]),          # NULL vs empty string
        ([("s1", "ab", 1)], [("s1", "a", 1)]),          # value change
        ([("s1", "x", 1)], [("s1", b"x", 1)]),          # text vs BLOB with the same bytes
        ([("s1", "x", 1)], [("s1", "x", 1), ("s2", "x", 1)]),  # row added
        ([("s1", "a\x1fb", 1)], [("s1", "a", 1)]),      # separator char inside a value
    ],
)
def test_signature_moves_for_column_level_changes(watcher_home, monkeypatch, a, b):
    home, _ = watcher_home
    _store(home / "state.db", a)
    before = _sig(home, monkeypatch)
    _store(home / "state.db", b)
    assert _sig(home, monkeypatch) != before


def test_signature_is_stable_and_independent_of_insert_order(watcher_home, monkeypatch):
    home, _ = watcher_home
    _store(home / "state.db", [("s1", "a", 1), ("s2", "b", 2)])
    first = _sig(home, monkeypatch)
    assert _sig(home, monkeypatch) == first
    _store(home / "state.db", [("s2", "b", 2), ("s1", "a", 1)])
    assert _sig(home, monkeypatch) == first


def test_empty_sessions_table_has_a_signature(watcher_home, monkeypatch):
    home, _ = watcher_home
    _store(home / "state.db", [])
    sig = _sig(home, monkeypatch)
    assert sig is not None and sig[0] != "mtime-fallback"


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


def test_invalid_utf8_title_still_gets_a_moving_digest(watcher_home, monkeypatch):
    home, _ = watcher_home
    db = home / "state.db"
    _store(db, [("s1", "a", 1)])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE sessions SET title = CAST(x'ff61' AS TEXT)")
    conn.commit()
    conn.close()
    bad = _sig(home, monkeypatch)
    assert bad is not None and bad[0] != "mtime-fallback"
    conn = sqlite3.connect(db)
    conn.execute("UPDATE sessions SET title = CAST(x'ff62' AS TEXT)")
    conn.commit()
    conn.close()
    assert _sig(home, monkeypatch) != bad


def test_pre_3_44_sqlite_uses_the_ordered_subquery(watcher_home, monkeypatch):
    home, _ = watcher_home
    _store(home / "state.db", [("s2", "b", 2), ("s1", "a", 1)])
    modern = _sig(home, monkeypatch)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 0))
    legacy = _sig(home, monkeypatch)
    assert legacy == modern and legacy[0] != "mtime-fallback"
