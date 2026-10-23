"""KC: ``SQLiteSessionMaintenance.session_projection_digest`` signs the sessions table in one SQLite statement (the TUI
change watcher's per-store probe), and the digest still moves for every column-level change of a projected field.

The store is the backend's own, seeded through its session write; column values no verb writes (NULL vs '', a BLOB,
invalid UTF-8) go in through the raw holder.
"""

from __future__ import annotations

import sqlite3

import pytest

from persistence.housekeeping.maintenance import SESSION_PROJECTION_FIELDS
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.housekeeping import maintenance as maintenance_module
from tests.fixtures.storage_holder import raw_holder


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


def _titles(store, rows):
    """Each ``(session_id, title)`` row's session created when missing, then its title set as given."""
    holder = raw_holder(store)
    with holder._read_ctx() as conn:
        present = {row[0] for row in conn.execute("SELECT id FROM sessions")}
    for sid, _title in rows:
        if sid not in present:
            create_session(holder, sid, "cli")
    holder._execute_write(lambda conn: conn.executemany("UPDATE sessions SET title = ? WHERE id = ?",
                                                        [(title, sid) for sid, title in rows]))


def _digest(store):
    digest = store.maintenance.session_projection_digest()
    assert digest is not None and digest[0] == SESSION_PROJECTION_FIELDS
    return digest


@pytest.mark.parametrize(
    "a, b",
    [
        ([("s1", None)], [("s1", "")]),             # NULL vs empty string
        ([("s1", "ab")], [("s1", "a")]),            # value change
        ([("s1", "x")], [("s1", b"x")]),            # text vs BLOB with the same bytes
        ([("s1", "x")], [("s1", "x"), ("s2", "y")]),  # row added (titles are unique)
        ([("s1", "a\x1fb")], [("s1", "a")]),        # separator char inside a value
    ],
)
def test_digest_moves_for_column_level_changes(store, a, b):
    _titles(store, a)
    before = _digest(store)
    _titles(store, b)
    assert _digest(store) != before


def _reinsert_in_reverse_id_order(store):
    """The same rows, physically stored in the reverse order (rowid order no longer follows id)."""
    def _do(conn):
        conn.execute("CREATE TEMP TABLE kc_sessions_copy AS SELECT * FROM sessions")
        conn.execute("DELETE FROM sessions")
        conn.execute("INSERT INTO sessions SELECT * FROM temp.kc_sessions_copy ORDER BY id DESC")
        conn.execute("DROP TABLE temp.kc_sessions_copy")
    raw_holder(store)._execute_write(_do)


def test_digest_is_stable_and_independent_of_insert_order(store):
    _titles(store, [("s1", "a"), ("s2", "b")])
    first = _digest(store)
    assert _digest(store) == first
    _reinsert_in_reverse_id_order(store)
    with raw_holder(store)._read_ctx() as conn:
        assert [row[0] for row in conn.execute("SELECT id FROM sessions ORDER BY rowid")] == ["s2", "s1"]
    assert _digest(store) == first


def test_empty_sessions_table_has_a_digest(store):
    with raw_holder(store)._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    _digest(store)


def test_invalid_utf8_title_still_gets_a_moving_digest(store):
    _titles(store, [("s1", "a")])
    holder = raw_holder(store)
    holder._execute_write(lambda conn: conn.execute("UPDATE sessions SET title = CAST(x'ff61' AS TEXT)"))
    bad = _digest(store)
    holder._execute_write(lambda conn: conn.execute("UPDATE sessions SET title = CAST(x'ff62' AS TEXT)"))
    assert _digest(store) != bad
    # The shared read connection's own text factory is restored after the probe.
    with holder._read_ctx() as conn:
        assert conn.text_factory is not bytes


def test_pre_3_44_sqlite_uses_the_ordered_subquery(store, monkeypatch):
    _titles(store, [("s2", "b"), ("s1", "a")])
    modern = _digest(store)
    monkeypatch.setattr(maintenance_module.sqlite3, "sqlite_version_info", (3, 40, 0))
    assert sqlite3.sqlite_version_info == (3, 40, 0)
    assert _digest(store) == modern
