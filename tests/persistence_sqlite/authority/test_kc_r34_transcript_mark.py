"""KC rollback window: every claim stores its transcript mark in KC r34's column.

KC release r34 declares a nullable ``session_admissions.transcript_mark`` column with each claim's mark, and its
restart recovery reads it (NULL reads as "claimed before the column existed"). Upstream computes the mark only
on the returned claim. If the fleet rolls back to r34, a row claimed under this release must carry its real mark,
on an r34 store and on a store created after r34 alike; a returned claim is unclaimed again (NULL).
"""
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, return_session_claim,
)
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session


def _columns(holder):
    return holder._execute_write(lambda conn: [c[1] for c in conn.execute('PRAGMA table_info(session_admissions)')])


def _stored_mark(holder, admission_id):
    return holder._execute_write(lambda conn: conn.execute(
        'SELECT transcript_mark FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()[0])


def _seed_messages(holder, n):
    def write(conn):
        for _ in range(n):
            conn.execute("INSERT INTO messages(session_id, role, content, timestamp) VALUES ('s', 'user', 'x', 0)")
    holder._execute_write(write)


def _admit_and_claim(holder, request_id='input', *, epoch=None):
    epoch = epoch if epoch is not None else begin_runtime_epoch(holder, instance_id='boot')
    accepted, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s',
                                      request_id=request_id, payload={'text': 'work'})
    return accepted, claim_session_input(holder, epoch=epoch, session_id='s'), epoch


def test_a_new_store_declares_the_r34_column_and_a_claim_writes_its_real_mark(backend, scope_a):
    holder = backend.open(scope_a).holder()
    assert 'transcript_mark' in _columns(holder)
    create_session(holder, 's', 'test')
    _seed_messages(holder, 3)
    accepted, claim, _ = _admit_and_claim(holder)
    assert _stored_mark(holder, accepted['admission_id']) is not None
    assert claim['status'] == 'started' and claim['transcript_mark'] > 0
    assert _stored_mark(holder, accepted['admission_id']) == claim['transcript_mark']


def test_a_returned_claim_is_unclaimed_again(backend, scope_a):
    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'test')
    _seed_messages(holder, 1)
    accepted, claim, epoch = _admit_and_claim(holder)
    assert _stored_mark(holder, accepted['admission_id']) == claim['transcript_mark']
    return_session_claim(holder, epoch=epoch, admission_id=accepted['admission_id'],
                         generation=claim['generation'], queued_owner_epoch=claim['queued_owner_epoch'])
    assert _stored_mark(holder, accepted['admission_id']) is None


def test_a_missing_declared_column_cannot_return_an_undurable_claim(backend, scope_a):
    import pytest
    import sqlite3

    holder = backend.open(scope_a).holder()
    holder._execute_write(lambda conn: conn.execute('ALTER TABLE session_admissions DROP COLUMN transcript_mark'))
    create_session(holder, 's', 'test')
    with pytest.raises(sqlite3.OperationalError, match='no such column'):
        _admit_and_claim(holder)


def test_storage_failure_rolls_back_the_whole_claim_and_retry_keeps_the_mark(backend, scope_a, monkeypatch):
    import sqlite3
    import pytest

    holder = backend.open(scope_a).holder()
    create_session(holder, 's', 'test')
    _seed_messages(holder, 3)
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    accepted, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s',
                                      request_id='input', payload={'text': 'work'})
    original = holder._execute_write

    class FaultConnection:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, parameters=()):
            if sql.startswith('UPDATE session_admissions SET') and 'transcript_mark' in sql:
                # Model SQLite's automatic rollback on a storage failure, not just an ordinary exception.
                self.conn.execute('ROLLBACK')
                exc = sqlite3.OperationalError('database or disk is full')
                exc.sqlite_errorcode = sqlite3.SQLITE_FULL
                raise exc
            return self.conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(holder, '_execute_write', lambda fn: original(lambda conn: fn(FaultConnection(conn))))
    with pytest.raises(sqlite3.OperationalError, match='disk is full'):
        claim_session_input(holder, epoch=epoch, session_id='s')
    monkeypatch.setattr(holder, '_execute_write', original)
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT status, generation, transcript_mark FROM session_admissions WHERE admission_id=?',
                           (accepted['admission_id'],)).fetchone()
        assert tuple(row) == ('queued', None, None)
        assert conn.execute("SELECT runtime_generation FROM sessions WHERE id='s'").fetchone()[0] == 0
    claim = claim_session_input(holder, epoch=epoch, session_id='s')
    assert claim['status'] == 'started' and claim['transcript_mark'] > 0
    assert _stored_mark(holder, accepted['admission_id']) == claim['transcript_mark']
