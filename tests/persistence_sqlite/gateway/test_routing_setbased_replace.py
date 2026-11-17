"""KC: the SQLite ``RoutingRepository.replace_routing_entries`` is set-based.

The routing index rewrite used to run one ``executemany`` row (and one fetch step) per route.
Each row releases and re-takes the GIL, so in a busy multiplex gateway the rewrite of ~6k
routes held the state.db writer mutex for 20-60 s (a GIL convoy) and starved every other
writer into ``WriterMutexTimeout``. These tests pin (1) exact parity with the old per-row
semantics, including every retired-route and fence case, and (2) a statement count that does
not grow with the size of the index.
"""
import json
import random
import sqlite3
import time

import pytest

from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.core.errors import CorruptStoreError
from persistence.sqlite import _registry
from persistence.sqlite._backend import translated
from persistence.sqlite.conversation._store.retirement import RETIRED_PREFIX
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.gateway._routes import (
    ROUTE_FENCE_PREFIX, ROUTE_RETIRED_SQL, route_retired_params, route_retired_sql)
from persistence.sqlite.gateway import routing as routing_module

SCOPE = '/tmp/sessions'


@pytest.fixture()
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def pair(backend, scope_a, scope_b):
    """Two stores: the legacy oracle's and the set-based rewrite's."""
    return backend.open(scope_a), backend.open(scope_b)


def _legacy_replace(store, entries, *, scope):
    """The pre-KC per-row implementation, kept verbatim as the parity oracle, under the repository's own
    error translation (``translated``), as ``replace_routing_entries`` runs it."""
    with translated():
        return _legacy_replace_raw(store, entries, scope=scope)


def _legacy_replace_raw(store, entries, *, scope):
    now = time.time()
    rows = [{'scope': scope, 'key': k, **route_retired_params(k, v, now)} for k, v in entries.items() if k and v]

    def _do(conn):
        conn.execute("DELETE FROM gateway_routing WHERE scope = ?", (scope,))
        if rows:
            conn.executemany(
                "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                "SELECT :scope, :key, :entry, :now WHERE NOT " + ROUTE_RETIRED_SQL,
                rows)
        kept = {r[0] for r in conn.execute("SELECT session_key FROM gateway_routing WHERE scope = ?", (scope,))}
        return {row['key'] for row in rows} - kept
    return store.holder()._execute_write(_do)


def _rows(store, scope=None):
    sql = "SELECT scope, session_key, entry_json FROM gateway_routing"
    args = ()
    if scope is not None:
        sql += " WHERE scope = ?"
        args = (scope,)
    with store.holder()._read_ctx() as conn:
        return sorted(tuple(r) for r in conn.execute(sql + " ORDER BY scope, session_key", args))


def _meta(store, key, value):
    store.holder()._execute_write(lambda c: c.execute(
        "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)", (key, value)))


def _entry(sid, **extra):
    return json.dumps({'session_id': sid, **extra})


# The single-row predicate exactly as it stood before the template (901e0d25154).
_LEGACY_ROUTE_RETIRED_SQL = """(
    EXISTS (SELECT 1 FROM state_meta m
             WHERE m.key = :retired_prefix || json_extract(:entry, '$.session_id')
               AND (:ns = 'main' OR NOT CASE WHEN json_valid(m.value)
                                             THEN json_extract(m.value, '$.own') IS NOT NULL ELSE 0 END))
    OR EXISTS (SELECT 1 FROM state_meta f
                WHERE f.key = :fence_prefix || :ns || ':' || json_extract(:entry, '$.session_id')
                  AND NOT CASE WHEN json_valid(f.value)
                               THEN coalesce(json_extract(f.value, '$.expires')
                                             AND json_extract(f.value, '$.retired_at') < :now - :horizon, 0)
                               ELSE 0 END
                  AND (:ns != 'main' OR NOT EXISTS (
                          SELECT 1 FROM sessions s WHERE s.id = json_extract(:entry, '$.session_id')))))"""


def test_template_reproduces_the_single_row_predicate_text():
    assert ROUTE_RETIRED_SQL == _LEGACY_ROUTE_RETIRED_SQL
    assert ROUTE_RETIRED_SQL == route_retired_sql("json_extract(:entry, '$.session_id')", ':ns')
    assert '@SID@' not in ROUTE_RETIRED_SQL and '@NS@' not in ROUTE_RETIRED_SQL


def _seed_retirement_cases(store):
    """Every branch of ROUTE_RETIRED_SQL, for main and secondary namespaces."""
    now = time.time()
    create_session(store.holder(), 'live-main', 'slack')
    create_session(store.holder(), 'fenced-main-but-held', 'slack')
    # Own-store retirement marker: fences a main route; a scoped marker (own) does not fence a
    # secondary route, an unscoped one (no own) fences every route, a malformed one fences too.
    _meta(store, RETIRED_PREFIX + 'ret-main', json.dumps({'own': 'main'}))
    _meta(store, RETIRED_PREFIX + 'ret-scoped', json.dumps({'own': 'main'}))
    _meta(store, RETIRED_PREFIX + 'ret-unscoped', json.dumps({'retired_at': now}))
    _meta(store, RETIRED_PREFIX + 'ret-malformed', 'not json{')
    # Secondary-profile fences: permanent (delete), fresh sweep, aged-out sweep, malformed.
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:fence-perm', json.dumps({'retired_at': now}))
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:fence-fresh', json.dumps({'expires': True, 'retired_at': now}))
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:fence-old',
          json.dumps({'expires': True, 'retired_at': now - RETRY_HORIZON_S - 60}))
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:fence-bad', '{{{')
    # A main-namespace fence only applies when this store has no session with that id.
    _meta(store, ROUTE_FENCE_PREFIX + 'main:fenced-main-absent', json.dumps({'retired_at': now}))
    _meta(store, ROUTE_FENCE_PREFIX + 'main:fenced-main-but-held', json.dumps({'retired_at': now}))
    # Effective lookups of the typed ids in CASES (json_extract yields 5, 1, and JSON text).
    _meta(store, RETIRED_PREFIX + '5', json.dumps({'own': 'main'}))
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:1', json.dumps({'retired_at': now}))
    _meta(store, RETIRED_PREFIX + '["ret-main"]', json.dumps({'retired_at': now}))
    _meta(store, ROUTE_FENCE_PREFIX + 'scout:{"id":"fence-perm"}', json.dumps({'retired_at': now}))


CASES = {
    'agent:main:slack:dm:1': _entry('live-main'),
    'agent:main:slack:dm:2': _entry('ret-main'),
    'agent:scout:slack:dm:3': _entry('ret-scoped'),
    'agent:scout:slack:dm:4': _entry('ret-unscoped'),
    'agent:brand:local:5': _entry('ret-malformed'),
    'agent:scout:slack:dm:6': _entry('fence-perm'),
    'agent:scout:slack:dm:7': _entry('fence-fresh'),
    'agent:scout:slack:dm:8': _entry('fence-old'),
    'agent:scout:slack:dm:9': _entry('fence-bad'),
    'agent:main:slack:dm:10': _entry('fenced-main-absent'),
    'agent:main:slack:dm:11': _entry('fenced-main-but-held'),
    'legacy-key-without-shape': _entry('ret-main'),
    'agent:dev:local:12': _entry('nobody', title='plain'),
    'agent:main:slack:dm:13': json.dumps({'title': 'no session id'}),
    'agent:scout:slack:dm:14': json.dumps({'session_id': None}),
    # Typed (non-text) ids against own markers and fences, held and absent.
    'agent:main:slack:dm:15': json.dumps({'session_id': 5}),
    'agent:scout:slack:dm:16': json.dumps({'session_id': True}),
    'agent:main:slack:dm:17': json.dumps({'session_id': ['ret-main']}),
    'agent:scout:slack:dm:18': json.dumps({'session_id': {'id': 'fence-perm'}}),
    # Transport: BMP, astral, control and NUL characters in keys and entries survive verbatim.
    'agent:main:slack:dm:é中': _entry('ünï'),
    'agent:main:slack:dm:😀': json.dumps({'session_id': 'astral-😀'}, ensure_ascii=False),
    'agent:main:slack:dm:ctl\t\x01': json.dumps({'session_id': 'ctl\x01', 'n': '\x00x'}),
    'agent:main:slack:dm:nul\x00tail': _entry('nul-sid'),
}


def test_malformed_entry_json_fails_the_whole_rewrite_like_legacy(pair):
    legacy, setbased = pair
    good = {'agent:main:slack:1': _entry('x')}
    for store in (legacy, setbased):
        store.routing.replace_routing_entries(good, routing_scope=SCOPE)
    bad = dict(good, **{'agent:main:slack:2': 'not json{'})
    with pytest.raises(Exception) as legacy_err:
        _legacy_replace(legacy, bad, scope=SCOPE)
    with pytest.raises(Exception) as new_err:
        setbased.routing.replace_routing_entries(bad, routing_scope=SCOPE)
    assert type(new_err.value) is type(legacy_err.value)
    # Atomic: the failed rewrite left the previous index intact in both.
    assert _rows(setbased) == _rows(legacy) == [(SCOPE, 'agent:main:slack:1', _entry('x'))]


def test_parity_with_legacy_on_every_retirement_branch(pair):
    legacy, setbased = pair
    for store in (legacy, setbased):
        _seed_retirement_cases(store)
    retired_legacy = _legacy_replace(legacy, CASES, scope=SCOPE)
    retired_new = setbased.routing.replace_routing_entries(CASES, routing_scope=SCOPE)
    assert retired_new == retired_legacy
    assert _rows(setbased) == _rows(legacy)
    # Sanity: the cases really exercise both outcomes.
    assert retired_new and len(retired_new) < len(CASES)


def test_parity_over_randomized_rewrite_sequences(pair):
    rnd = random.Random(1729)
    legacy, setbased = pair
    for store in (legacy, setbased):
        _seed_retirement_cases(store)
        store.routing.replace_routing_entries({'agent:main:other:1': _entry('live-main')}, routing_scope='other')
    pool = list(CASES.items()) + [
        (f'agent:{rnd.choice(["main", "scout", "brand"])}:slack:dm:r{i}', _entry(f'sid-{i}'))
        for i in range(300)]
    for _round in range(12):
        chosen = dict(rnd.sample(pool, rnd.randint(0, len(pool))))
        for key in rnd.sample(sorted(chosen), min(5, len(chosen))):
            chosen[key] = _entry(json.loads(chosen[key])['session_id'], rev=rnd.random())
        chosen[''] = _entry('skipped-empty-key')
        chosen['agent:main:slack:empty-value'] = ''
        assert setbased.routing.replace_routing_entries(chosen, routing_scope=SCOPE) == \
            _legacy_replace(legacy, chosen, scope=SCOPE)
        assert _rows(setbased) == _rows(legacy)
    # The other scope is never touched by either implementation.
    assert _rows(setbased, 'other') == [('other', 'agent:main:other:1', _entry('live-main'))]


def test_empty_batch_clears_only_its_scope(store):
    store.routing.replace_routing_entries({'agent:main:a': _entry('x')}, routing_scope=SCOPE)
    store.routing.replace_routing_entries({'agent:main:b': _entry('y')}, routing_scope='other')
    assert store.routing.replace_routing_entries({}, routing_scope=SCOPE) == set()
    assert _rows(store, SCOPE) == []
    assert _rows(store, 'other') == [('other', 'agent:main:b', _entry('y'))]


def _stamps(store):
    with store.holder()._read_ctx() as conn:
        return dict(conn.execute("SELECT session_key, updated_at FROM gateway_routing WHERE scope = ?", (SCOPE,)))


def test_unchanged_rows_are_not_rewritten(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(routing_module.time, 'time', lambda: clock[0])
    entries = {f'agent:main:slack:{i}': _entry(f's{i}') for i in range(50)}
    store.routing.replace_routing_entries(entries, routing_scope=SCOPE)
    before = _stamps(store)
    clock[0] = 2000.0
    changed = dict(entries)
    changed['agent:main:slack:7'] = _entry('s7', rev=2)
    store.routing.replace_routing_entries(changed, routing_scope=SCOPE)
    after = _stamps(store)
    assert {k for k in after if after[k] != before[k]} == {'agent:main:slack:7'}


def test_a_route_that_becomes_retired_is_removed(store):
    store.routing.replace_routing_entries({'agent:main:slack:1': _entry('gone')}, routing_scope=SCOPE)
    _meta(store, RETIRED_PREFIX + 'gone', json.dumps({'own': 'main'}))
    assert store.routing.replace_routing_entries({'agent:main:slack:1': _entry('gone')}, routing_scope=SCOPE) == \
        {'agent:main:slack:1'}
    assert _rows(store, SCOPE) == []


@pytest.mark.parametrize('size', [10, 3000])
def test_statement_count_does_not_grow_with_the_index(store, size):
    """The GIL-convoy guard: the rewrite is a fixed handful of sqlite3 statements."""
    entries = {f'agent:main:slack:dm:{i}': _entry(f'sid-{i}', n=i) for i in range(size)}
    statements = []
    holder = store.holder()
    with holder._lock:
        holder._conn.set_trace_callback(statements.append)
    try:
        store.routing.replace_routing_entries(entries, routing_scope=SCOPE)
    finally:
        with holder._lock:
            holder._conn.set_trace_callback(None)
    routing = [s for s in statements if 'gateway_routing' in s]
    assert len(routing) <= 3, routing[:5]
    # Every statement the rewrite runs, temp-table steps included, is a fixed handful.
    assert len(statements) <= 10, [s[:80] for s in statements]
    assert len(_rows(store, SCOPE)) == size


class _CountingCursor:
    def __init__(self, cursor, counter):
        self._cursor, self._counter = cursor, counter

    def __iter__(self):
        for row in self._cursor:
            self._counter[0] += 1
            yield row

    def fetchone(self):
        row = self._cursor.fetchone()
        self._counter[0] += row is not None
        return row

    def fetchall(self):
        rows = self._cursor.fetchall()
        self._counter[0] += len(rows)
        return rows

    def fetchmany(self, *args, **kwargs):
        rows = self._cursor.fetchmany(*args, **kwargs)
        self._counter[0] += len(rows)
        return rows

    def execute(self, *args, **kwargs):
        self._cursor.execute(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _CountingConn:
    """Counts result rows Python steps through; a per-row fetch would scale with the index."""
    def __init__(self, conn, counter):
        self._conn, self._counter = conn, counter

    def execute(self, *args, **kwargs):
        return _CountingCursor(self._conn.execute(*args, **kwargs), self._counter)

    def executemany(self, *args, **kwargs):
        raise AssertionError('executemany steps once per row; the rewrite must stay set-based')

    def cursor(self, *args, **kwargs):
        return _CountingCursor(self._conn.cursor(*args, **kwargs), self._counter)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.mark.parametrize('size', [10, 3000])
def test_rows_fetched_do_not_grow_with_the_index(store, size):
    """The fetch side of the GIL-convoy guard (the old code stepped one result row per route)."""
    entries = {f'agent:main:slack:dm:{i}': _entry(f'sid-{i}', n=i) for i in range(size)}
    counter = [0]
    holder = store.holder()
    with holder._lock:
        real = holder._conn
        holder._conn = _CountingConn(real, counter)
    try:
        store.routing.replace_routing_entries(entries, routing_scope=SCOPE)
    finally:
        with holder._lock:
            holder._conn = real
    assert counter[0] <= 2, counter[0]
    assert len(_rows(store, SCOPE)) == size


@pytest.mark.parametrize('bad', [{'agent:main:x': {'session_id': 'x'}}, {'agent:main:x': 5}, {7: _entry('x')}])
def test_non_string_routes_are_rejected_before_any_write(store, bad):
    store.routing.replace_routing_entries({'agent:main:keep': _entry('k')}, routing_scope=SCOPE)
    with pytest.raises(TypeError):
        store.routing.replace_routing_entries(bad, routing_scope=SCOPE)
    assert _rows(store, SCOPE) == [(SCOPE, 'agent:main:keep', _entry('k'))]


@pytest.mark.parametrize('bad', [{'agent:main:x': '{"session_id": "\ud800"}'}, {'agent:main:\ud800': _entry('x')}])
def test_lone_surrogate_aborts_the_rewrite_like_legacy(pair, bad):
    legacy, setbased = pair
    for store in (legacy, setbased):
        store.routing.replace_routing_entries({'agent:main:keep': _entry('k')}, routing_scope=SCOPE)
    with pytest.raises(UnicodeEncodeError):
        _legacy_replace(legacy, bad, scope=SCOPE)
    with pytest.raises(UnicodeEncodeError):
        setbased.routing.replace_routing_entries(bad, routing_scope=SCOPE)
    assert _rows(setbased) == _rows(legacy) == [(SCOPE, 'agent:main:keep', _entry('k'))]


def test_returns_a_plain_mutable_set(store):
    store.routing.replace_routing_entries({'agent:main:a': _entry('a')}, routing_scope=SCOPE)
    for entries in ({}, {'agent:main:a': _entry('a')}):
        result = store.routing.replace_routing_entries(entries, routing_scope=SCOPE)
        assert type(result) is set


def test_lone_surrogate_is_rejected_before_taking_the_writer_mutex(store, monkeypatch):
    def must_not_write(*args, **kwargs):
        raise AssertionError('writer mutex taken for an unencodable batch')
    monkeypatch.setattr(store.holder(), '_execute_write', must_not_write)
    with pytest.raises(UnicodeEncodeError):
        store.routing.replace_routing_entries({'agent:main:x': '{"session_id": "\ud800"}'}, routing_scope=SCOPE)


def _batch_table_count(holder, sql):
    with holder._lock:
        return holder._conn.execute(sql).fetchone()[0]


def test_first_ever_rewrite_failing_rolls_back_the_temp_table_and_recovers(store):
    # SQLite's "malformed JSON" reaches the caller translated, as the per-row rewrite's did.
    with pytest.raises(CorruptStoreError):
        store.routing.replace_routing_entries({'agent:main:x': 'not json{'}, routing_scope=SCOPE)
    assert _rows(store, SCOPE) == []
    assert _batch_table_count(
        store.holder(), "SELECT count(*) FROM temp.sqlite_master WHERE name = 'kc_routing_batch'") == 0
    assert store.routing.replace_routing_entries({'agent:main:x': _entry('x')}, routing_scope=SCOPE) == set()
    assert _rows(store, SCOPE) == [(SCOPE, 'agent:main:x', _entry('x'))]


def test_callback_retry_replays_cleanly_through_execute_write(store, monkeypatch):
    """The holder's _execute_write retries the WHOLE callback on a lock error; the rewrite must be idempotent."""
    store.routing.replace_routing_entries({'agent:main:old': _entry('o')}, routing_scope=SCOPE)
    holder = store.holder()
    real = holder._execute_write
    calls = []

    def flaky(fn, *args, **kwargs):
        def once(conn):
            calls.append(1)
            result = fn(conn)
            if len(calls) == 1:
                # The raw driver error the holder's own retry loop sees, after the full body ran.
                raise sqlite3.OperationalError('database is locked')
            return result
        return real(once, *args, **kwargs)

    monkeypatch.setattr(holder, '_execute_write', flaky)
    entries = {'agent:main:a': _entry('a'), 'agent:main:b': _entry('b')}
    assert store.routing.replace_routing_entries(entries, routing_scope=SCOPE) == set()
    assert len(calls) == 2
    assert _rows(store, SCOPE) == [(SCOPE, 'agent:main:a', _entry('a')), (SCOPE, 'agent:main:b', _entry('b'))]
    assert _batch_table_count(holder, "SELECT count(*) FROM temp.kc_routing_batch") == 0


def test_timestamps_exact_for_unchanged_changed_and_new_rows(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(routing_module.time, 'time', lambda: clock[0])
    store.routing.replace_routing_entries({'agent:main:a': _entry('a'), 'agent:main:b': _entry('b')}, routing_scope=SCOPE)
    clock[0] = 2000.0
    store.routing.replace_routing_entries(
        {'agent:main:a': _entry('a'), 'agent:main:b': _entry('b', rev=2), 'agent:main:c': _entry('c')}, routing_scope=SCOPE)
    assert _stamps(store) == {'agent:main:a': 1000.0, 'agent:main:b': 2000.0, 'agent:main:c': 2000.0}


def test_rewrite_leaves_no_batch_rows_behind_and_survives_reopen(store, backend, scope_a):
    store.routing.replace_routing_entries({'agent:main:a': _entry('x')}, routing_scope=SCOPE)
    assert _batch_table_count(store.holder(), "SELECT count(*) FROM temp.kc_routing_batch") == 0
    store.holder().close()
    _registry.close_all()
    reopened = backend.open(scope_a)
    assert reopened.routing.replace_routing_entries({'agent:main:b': _entry('y')}, routing_scope=SCOPE) == set()
    assert _rows(reopened, SCOPE) == [(SCOPE, 'agent:main:b', _entry('y'))]
