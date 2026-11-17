"""KC: the SQLite store's route retirement (``retire_routes``) and shared-route fencing (``fence_shared_routes``) are
set-based.

``retire_routes`` and ``fence_shared_routes`` used to scan every routing row in Python under the
state.db writer mutex (one fetch step and one json.loads per row), the same GIL-convoy shape that
held the mutex for 20-60 s in the busy multiplex gateway. They now issue one DELETE each. These
tests pin parity with the verbatim per-row implementations and the SQL twins of the helpers.
"""
import json
import random
import sqlite3
import time

import pytest

from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.gateway.routing import route_namespace
from persistence.sqlite.conversation._store import retirement as retirement_module
from persistence.sqlite.conversation._store.retirement import (
    RETIRED_PREFIX,
    VIEWER_CLOSE_PREFIX,
    entry_session_id,
    entry_session_id_sql,
    retire_routes,
)
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.gateway._routes import (
    ROUTE_FENCE_PREFIX,
    _reclaim_expired,
    fence_shared_routes,
    route_fence_key,
    route_namespace_sql,
)


@pytest.fixture()
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def pair(backend, scope_a, scope_b):
    """Two stores: the legacy oracle's and the set-based implementation's."""
    return backend.open(scope_a), backend.open(scope_b)


def _write(store, fn):
    return store.holder()._execute_write(fn)


KEYS = ['agent:main:slack:dm:1', 'agent:scout:slack:dm:2', 'agent:brand:local:dm:3', 'agent::x:y', 'agent:x',
        'Agent:scout:slack:1', 'agent:scout', 'agent:scout:', 'plain', '', ':scout:x', 'agent:sc:out:z',
        'agent:ünï:slack:1', 'agent:main:', 'agent:%:x', 'agent:_:x', 'xagent:scout:y',
        'agent:sc\x00out:x', 'agent:scout\x00:x', 'agent:scout:x\x00y', 'agent:\x00:x', 'agent:😀:x']


@pytest.mark.parametrize('key', KEYS)
def test_route_namespace_sql_matches_python(key):
    conn = sqlite3.connect(':memory:')
    assert conn.execute(f'SELECT {route_namespace_sql("v")} FROM (SELECT ? AS v)', (key,)).fetchone()[0] == \
        route_namespace(key).encode()


def test_route_namespace_sql_matches_python_randomized():
    rnd = random.Random(7)
    alphabet = ['agent', ':', 'a', 'main', 'scout', '', 'x', 'é', '%', '\x00', '😀']
    conn = sqlite3.connect(':memory:')
    for _ in range(3000):
        key = ''.join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 7)))
        assert conn.execute(f'SELECT {route_namespace_sql("v")} FROM (SELECT ? AS v)', (key,)).fetchone()[0] == \
        route_namespace(key).encode(), key


ENTRIES = [
    json.dumps({'session_id': 's1'}), json.dumps({'session_id': 's2', 'x': [1, {'session_id': 'nested'}]}),
    json.dumps({'title': 'none'}), json.dumps({'session_id': None}), json.dumps(['s1']), json.dumps('s1'),
    'not json{', '', json.dumps({'session_id': 's1 '}), json.dumps({'session_id': 'S1'}),
    '{"session_id": "other", "session_id": "s1"}', '{"session_id": "s1", "session_id": "other"}',
    '{"session\\u005fid": "s1"}', '{"session_id": "s1", "cost": NaN}', '{"session_id": "s1", "c": -Infinity}',
    json.dumps({'session_id': ['s1']}), json.dumps({'session_id': {'a': 1}}), json.dumps({'session_id': 5}),
    json.dumps({'session_id': True}), json.dumps({'session_id': 'a\x00b'}), json.dumps({'session_id': '😀é'}),
    json.dumps({'session_id': 'x'}, ensure_ascii=False), json.dumps({'x': {'session_id': 'nested'}}),
]


@pytest.mark.parametrize('raw', ENTRIES)
def test_entry_session_id_sql_matches_python(raw):
    """Text ids match the Python twin exactly (last duplicate wins, escaped keys, NaN); a non-text
    id is NULL in SQL (the twin returns it and the old scan raised on the unhashable ones)."""
    conn = sqlite3.connect(':memory:')
    expected = entry_session_id(raw)
    expected = expected if isinstance(expected, str) else None
    assert conn.execute(f'SELECT {entry_session_id_sql("v")} FROM (SELECT ? AS v)', (raw,)).fetchone()[0] == expected


def test_json_guards_are_nested_cases_not_and():
    """json_type/json_each must sit inside CASE WHEN json_valid(...): AND does not short-circuit."""
    sql = entry_session_id_sql('x')
    assert sql.startswith("(CASE WHEN json_valid(x, 2) THEN CASE WHEN json_type(x) = 'object' THEN ")
    assert sql.index('json_valid(x, 2)') < sql.index('json_type(x)') < sql.index('json_each(x)')
    assert ' AND ' not in sql.split('json_each')[0]


def test_entry_session_id_sql_falls_back_on_sqlite_before_3_45(monkeypatch):
    monkeypatch.setattr(retirement_module.sqlite3, 'sqlite_version_info', (3, 44, 2))
    sql = retirement_module.entry_session_id_sql('x')
    assert 'json_valid(x)' in sql and 'json_valid(x, 2)' not in sql
    conn = sqlite3.connect(':memory:')
    assert conn.execute(f'SELECT {sql} FROM (SELECT ? AS x)', ('{"session_id": "s1"}',)).fetchone()[0] == 's1'


def test_fence_accepts_a_none_namespace_like_legacy(pair):
    old, new = pair
    for store in (old, new):
        _seed(store, random.Random(21), n=60)
    _write(old, lambda c: _legacy_fence_shared_routes(c, None, ['s1', 's2']))
    _write(new, lambda c: fence_shared_routes(c, None, ['s1', 's2']))
    assert _meta(new) == _meta(old)
    assert _rows(new) == _rows(old)


def test_structured_ids_never_collide_with_a_text_target(store):
    rows = [('', 'agent:scout:slack:1', json.dumps({'session_id': ['s1']}), 0.0),
            ('', 'agent:scout:slack:2', json.dumps({'session_id': {'a': 1}}), 0.0)]
    _write(store, lambda c: c.executemany('INSERT INTO gateway_routing VALUES (?,?,?,?)', rows))
    _write(store, lambda c: retire_routes(c, ['["s1"]', '{"a":1}', '{"a": 1}']))
    _write(store, lambda c: fence_shared_routes(c, 'scout', ['["s1"]', '{"a": 1}']))
    assert [r[1] for r in _rows(store)] == ['agent:scout:slack:1', 'agent:scout:slack:2']


def test_nul_in_namespace_is_not_reclassified_as_main(store):
    rows = [('', 'agent:other\x00:x', json.dumps({'session_id': 's1'}), 0.0),
            ('', 'agent:main:x', json.dumps({'session_id': 's1'}), 0.0)]
    _write(store, lambda c: c.executemany('INSERT INTO gateway_routing VALUES (?,?,?,?)', rows))
    _write(store, lambda c: fence_shared_routes(c, 'scout', ['s1']))  # s1 not held: fences scout + main
    assert [r[1] for r in _rows(store)] == ['agent:other\x00:x']


def _legacy_retire_scan(conn, session_ids):
    targets = set(session_ids)
    for row in conn.execute('SELECT scope,session_key,entry_json FROM gateway_routing').fetchall():
        if entry_session_id(row['entry_json']) in targets:
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=?',
                         (row['scope'], row['session_key']))


def _legacy_fence_scan(conn, owned):
    for row in conn.execute('SELECT scope,session_key,entry_json FROM gateway_routing').fetchall():
        if route_namespace(row['session_key']) in owned.get(entry_session_id(row['entry_json']), ()):
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=?',
                         (row['scope'], row['session_key']))


def _seed(store, rnd, n=400):
    rows = []
    for i in range(n):
        key = rnd.choice(['agent:main:slack:dm:', 'agent:scout:slack:dm:', 'agent:brand:local:dm:', 'odd:'])
        sid = rnd.choice(['s1', 's2', 's3', f's{i}'])
        entry = rnd.choice([json.dumps({'session_id': sid, 'n': i}), json.dumps({'n': i}), 'bad{', json.dumps([sid])])
        rows.append((rnd.choice(['', '/x/sessions']), f'{key}{i}', entry, 0.0))
    _write(store, lambda c: c.executemany('INSERT INTO gateway_routing VALUES (?,?,?,?)', rows))


def _rows(store):
    with store.holder()._read_ctx() as conn:
        return sorted(tuple(r) for r in conn.execute('SELECT scope, session_key, entry_json FROM gateway_routing'))


def test_retire_routes_parity_with_per_row_scan(pair):
    rnd = random.Random(11)
    old, new = pair
    seed = rnd.random()
    _seed(old, random.Random(seed))
    _seed(new, random.Random(seed))
    targets = ['s1', 's3', 's17', 'missing']
    _write(old, lambda c: _legacy_retire_scan(c, targets))
    _write(new, lambda c: retire_routes(c, targets))
    assert _rows(new) == _rows(old)
    assert len(_rows(new)) < 400


def test_fence_shared_routes_parity_with_per_row_scan(pair):
    old, new = pair
    for store in (old, new):
        _seed(store, random.Random(5))
        create_session(store.holder(), 's2', 'slack')  # held here: main is NOT fenced for s2
    ids = ['s1', 's2', 's3', 's40']
    owned = {sid: ({'scout', 'main'} if sid != 's2' else {'scout'}) for sid in ids}
    _write(old, lambda c: _legacy_fence_scan(c, owned))
    _write(new, lambda c: fence_shared_routes(c, 'scout', ids))
    assert _rows(new) == _rows(old)


# --- Whole-function parity: the verbatim 901e0d25154 implementations (markers + routes) ---------

def _legacy_mark_retired(conn, key, *, expires, **fields):
    value = {'retired_at': time.time(), **fields}
    if expires:
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE '
                     "SET value=excluded.value WHERE json_valid(state_meta.value) "
                     "AND json_extract(state_meta.value, '$.expires')",
                     (key, json.dumps({**value, 'expires': True})))
    else:
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=CASE '
                     "WHEN NOT json_valid(state_meta.value) THEN json_remove(excluded.value, '$.own') "
                     "WHEN json_extract(state_meta.value, '$.own') IS NOT NULL "
                     "OR json_extract(state_meta.value, '$.expires') THEN excluded.value "
                     "ELSE json_remove(excluded.value, '$.own') END", (key, json.dumps(value)))


def _legacy_retire_routes(conn, session_ids, *, expiring=()):
    expiring = set(expiring)
    for sid in session_ids:
        _legacy_mark_retired(conn, RETIRED_PREFIX + sid, expires=sid in expiring, own=True)
        conn.execute('DELETE FROM state_meta WHERE key=?', ('gateway.api.settings.v1.' + sid,))
        conn.execute('DELETE FROM state_meta WHERE key=?', (VIEWER_CLOSE_PREFIX + sid,))
    _legacy_retire_scan(conn, session_ids)


def _legacy_fence_shared_routes(conn, namespace, session_ids, *, expiring=()):
    _reclaim_expired(conn, ROUTE_FENCE_PREFIX, time.time() - RETRY_HORIZON_S)
    expiring, owned = set(expiring), {}
    for sid in session_ids:
        names = {namespace}
        if not conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
            names.add('main')
        for name in names:
            _legacy_mark_retired(conn, route_fence_key(name, sid), expires=sid in expiring)
        owned[sid] = names
    _legacy_fence_scan(conn, owned)


IDS = ['s1', 's2', 's3', 's4', 's5', 's6', 'é7', 's1']  # duplicate on purpose


def _seed_markers(store):
    """Pre-existing markers of every shape the conflict rules distinguish."""
    old = time.time() - RETRY_HORIZON_S - 3600
    rows = {
        RETIRED_PREFIX + 's1': json.dumps({'retired_at': 1.0, 'own': True}),                    # scoped permanent
        RETIRED_PREFIX + 's2': json.dumps({'retired_at': 1.0}),                                 # unscoped permanent
        RETIRED_PREFIX + 's3': json.dumps({'retired_at': 1.0, 'expires': True, 'own': True}),   # sweep
        RETIRED_PREFIX + 's4': 'not json{',                                                     # malformed
        'gateway.api.settings.v1.s1': '{}', VIEWER_CLOSE_PREFIX + 's2': '{}',
        'gateway.api.settings.v1.keep': '{}',
        route_fence_key('scout', 's1'): json.dumps({'retired_at': 1.0}),
        route_fence_key('main', 's3'): json.dumps({'retired_at': 1.0, 'expires': True}),
        route_fence_key('scout', 's5'): 'bad{',
        route_fence_key('brand', 'old'): json.dumps({'retired_at': old, 'expires': True}),  # reclaimed
    }
    _write(store, lambda c: c.executemany('INSERT INTO state_meta(key, value) VALUES (?, ?)', rows.items()))


def _meta(store):
    """state_meta with the volatile retired_at stripped from markers this run wrote."""
    out = {}
    with store.holder()._read_ctx() as conn:
        for key, value in conn.execute('SELECT key, value FROM state_meta ORDER BY key'):
            if not (key.startswith(RETIRED_PREFIX) or key.startswith(ROUTE_FENCE_PREFIX)
                    or key.startswith('gateway.api.settings.v1.') or key.startswith(VIEWER_CLOSE_PREFIX)):
                continue
            try:
                parsed = json.loads(value)
            except ValueError:
                out[key] = value
                continue
            if isinstance(parsed, dict) and parsed.get('retired_at', 0) > 1e9:
                parsed['retired_at'] = 'NOW'
            out[key] = parsed
    return out


@pytest.mark.parametrize('expiring', [(), ('s1', 's3', 's5'), tuple(IDS)])
def test_retire_routes_full_parity_markers_and_routes(pair, expiring):
    old, new = pair
    for store in (old, new):
        _seed(store, random.Random(9))
        _seed_markers(store)
    _write(old, lambda c: _legacy_retire_routes(c, IDS, expiring=expiring))
    _write(new, lambda c: retire_routes(c, IDS, expiring=expiring))
    assert _meta(new) == _meta(old)
    assert _rows(new) == _rows(old)


@pytest.mark.parametrize('namespace', ['scout', 'main', 'brand'])
@pytest.mark.parametrize('expiring', [(), ('s1', 's3'), tuple(IDS)])
def test_fence_shared_routes_full_parity_markers_and_routes(pair, namespace, expiring):
    old, new = pair
    for store in (old, new):
        _seed(store, random.Random(13))
        create_session(store.holder(), 's2', 'slack')  # held: 'main' is not fenced for these
        create_session(store.holder(), 's6', 'slack')
        _seed_markers(store)
    _write(old, lambda c: _legacy_fence_shared_routes(c, namespace, IDS, expiring=expiring))
    _write(new, lambda c: fence_shared_routes(c, namespace, IDS, expiring=expiring))
    assert _meta(new) == _meta(old)
    assert _rows(new) == _rows(old)


def test_empty_id_lists_are_no_ops(store):
    _seed(store, random.Random(1), n=50)
    _seed_markers(store)
    before_rows, before_meta = _rows(store), _meta(store)
    _write(store, lambda c: retire_routes(c, []))
    assert (_rows(store), _meta(store)) == (before_rows, before_meta)


def test_nan_and_infinity_entries_are_retired_like_json_loads(store):
    rows = [('', 'agent:main:slack:1', '{"session_id": "s1", "cost": NaN}', 0.0),
            ('', 'agent:main:slack:2', '{"session_id": "s1", "cost": Infinity}', 0.0),
            ('', 'agent:main:slack:3', '{"session_id": "s2", "cost": NaN}', 0.0)]
    _write(store, lambda c: c.executemany('INSERT INTO gateway_routing VALUES (?,?,?,?)', rows))
    assert all(entry_session_id(r[2]) for r in rows)  # Python twin reads them
    _write(store, lambda c: retire_routes(c, ['s1']))
    assert [r[1] for r in _rows(store)] == ['agent:main:slack:3']


def test_retire_and_fence_statement_count_does_not_grow_with_ids(store):
    _seed(store, random.Random(2), n=100)
    for n_ids in (2, 500):
        seen = []
        ids = [f'id{i}' for i in range(n_ids)]
        def _do(conn, ids=ids, seen=seen):
            conn.set_trace_callback(seen.append)
            try:
                retire_routes(conn, ids, expiring=ids[::2])
                fence_shared_routes(conn, 'scout', ids, expiring=ids[::3])
            finally:
                conn.set_trace_callback(None)
        _write(store, _do)
        assert len(seen) <= 12, (n_ids, len(seen))


def test_retire_and_fence_are_one_routing_statement_each(store):
    _seed(store, random.Random(3), n=2000)
    seen = []
    def run(fn):
        def _do(conn):
            conn.set_trace_callback(seen.append)
            try:
                fn(conn)
            finally:
                conn.set_trace_callback(None)
        _write(store, _do)
    run(lambda c: retire_routes(c, ['s1', 's2']))
    run(lambda c: fence_shared_routes(c, 'scout', ['s3']))
    routing = [s for s in seen if 'gateway_routing' in s]
    assert len(routing) == 2, routing
