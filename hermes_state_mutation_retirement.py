"""Durable deletion fences outside prunable transcript rows."""
import json
import sqlite3
import time
from hermes_state_runtime import RuntimeStoreError, admission_fingerprint, mutation_receipt_key, _json, _text

# This store's own retired session (its profile is the store's). Written as ``own`` since the
# shared routing index also carries other profiles' fences; a marker without ``own`` predates
# that and still fences every route in the index.
RETIRED_PREFIX = 'gateway.retired_session.v1.'
# A multiplexed secondary's retired session, in the routing index every served profile shares
# (the launch home's state.db): ``<prefix><namespace>:<id>``, keyed by the ``agent:<namespace>:``
# slot of that profile's routing keys. Only routing saves read it, so the launch profile's own
# creates, binds and generation walks never see another profile's retirement.
ROUTE_FENCE_PREFIX = 'gateway.retired_route.v1.'
# A last viewer's close stored until the owner's idle point (gateway/session_viewer_close).
VIEWER_CLOSE_PREFIX = 'gateway.viewer_close.v1:'


def retired_in(conn, session_id):
    return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + session_id,)).fetchone() is not None


def retired_session(db, session_id):
    with db._read_ctx() as conn:
        return retired_in(conn, session_id)


def create_conflict(conn, session_id):
    """``(code, message)`` refusing a create of the explicit *session_id*, else None: a live row, or a
    deleted (retired) id. A row recreated beside the fence would list fine and refuse every turn."""
    if conn.execute('SELECT 1 FROM sessions WHERE id=?', (session_id,)).fetchone():
        return 'session_exists', 'Session already exists'
    if retired_in(conn, session_id):
        return 'session_deleted', 'Session was deleted; its id cannot be reused'
    return None


def first_generation(db, derive, *, pinned=None, spent=None):
    """``(n, derive(n))``: the first generation of a derived session id a NEW request may use. A
    user-deleted id stays retired for good, so a derived id (header-less chat, Bot Chat, hosted-room
    member, A2A context) whose session was deleted moves to its next generation, never recreated.
    ``pinned(conn, sid)`` keeps an exact retry on the retired generation holding its identity;
    ``spent(conn, sid)`` skips a live one used up another way (a renamed Bot Chat). Read from
    durable rows only, so a restart and every concurrent caller pick the same id."""
    generation = 0
    with db._read_ctx() as conn:
        while True:
            sid = derive(generation)
            if retired_in(conn, sid):
                if pinned is not None and pinned(conn, sid):
                    return generation, sid
            elif spent is None or not spent(conn, sid):
                return generation, sid
            generation += 1


def keep_passed_fences(conn, derive, session_id):
    """Make permanent the fences a walk (``derive`` as for :func:`first_generation`) passed on its
    way to *session_id*, inside the write that binds *session_id*: that successor now holds the
    conversation, so a sweep marker aging out would send a later walk back to the passed
    generation. False when a passed generation has lost its marker since the walk (a sweep's was
    reclaimed): a walk now stops there, so *session_id* must not be bound as its successor."""
    generation = 0
    while (sid := derive(generation)) != session_id:
        if not conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + sid,)).fetchone():
            return False
        conn.execute("UPDATE state_meta SET value=json_remove(value, '$.expires') WHERE key=? AND json_valid(value)",
                     (RETIRED_PREFIX + sid,))
        generation += 1
    return True


def has_mutation_receipt(db, principal_id, session_id, request_id):
    for value in (principal_id, session_id, request_id):
        _text(value)
    key, _digest = mutation_receipt_key(principal_id, session_id, request_id, operation='model', payload={},
                                        expected_revision=0)
    with db._read_ctx() as conn:
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (key,)).fetchone() is not None


_TRANSCRIPT_FIELDS = frozenset({'messages', 'last_reasoning', 'tools', '_turn_user_row'})


def _compact_result(conn, admission_id):
    """A retired result keeps the exact-retry outcome (final response, flags, usage/accounting)
    but not the transcript copies a turn result carries (cumulative ``messages``, reasoning, the
    settled ``_turn_user_row``)."""
    from hermes_state_terminal import RESULT_PREFIX, encode_result, load_result
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone()
    if saved is None:
        return
    value = load_result(saved[0])
    result = value.get('result')
    if isinstance(result, dict):
        value['result'] = {**{k: v for k, v in result.items() if k not in _TRANSCRIPT_FIELDS}, 'messages': []}
    conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(encode_result(value)), RESULT_PREFIX + admission_id))


def retire_terminal_receipts(conn, session_ids):
    from hermes_state_terminal import ADMISSION_PREFIX, WORKER_PREFIX, identity_key
    for sid in session_ids:
        admissions = conn.execute('SELECT * FROM session_admissions WHERE target_session_id=?', (sid,)).fetchall()
        workers = conn.execute('SELECT * FROM worker_executions WHERE session_id=?', (sid,)).fetchall()
        states = {r['status'] for r in [*admissions, *workers]} - {'terminal'}
        if states:
            raise RuntimeStoreError('unknown_execution' if 'unknown' in states else 'session_busy')
        for raw in admissions:
            row = dict(raw)
            from hermes_state_media import retire_media
            retire_media(conn, json.loads(row['payload_json']), row['request_id'])
            # Keep the digest for exact retries, not another copy of user input/history.
            row['payload_json'] = '{}'
            row['lineage_json'] = '[]'
            row['retired_at'] = time.time()
            _compact_result(conn, row['admission_id'])
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (ADMISSION_PREFIX + row['admission_id'], _json(row)))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (identity_key(row['principal_id'], sid, row['request_id']), json.dumps(row['admission_id'])))
        for raw in workers:
            row = dict(raw)
            receipts = [dict(r) for r in conn.execute(
                'SELECT sequence,payload_digest,result_json FROM worker_receipts WHERE execution_id=? ORDER BY sequence',
                (row['execution_id'],))]
            # A terminal worker can only replay an explicit ``execution.finish`` receipt. Every
            # other result (history/context reads) is user data that must not outlive the delete,
            # even when it is the last one: settlement terminalizes a failed worker without a
            # finish receipt. Digests stay so a late duplicate is still recognised as a conflict.
            closing = admission_fingerprint(canonical_target=sid,
                payload={'operation': 'execution.finish', 'payload': {}})
            for receipt in receipts:
                if receipt['payload_digest'] != closing:
                    receipt['result_json'] = None
            row['receipts'] = receipts
            row['retired_at'] = time.time()
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
                         (WORKER_PREFIX + row['execution_id'], _json(row)))
            conn.execute('DELETE FROM worker_receipts WHERE execution_id=?', (row['execution_id'],))
        conn.execute('DELETE FROM worker_executions WHERE session_id=?', (sid,))
        conn.execute('DELETE FROM session_admissions WHERE target_session_id=?', (sid,))


_LIVE_LEDGER_SQL = """SELECT 1 FROM session_admissions WHERE target_session_id=? AND status IN ('queued','started','unknown')
    UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? AND status IN ('registered','running','unknown') LIMIT 1"""


def retire_sessions(conn, session_ids, *, expiring=()):
    """The complete deletion fence for ids that are being deleted in this transaction.

    Terminal receipts alone are not enough: without the ``RETIRED_PREFIX`` marker an exact
    retry of a settled request reports ``not_found`` instead of its terminal receipt, a late
    accounting/constructor backfill recreates the deleted row, and the same request is then
    admitted a second time against it. Every delete path (canonical mutate, legacy
    ``delete_session*``, prunes and sweeps) must publish the whole fence, so it lives here once.
    The marker of an id in *expiring* (a housekeeping sweep's) ages out at the retry horizon."""
    from hermes_state_media import retire_spillover
    retire_spillover(conn, session_ids)  # before the caller's DELETEs take the pointers away
    retire_terminal_receipts(conn, session_ids)
    retire_mutation_receipts(conn, session_ids)
    retire_routes(conn, session_ids, expiring=expiring)
    from hermes_state_local import retire_local_receipts
    retire_local_receipts(conn, session_ids)


# Mutation-receipt fields that copy user content: the rewound message, a compaction summary and
# a user-set title (rename/sidebar).
_MUTATION_TRANSCRIPT_FIELDS = ('target_message', 'summary', 'title')
# Every session a receipt result can name. A local owner's receipt is keyed by its logical id, but
# its rewind/compress content comes from the PHYSICAL target (a reset child): deleting only that
# child must strip it too, or the surviving owner's exact retry replays the deleted text.
_RECEIPT_SESSION_PATHS = ('$.result.session_id', '$.result.target_message.session_id',
                          '$.result.target_session_id', '$.result.previous_target_session_id')


def retire_mutation_receipts(conn, session_ids):
    """Keep each exact-retry mutation receipt (digest, ids, revision) but drop the user content
    a rewind/compress/rename result carries, so a retry after the delete cannot read it back."""
    if not session_ids:
        return
    named = ' OR '.join(f"json_extract(value,'{path}') IN (SELECT value FROM json_each(?1))"
                        for path in _RECEIPT_SESSION_PATHS)
    rows = conn.execute(
        "SELECT key,value FROM state_meta WHERE key GLOB 'gateway.mutation.v1.*' AND "
        f"CASE WHEN json_valid(value) THEN ({named}) END", (json.dumps(list(session_ids)),)).fetchall()
    for key, raw in rows:
        receipt = json.loads(raw)
        result = receipt.get('result')
        if isinstance(result, dict) and any(f in result for f in _MUTATION_TRANSCRIPT_FIELDS):
            receipt['result'] = {k: (None if k in _MUTATION_TRANSCRIPT_FIELDS else v) for k, v in result.items()}
            conn.execute('UPDATE state_meta SET value=? WHERE key=?', (_json(receipt), key))


# Creation ids of the local conversations a gateway door derives and walks by generation
# (``first_generation``): Bot Chat and A2A contexts.
_DERIVED_LOCAL_PREFIXES = ('bot-chat:', 'a2a:')
# The API sessions whose ids a door derives: a hosted-room member (``bot_room``), and the
# ``api-<16 hex>`` ids of a header-less chat (``_derive_chat_session_id``) and of a deleted chat's or
# Responses/Runs chain's next generation (``implicit_api_session``). A caller-named ``api_server`` id
# (``POST /api/sessions``, ``X-Hermes-Session-Id``) is never derived. Matched on the id, not on a
# binding-receipt field, so it also holds for rows an older release bound. Any id can root a
# Responses/Runs chain: its fence becomes permanent once a chain moves past it (``keep_passed_fences``).
_DERIVED_API_SQL = ("(source='bot_room' OR (source='api_server' AND id GLOB 'api-" + '[0-9a-f]' * 16 + "'))")


def _local_conversations(conn):
    """``(members, derived)``: member id -> every id of its local conversation (the creation id
    (policy, FIFO, generation), each receipt lineage segment and the current target), and the ids
    of every derived (Bot Chat, A2A) conversation. One prefix scan, not a lookup per swept id.
    Malformed receipts name nothing."""
    from hermes_state_local import POLICY_PREFIX
    members, derived = {}, set()
    for (raw,) in conn.execute('SELECT value FROM state_meta WHERE key GLOB ?', (POLICY_PREFIX + '*',)):
        try:
            receipt = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(receipt, dict):
            continue
        entry, lineage = receipt.get('entry'), receipt.get('lineage')
        ids = tuple(x for x in [receipt.get('session_id'), *(lineage if isinstance(lineage, list) else []),
                                entry.get('session_id') if isinstance(entry, dict) else None] if isinstance(x, str))
        members.update(dict.fromkeys(ids, ids))
        if str(receipt.get('request_id') or '').startswith(_DERIVED_LOCAL_PREFIXES):
            derived.update(ids)
    return members, derived


def _derived_api_sessions(conn, session_ids):
    """The ids among *session_ids* of derived API sessions (``_DERIVED_API_SQL``)."""
    return {row[0] for row in conn.execute(
        f"SELECT id FROM sessions WHERE {_DERIVED_API_SQL} AND id IN (SELECT value FROM json_each(?))",
        (json.dumps(list(session_ids)),))}


def retire_prunable(conn, session_ids):
    """Sweep variant of :func:`retire_sessions`: fence the idle sessions and return only those ids.
    A session with live or unknown work is skipped, so one busy row cannot abort a whole
    prune/empty-session sweep (explicit deletes still refuse with ``session_busy``). A local
    reset/compression conversation is swept only whole: while any of its ids is kept or has live
    work (the owner's queued admission continues on the current segment), every one is skipped.

    A sweep's markers age out at the retry horizon: housekeeping retires thousands of ids a pass
    (cron), and a permanent marker each would grow ``state_meta`` without bound. A derived id's
    stays (Bot Chat, A2A context, header-less chat, a chain's next generation, hosted-room
    member): its conversation resumes on the next generation, and the generation walk stops at the
    first unfenced id, so an aged-out marker would send the next turn back to the swept generation.
    A Bot Chat's mailbox receipts also outlive the horizon. A caller-named API id is not derived:
    its marker ages out, so the caller can use that id again, as for any other swept session."""
    swept, (conversations, derived) = set(session_ids), _local_conversations(conn)

    def removable(sid):
        whole = conversations.get(sid, (sid,))
        return swept.issuperset(whole) and not any(conn.execute(_LIVE_LEDGER_SQL, (x, x)).fetchone() for x in whole)
    quiet = [sid for sid in session_ids if removable(sid)]
    derived |= _derived_api_sessions(conn, quiet)
    retire_sessions(conn, quiet, expiring=[sid for sid in quiet if sid not in derived])
    return quiet


def delete_in_transaction(db, conn, session_id, payload):
    from hermes_state_mutation_guards import require_idle, delete_targets
    targets = delete_targets(conn, session_id)
    require_idle(db, conn, targets)
    retire_sessions(conn, targets)
    for sid in targets:
        db._bump_conversation_generation(conn, sid, 'session_reset')
        conn.execute('UPDATE sessions SET parent_session_id=NULL, runtime_revision=runtime_revision+1 WHERE parent_session_id=?', (sid,))
        conn.execute('DELETE FROM messages WHERE session_id=?', (sid,))
    conn.executemany('DELETE FROM sessions WHERE id=?', [(sid,) for sid in targets])
    db._delete_unreferenced_system_prompts(conn)
    return set(), {'deleted_ids': targets}


def entry_session_id(raw):
    """The ``session_id`` a stored routing/receipt JSON object names, else None. Unrelated rows
    are not validated here: a malformed one (an array, invalid JSON) names no target, so it can
    neither be retired nor abort the delete of a different session; the routing loader skips it too."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value.get('session_id') if isinstance(value, dict) else None


def _mark_retired_many(conn, keys, *, expires, **fields):
    """Write the retirement marker of every key in ONE statement (KC: set-based, one shared
    ``retired_at``), so a bulk delete or prune does not step once per id under the writer mutex.

    Replace: the marker is a fact, not a receipt; re-retiring an id that was recreated beside its
    tombstone must not abort the delete that removes it again (and restarts its horizon). An
    ``expires`` (sweep) marker ages out at the horizon and never shortens a permanent one. A
    permanent marker without ``own`` (an earlier release's, maybe another profile's shared fence)
    stays unscoped: it is the only fence of that profile's route. An earlier release's sweep marker
    came only from this store's own sweep, so it is replaced as written."""
    keys = sorted(set(keys))
    if not keys:
        return
    value = {'retired_at': time.time(), **fields}
    batch = json.dumps(keys, ensure_ascii=False)
    if expires:
        conn.execute('INSERT INTO state_meta(key,value) SELECT j.value, ? FROM json_each(?) AS j WHERE true '
                     'ON CONFLICT(key) DO UPDATE '
                     "SET value=excluded.value WHERE json_valid(state_meta.value) "
                     "AND json_extract(state_meta.value, '$.expires')",
                     (json.dumps({**value, 'expires': True}), batch))
    else:
        conn.execute('INSERT INTO state_meta(key,value) SELECT j.value, ? FROM json_each(?) AS j WHERE true '
                     'ON CONFLICT(key) DO UPDATE SET value=CASE '
                     "WHEN NOT json_valid(state_meta.value) THEN json_remove(excluded.value, '$.own') "
                     "WHEN json_extract(state_meta.value, '$.own') IS NOT NULL "
                     "OR json_extract(state_meta.value, '$.expires') THEN excluded.value "
                     "ELSE json_remove(excluded.value, '$.own') END", (json.dumps(value), batch))


def retire_routes(conn, session_ids, *, expiring=()):
    # Set-based (KC): a fixed handful of statements whatever the number of ids or routes.
    ids, expiring = set(session_ids), set(expiring)
    _mark_retired_many(conn, [RETIRED_PREFIX + sid for sid in ids if sid in expiring], expires=True, own=True)
    _mark_retired_many(conn, [RETIRED_PREFIX + sid for sid in ids if sid not in expiring], expires=False, own=True)
    gone = [prefix + sid for sid in ids for prefix in ('gateway.api.settings.v1.', VIEWER_CLOSE_PREFIX)]
    conn.execute('DELETE FROM state_meta WHERE key IN (SELECT value FROM json_each(?))',
                 (json.dumps(sorted(gone), ensure_ascii=False),))
    # Set-based (KC): one statement, not one Python step per routing row under the writer mutex
    # (the per-row scan convoys on the GIL in a busy multiplex gateway).
    conn.execute(f'DELETE FROM gateway_routing WHERE {entry_session_id_sql("entry_json")} '
                 'IN (SELECT value FROM json_each(?))', (json.dumps(sorted(set(session_ids))),))


def route_namespace(session_key):
    """The ``<ns>`` slot of an ``agent:<ns>:...`` routing key (``main`` for the launch profile, else
    the profile's ``_session_key_namespace`` slot); ``main`` for a key outside that shape."""
    parts = str(session_key or '').split(':')
    return parts[1] if len(parts) > 2 and parts[0] == 'agent' and parts[1] else 'main'


def route_namespace_sql(key_expr):
    """SQL twin of :func:`route_namespace` for the routing key in *key_expr*, as UTF-8 BYTES (a
    BLOB): the second ``:``-separated segment of an ``agent:<ns>:...`` key, else ``b'main'``.
    Byte-wise because SQLite's text functions stop at an embedded NUL where ``str.split`` does
    not; ``:`` is one byte in UTF-8, so byte and character splitting agree. Compare the result
    with ``CAST(<name> AS BLOB)``. Case-sensitive (no ``LIKE``); ``instr(...) > 1`` means a later
    ``:`` exists and <ns> is non-empty."""
    key = f"CAST({key_expr} AS BLOB)"
    rest = f"substr({key}, 7)"
    return (f"(CASE WHEN substr({key}, 1, 6) = CAST('agent:' AS BLOB) AND instr({rest}, x'3a') > 1 "
            f"THEN substr({rest}, 1, instr({rest}, x'3a') - 1) ELSE CAST('main' AS BLOB) END)")


def entry_session_id_sql(json_expr):
    """SQL twin of :func:`entry_session_id`: the text ``session_id`` of a stored JSON object, else
    NULL; never raises. Like ``json.loads`` it takes the LAST top-level ``session_id`` member
    (``json_extract`` would take the first) and matches escaped keys. A non-text id (number,
    array, object) is NULL: it names no target (the twin raised ``TypeError`` on unhashable ids
    and aborted the whole delete; here such a malformed route is simply left alone). Each JSON
    read sits in its own ``CASE WHEN`` because ``AND`` does not short-circuit on malformed text.
    ``json_valid(x, 2)`` accepts JSON5 so NaN/Infinity (which ``json.dumps`` writes for float
    nan/inf) parse as they do for ``json.loads``; other JSON5-only syntax (comments, unquoted
    keys) is never written by ``json.dumps``."""
    last_id = (f"(SELECT CASE WHEN j.type = 'text' THEN j.value END FROM json_each({json_expr}) AS j "
               "WHERE j.key = 'session_id' ORDER BY j.id DESC LIMIT 1)")
    # The flags argument of json_valid needs SQLite 3.45; an older library keeps strict JSON (NaN
    # entries are then left alone, as before this change) instead of failing every retirement.
    valid = f"json_valid({json_expr}, 2)" if sqlite3.sqlite_version_info >= (3, 45, 0) else f"json_valid({json_expr})"
    return (f"(CASE WHEN {valid} THEN CASE WHEN json_type({json_expr}) = 'object' "
            f"THEN {last_id} END END)")


def route_fence_key(namespace, session_id):
    return f'{ROUTE_FENCE_PREFIX}{namespace}:{session_id}'


# True when a routing save must skip an entry, checked only against the profile the route belongs
# to: this store's own marker for a launch (``main``) route, that profile's fence for a secondary's.
# A launch-keyed route to an id this store holds no session of can only name another profile's
# session, so a secondary's fence covers it too (``fence_shared_routes``). A sweep's fence stops
# fencing at the horizon by itself: only the owner's own store sweeps the owner's marker, and nothing
# guarantees the launch profile ever sweeps this one. A delete's fence (no ``expires``) never does:
# its age test is NULL, which ``coalesce`` reads as not expired. An unscoped marker (no ``own``: an
# earlier release's, written for either) still fences every route. So does a marker or fence that is
# not valid JSON (no writer stores one): ``json_extract`` raises ``malformed JSON`` on it even behind
# ``json_valid(...) AND``, so each JSON read sits in a ``CASE WHEN json_valid(...)`` branch.
# Bound by name (``route_retired_params``): :entry is the route's JSON, :ns its key's
# ``route_namespace``, :now the save's time.
_ROUTE_RETIRED_TEMPLATE = """(
    EXISTS (SELECT 1 FROM state_meta m
             WHERE m.key = :retired_prefix || @SID@
               AND (@NS@ = 'main' OR NOT CASE WHEN json_valid(m.value)
                                             THEN json_extract(m.value, '$.own') IS NOT NULL ELSE 0 END))
    OR EXISTS (SELECT 1 FROM state_meta f
                WHERE f.key = :fence_prefix || @NS@ || ':' || @SID@
                  AND NOT CASE WHEN json_valid(f.value)
                               THEN coalesce(json_extract(f.value, '$.expires')
                                             AND json_extract(f.value, '$.retired_at') < :now - :horizon, 0)
                               ELSE 0 END
                  AND (@NS@ != 'main' OR NOT EXISTS (
                          SELECT 1 FROM sessions s WHERE s.id = @SID@))))"""


def route_retired_sql(sid_expr, ns_expr):
    """``ROUTE_RETIRED_SQL`` with the route's session id and namespace read from SQL expressions
    (for example the columns of a materialized ``json_each`` batch) instead of the single-row
    :entry / :ns parameters. Same predicate text, so the per-row and set-based saves cannot drift
    apart. *sid_expr* must evaluate to ``json_extract(<entry json>, '$.session_id')``."""
    return _ROUTE_RETIRED_TEMPLATE.replace('@SID@', sid_expr).replace('@NS@', ns_expr)


ROUTE_RETIRED_SQL = route_retired_sql("json_extract(:entry, '$.session_id')", ':ns')


def route_retired_params(session_key, entry_json, now):
    """The named parameters of ``ROUTE_RETIRED_SQL`` for one routing entry saved at *now*."""
    return {'entry': entry_json, 'ns': route_namespace(session_key), 'now': now, 'horizon': RETRY_HORIZON_S,
            'retired_prefix': RETIRED_PREFIX, 'fence_prefix': ROUTE_FENCE_PREFIX}


def route_retired(db, session_key, entry_json):
    """A routing save (or the legacy sessions.json import) must skip this entry."""
    with db._read_ctx() as conn:
        return bool(conn.execute('SELECT ' + ROUTE_RETIRED_SQL,
                                 route_retired_params(session_key, entry_json, time.time())).fetchone()[0])


def expiring_retirements(db, session_ids):
    """The ids among *session_ids* whose own marker in *db* is a sweep's (ages out at the horizon)."""
    with db._read_ctx() as conn:
        return {key[len(RETIRED_PREFIX):] for (key,) in conn.execute(
            "SELECT key FROM state_meta WHERE key IN (SELECT ? || value FROM json_each(?)) "
            "AND json_valid(value) AND json_extract(value, '$.expires')",
            (RETIRED_PREFIX, json.dumps(list(session_ids))))}


def fence_shared_routes(conn, namespace, session_ids, *, expiring=()):
    """In the shared routing index (*conn*), fence the routes of *session_ids*, retired in the
    profile whose routing keys use *namespace*, and drop them. The fence is keyed by (profile, id)
    and copies the owner's marker: permanent for a delete, aging out with an *expiring* sweep's.
    Unless this store holds a session of that id, the launch (``main``) namespace is fenced too: a
    launch-keyed route to it names the retired session. The sweep fences that aged out here are
    reclaimed in the same write, so the owner's prunes bound them without the launch profile's sweep."""
    _reclaim_expired(conn, ROUTE_FENCE_PREFIX, time.time() - RETRY_HORIZON_S)
    expiring, ids = set(expiring), set(session_ids)
    # One read for which ids this store holds (no per-id query), then one marker statement per
    # expiry class (set-based, KC).
    held = set(json.loads(conn.execute(
        'SELECT json_group_array(id) FROM sessions WHERE id IN (SELECT value FROM json_each(?))',
        (json.dumps(list(ids), ensure_ascii=False),)).fetchone()[0] or '[]'))
    owned = {sid: ({namespace} if sid in held else {namespace, 'main'}) for sid in ids}
    for expires in (True, False):
        _mark_retired_many(conn, [route_fence_key(name, sid) for sid, names in owned.items()
                                  if (sid in expiring) is expires for name in names], expires=expires)
    # Set-based (KC): drop every route whose (key namespace, session id) pair is fenced, in one
    # statement instead of a per-row Python scan under the writer mutex.
    pairs = [[name, sid] for sid, names in owned.items() for name in names]  # order-free: no sort
    conn.execute(
        f'DELETE FROM gateway_routing WHERE ({route_namespace_sql("session_key")}, '
        f'{entry_session_id_sql("entry_json")}) IN (SELECT CAST(json_extract(p.value, \'$[0]\') AS BLOB), '
        'json_extract(p.value, \'$[1]\') FROM json_each(?) AS p)', (json.dumps(pairs),))


RETRY_HORIZON_S = 7 * 86400  # exact-retry horizon: a retry within it gets the byte-exact settled reply
_EXPIRED = "json_valid(value) AND json_extract(value, '$.retired_at') < ?"


def _reclaim_expired(conn, prefix, cutoff):
    """Drop the sweep (``expires``) markers under *prefix* retired before *cutoff*."""
    conn.execute(f"DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED} AND json_extract(value, '$.expires')",
                 (prefix + '*', cutoff))


def prune_retired_tombstones(conn, older_than=RETRY_HORIZON_S):
    """Reclaim deletion tombstones retired more than *older_than* seconds ago: the terminal admission
    with its identity pointer and compacted result, terminal workers and a sweep's (``expires``)
    retired-session marker or shared routing fence. A user delete's marker is the deletion fence itself and is never
    reclaimed: it holds only ``retired_at``, and without it a deleted id could be recreated,
    re-admitted or reissued as a Bot Chat generation.
    Tombstones written before ``retired_at`` existed are stamped now and age from this pass."""
    from hermes_state_terminal import ADMISSION_PREFIX, RESULT_PREFIX, WORKER_PREFIX, identity_key
    prefixes = (ADMISSION_PREFIX, WORKER_PREFIX)
    now = time.time()
    for prefix in prefixes:
        conn.execute("UPDATE state_meta SET value=json_set(value, '$.retired_at', ?) WHERE key GLOB ? "
                     "AND json_valid(value) AND json_extract(value, '$.retired_at') IS NULL", (now, prefix + '*'))
    cutoff = now - older_than
    for key, raw in conn.execute(f'SELECT key,value FROM state_meta WHERE key GLOB ? AND {_EXPIRED}',
                                 (ADMISSION_PREFIX + '*', cutoff)).fetchall():
        row = json.loads(raw)
        conn.execute('DELETE FROM state_meta WHERE key IN (?,?,?)', (key, RESULT_PREFIX + row['admission_id'],
                     identity_key(row['principal_id'], row['target_session_id'], row['request_id'])))
    conn.execute(f'DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED}', (WORKER_PREFIX + '*', cutoff))
    for prefix in (RETIRED_PREFIX, ROUTE_FENCE_PREFIX):
        _reclaim_expired(conn, prefix, cutoff)
