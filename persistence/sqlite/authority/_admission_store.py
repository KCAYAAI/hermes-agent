"""The admission ledger's transactions: admit, the row reads, claim and its return, settlement, cancellation, the
steering handoff, rehoming, retargeting, boot recovery and the operator's resolution of an unknown turn.

Each function takes the store holder and runs its statements inside one ``_execute_write`` or ``_read_ctx``. The
SQL-bearing checks the owner passes (the admission gate, the typed extras, the discarded-turn closer) arrive as typed
values and run at a fixed point inside that transaction; their bodies live in :mod:`._admission_owner`.
"""

from __future__ import annotations

import heapq
import json
import uuid

from persistence.authority.vocabulary import ADMISSION_QUEUE_MAX, ADMITTED_REQUEST_CONFLICT, STEERING, admission_fingerprint
from persistence.authority.admissions import raise_gate
from persistence.core.errors import AuthorityRefusal

from ._admission_owner import authorize_admission, close_discarded_turn
from ._runtime import (
    _CANCEL_QUEUED,
    _admission,
    _epoch,
    _json,
    _requeue_transfer,
    _retire_admission_workers,
    _row,
    _session,
    _text,
    check_replaced_input,
    compact_result,
    encode_result,
    replacement_binding,
    retry_terminal_admission,
    settle_steer_transfers,
    terminal_admission,
)

__all__ = [
    "_ANY_HEAD",
    "_canonical_chain",
    "admit_session_input",
    "cancel_session_input",
    "claim_recovery_notice",
    "claim_session_input",
    "clear_recovery_owed",
    "find_session_admission",
    "get_session_admission",
    "latest_session_admission",
    "list_session_admissions",
    "lost_turn_transcript",
    "owed_recovery_answers",
    "recover_unknown_session_input",
    "recovery_attempts",
    "recover_session_inputs",
    "rehome_queued_inputs",
    "resolve_unknown_session_input",
    "retarget_local_automation",
    "return_session_claim",
    "revert_steer_input",
    "settle_session_input",
    "steer_session_input",
]


def admit_session_input(holder, *, epoch: int, principal_id: str, session_id: str,
                        request_id: str, payload: dict, intent: str = 'queue',
                        gate=None, replaces=None, supersedes=None, api_target=None, rewrite=None,
                        hosted=None, fold=None) -> tuple[dict, bool]:
    """Admit input; ``(row, created)``. ``gate`` (the owner's ``GateRefusal``, or a callable answered here on every
    attempt) refuses a NEW write after the queue cap, then the typed extras run in the same write before the insert. Exact
    existing and terminal replays bypass both; ``replaces`` runs its replay check on the exact live replay."""
    for value in (principal_id, session_id, request_id):
        _text(value)
    if intent not in ('queue', 'steer', 'redirect'):
        raise AuthorityRefusal('invalid_params')
    encoded = _json(payload)
    retired = retry_terminal_admission(holder, epoch=epoch, principal_id=principal_id, session_id=session_id,
        request_id=request_id, payload=payload, intent=intent)
    if retired is not None:
        return retired, False
    digest = admission_fingerprint(canonical_target=session_id, payload={'input': json.loads(encoded), 'intent': intent})
    def write(conn):
        _epoch(conn, epoch)
        _session(conn, session_id)
        old = conn.execute('''SELECT * FROM session_admissions
            WHERE principal_id=? AND target_session_id=? AND request_id=?''', (principal_id, session_id, request_id)).fetchone()
        if old is not None:
            if old['payload_digest'] != digest:
                raise AuthorityRefusal('admission_conflict', **ADMITTED_REQUEST_CONFLICT)
            if replaces is not None:
                check_replaced_input(conn, replaces.admission_id,
                                     replacement_binding(principal_id, session_id, request_id))
            return _row(old), False
        queued = conn.execute('''SELECT COUNT(*) FROM session_admissions WHERE target_session_id=?
            AND status='queued' ''', (session_id,)).fetchone()[0]
        # Main's cap holds in both states: a FIFO paused behind an unknown turn runs every accepted
        # input after Discard, so it must not grow without bound while nobody resolves it.
        if queued >= ADMISSION_QUEUE_MAX:
            raise AuthorityRefusal('queue_full')
        raise_gate(gate)
        authorize_admission(holder, conn, epoch=epoch, principal_id=principal_id, session_id=session_id,
                            request_id=request_id, replaces=replaces, supersedes=supersedes,
                            api_target=api_target, rewrite=rewrite, hosted=hosted, fold=fold)
        admission_id = uuid.uuid4().hex
        conn.execute('''INSERT INTO session_admissions(admission_id,request_id,principal_id,
            target_session_id,lineage_json,payload_json,payload_digest,intent,status,owner_epoch)
            VALUES(?,?,?,?,?,?,?,?,'queued',?)''',
            (admission_id, request_id, principal_id, session_id, json.dumps([session_id]), encoded, digest, intent, epoch))
        return _row(_admission(conn, admission_id)), True
    return holder._execute_write(write)


def get_session_admission(holder, *, admission_id: str) -> dict | None:
    with holder._read_ctx() as conn:
        row = conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()
        return _row(row if row is not None else terminal_admission(conn, admission_id))


def find_session_admission(holder, *, principal_id: str, session_id: str, request_id: str) -> dict | None:
    """The admission at its exact durable identity: the UNIQUE (principal, target, request) key."""
    with holder._read_ctx() as conn:
        return _row(conn.execute('SELECT * FROM session_admissions WHERE principal_id=? AND target_session_id=? '
                                 'AND request_id=?', (principal_id, session_id, request_id)).fetchone())


def list_session_admissions(holder, *, session_id: str, pending_only: bool = True) -> list[dict]:
    with holder._read_ctx() as conn:
        # Pending is an explicit index-range IN, not ``!='terminal'``: a negated status cannot
        # bound the (target, status, seq) index, so every probe scanned the session's history.
        sql = ("SELECT * FROM session_admissions WHERE target_session_id=? "
               + ("AND status IN ('queued','started','unknown') " if pending_only else "") + "ORDER BY seq")
        return [_row(row) for row in conn.execute(sql, (session_id,))]


_ADMISSION_STATUSES = ('queued', 'started', 'unknown', 'terminal')  # the table's status CHECK


def latest_session_admission(holder, *, session_id: str, payload_marker: str) -> dict | None:
    """The newest admission whose decoded payload holds the top-level key ``payload_marker``, stopping at the first
    match: a restored lane needs only its latest envelope, not the session's whole history. One newest-first stream
    per status, merged by ``seq``, so each walks the (target, status, seq) index with no sort."""
    sql = "SELECT * FROM session_admissions WHERE target_session_id=? AND status=? ORDER BY seq DESC"
    with holder._read_ctx() as conn:
        cursors = [conn.execute(sql, (session_id, status)) for status in _ADMISSION_STATUSES]
        try:
            rows = heapq.merge(*cursors, key=lambda row: row['seq'], reverse=True)
            return next((row for row in map(_row, rows)
                         if isinstance(row['payload'], dict) and payload_marker in row['payload']), None)
        finally:
            for cursor in cursors:  # before the connection goes back to the pool
                cursor.close()


_ANY_HEAD = object()


def claim_session_input(holder, *, epoch: int, session_id: str, gate=None, head=_ANY_HEAD) -> dict | bool | None:
    """Claim the FIFO head. ``gate`` (the owner's ``GateRefusal``, or a callable answered here on every attempt)
    refuses right after the epoch fence. ``head`` is the admission id the caller's preclaim checks validated (None: it saw nothing queued); if
    the head is another row by the time this write runs, nothing is claimed and False tells the caller to re-read and
    validate the row that is now first. The claimed row carries ``queued_owner_epoch``, the ``owner_epoch`` it held
    while queued."""
    def write(conn):
        _epoch(conn, epoch)
        raise_gate(gate)
        session = _session(conn, session_id)
        blocked = conn.execute("SELECT status FROM session_admissions WHERE target_session_id=? AND status IN ('started','unknown')", (session_id,)).fetchall()
        if any(row[0] == 'unknown' for row in blocked):
            raise AuthorityRefusal('unknown_execution')
        from ..workers._runtime_workers import worker_states
        workers = worker_states(conn, session_id)
        if 'unknown' in workers:
            raise AuthorityRefusal('unknown_execution')
        if blocked or workers:
            return None
        row = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' ORDER BY seq LIMIT 1", (session_id,)).fetchone()
        if head is not _ANY_HEAD and (row['admission_id'] if row is not None else None) != head:
            return False
        if row is None:
            return None
        generation = session['runtime_generation'] + 1
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation, session_id))
        # The transcript high-water mark at the claim (``messages.id`` is AUTOINCREMENT across every
        # session): a row above it on this conversation was written by this turn, so settlement can
        # tell this turn's own failed-turn boundary from an earlier turn's (#134889 GW-21). KC: it is
        # also stored on the row, so restart recovery never counts a row written before the claim
        # (an idle cache-expiry compaction summary, say) as this turn having run.
        mark = conn.execute('SELECT COALESCE(MAX(id), 0) FROM messages').fetchone()[0]
        changed = conn.execute("UPDATE session_admissions SET status='started',owner_epoch=?,generation=?,transcript_mark=? "
                               "WHERE admission_id=? AND status='queued'", (epoch, generation, mark, row['admission_id']))
        if changed.rowcount != 1:
            raise AuthorityRefusal('stale_generation')
        claimed = _row(_admission(conn, row['admission_id']))
        claimed['transcript_mark'] = mark
        claimed['queued_owner_epoch'] = row['owner_epoch']
        return claimed
    return holder._execute_write(write)


def return_session_claim(holder, *, epoch: int, admission_id: str, generation: int, queued_owner_epoch) -> dict:
    """Undo a claim whose turn never started: the claim's two updates reversed, so the ledger is the one a claim
    refused by its gate leaves. A row already queued with no generation is returned unchanged; otherwise the row
    must still be this owner's started claim at ``generation``, the session at ``generation``, with no worker and
    no steering transfer bound to the claim."""
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if row['status'] == 'queued' and row['generation'] is None:
            return _row(row)
        if (row['status'] != 'started' or row['owner_epoch'] != epoch
                or type(generation) is not int or row['generation'] != generation):
            raise AuthorityRefusal('stale_generation')
        session = _session(conn, row['target_session_id'])
        lineage = json.loads(row['lineage_json'])
        targets = list({row['target_session_id'], lineage[-1] if lineage else row['target_session_id']})
        if (session['runtime_generation'] != generation
                or conn.execute(f"SELECT 1 FROM worker_executions WHERE session_id IN ({','.join('?' * len(targets))}) "
                                "AND owner_epoch=? AND generation=? LIMIT 1", (*targets, epoch, generation)).fetchone()
                or conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='terminal' "
                                "AND outcome=? AND owner_epoch=? AND generation=? LIMIT 1",
                                (row['target_session_id'], STEERING, epoch, generation)).fetchone()):
            raise AuthorityRefusal('stale_generation')
        conn.execute("UPDATE session_admissions SET status='queued',owner_epoch=?,generation=NULL "
                     "WHERE admission_id=? AND status='started'", (queued_owner_epoch, admission_id))
        conn.execute('UPDATE sessions SET runtime_generation=? WHERE id=?', (generation - 1, row['target_session_id']))
        return _row(_admission(conn, admission_id))
    return holder._execute_write(write)


def settle_session_input(holder, *, epoch: int, admission_id: str, generation: int, outcome: str,
                         result: dict | None = None, _transfers=None, boundary=None) -> dict:
    """``boundary`` (a ``TurnBoundary``) closes the turn on its current transcript in this same write, before the
    result and the terminal status are stored; raising rolls the settlement back."""
    if outcome not in ('completed', 'interrupted', 'rejected', 'failed'):
        raise AuthorityRefusal('invalid_params')
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        session = _session(conn, row['target_session_id'])
        if (row['status'] != 'started' or row['owner_epoch'] != epoch
                or type(generation) is not int or row['generation'] != generation
                or session['runtime_generation'] != generation):
            raise AuthorityRefusal('stale_generation')
        _retire_admission_workers(conn, row, epoch)
        if boundary is not None:
            close_discarded_turn(holder, conn, row, boundary.notice, replace_after=boundary.replace_after,
                                 display_metadata=boundary.display_metadata)
        if result is not None:
            from ._runtime import RESULT_PREFIX
            encoded = _json(encode_result(compact_result(result, user_message=json.loads(row['payload_json']).get('text'))))
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, encoded))
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?", (outcome, admission_id))
        settle_steer_transfers(conn, epoch=epoch, generation=generation, transfers=_transfers, outcome=outcome)
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return holder._execute_write(write)


def cancel_session_input(holder, *, epoch: int, admission_id: str) -> dict:
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if row['status'] == 'unknown':
            raise AuthorityRefusal('unknown_execution')
        # A steer in transfer reads as its running claim and settles with it: as started, not cancellable.
        if row['status'] == 'started' or row['outcome'] == STEERING:
            raise AuthorityRefusal('stale_generation')
        if row['status'] == 'queued':
            conn.execute(_CANCEL_QUEUED, (admission_id,))
        return _row(_admission(conn, admission_id))
    return holder._execute_write(write)


def steer_replayed(row) -> bool:
    """``row`` is a committed transfer (still in flight, or settled ``steered``): an exact retry of
    ``prompt.steer_queued`` replays it."""
    return row['status'] == 'terminal' and row['outcome'] in (STEERING, 'steered')


def steer_session_input(holder, *, epoch: int, admission_id: str, generation: int) -> tuple[dict, bool]:
    """Hand a still-queued admission to its FIFO's running claim: queued -> terminal ``steering``
    bound to ``generation`` in one write, only while ``generation`` is the started claim this owner
    holds, so the drain can never also claim it. ``(row, True)`` when this call committed the
    transfer, ``(row, False)`` for an exact retry of one already committed (never delivered twice)."""
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if steer_replayed(row):
            return _row(row), False
        if row['status'] != 'queued':
            raise AuthorityRefusal('admission_started')
        if type(generation) is not int or conn.execute(
                "SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='started' "
                "AND owner_epoch=? AND generation=?", (row['target_session_id'], epoch, generation)).fetchone() is None:
            raise AuthorityRefusal('session_idle')
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=?,owner_epoch=?,generation=? "
                     "WHERE admission_id=?", (STEERING, epoch, generation, admission_id))
        return _row(_admission(conn, admission_id)), True
    return holder._execute_write(write)


def revert_steer_input(holder, *, epoch: int, admission_id: str, generation: int) -> bool:
    """A transfer whose text never reached the claim's agent: the admission is queued again, in its
    original FIFO place (its ``seq``). False when the claim's settlement already decided it."""
    def write(conn):
        _epoch(conn, epoch)
        return _requeue_transfer(conn, admission_id=admission_id, epoch=epoch, generation=generation)
    return holder._execute_write(write)


def rehome_queued_inputs(holder, *, epoch: int, old_session_id: str, new_session_id: str,
                         principal_prefix: str) -> list[str]:
    """Re-target the old owner's still-queued ``principal_prefix*`` admissions onto
    ``new_session_id`` in one transaction (the compression-exhaustion auto-reset). Each row keeps
    its admission id, request id and seq, so FIFO order and retry dedupe survive; the digest is
    re-keyed to the new target. Nothing moves while the old FIFO holds an ``unknown`` row (the
    operator's to resume). Returns the moved admission ids in FIFO order."""
    for value in (old_session_id, new_session_id, principal_prefix):
        _text(value)
    def write(conn):
        _epoch(conn, epoch)
        _session(conn, new_session_id)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'",
                        (old_session_id,)).fetchone():
            return []
        rows = conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' "
                            "AND substr(principal_id,1,?)=? ORDER BY seq",
                            (old_session_id, len(principal_prefix), principal_prefix)).fetchall()
        for row in rows:
            digest = admission_fingerprint(canonical_target=new_session_id, payload={
                'input': json.loads(row['payload_json']), 'intent': row['intent']})
            conn.execute("UPDATE session_admissions SET target_session_id=?,lineage_json=?,payload_digest=? "
                         "WHERE admission_id=? AND status='queued'",
                         (new_session_id, json.dumps([new_session_id]), digest, row['admission_id']))
        if rows:
            for sid in (old_session_id, new_session_id):
                conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (sid,))
        return [row['admission_id'] for row in rows]
    return holder._execute_write(write)


def retarget_local_automation(holder, *, epoch: int, session_id: str, principal_id: str,
                               old_targets, new_target: str) -> list[str]:
    """Point a LOCAL owner's still-queued ``principal_id`` automation rows (a Bot Chat's queued
    ``hermes peer dm`` turns, plugin and completion notices) that name a pre-reset transcript in
    ``old_targets`` at ``new_target``, in one transaction (the compression-exhaustion auto-reset).
    A LOCAL FIFO's owner is its receipt id and survives the reset; only the descriptor's ``target``
    goes stale, and the drain refuses a stale one (``admission_conflict``), pausing the FIFO for
    good. Admission id, request id and seq are kept; the digest is re-keyed to the new payload.
    Nothing moves behind an ``unknown`` head. Returns the retargeted admission ids in FIFO order."""
    for value in (session_id, principal_id, new_target, *old_targets):
        _text(value)
    old_targets = set(old_targets)

    def write(conn):
        _epoch(conn, epoch)
        _session(conn, new_target)
        if conn.execute("SELECT 1 FROM session_admissions WHERE target_session_id=? AND status='unknown'",
                        (session_id,)).fetchone():
            return []
        moved = []
        for row in conn.execute("SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' "
                                "AND principal_id=? ORDER BY seq", (session_id, principal_id)).fetchall():
            payload = json.loads(row['payload_json'])
            descriptor = payload.get('local_automation_v1') if isinstance(payload, dict) else None
            if not isinstance(descriptor, dict) or descriptor.get('target') not in old_targets:
                continue
            descriptor['target'] = new_target
            digest = admission_fingerprint(canonical_target=session_id, payload={'input': payload, 'intent': row['intent']})
            conn.execute("UPDATE session_admissions SET payload_json=?,payload_digest=? WHERE admission_id=? AND status='queued'",
                         (_json(payload), digest, row['admission_id']))
            moved.append(row['admission_id'])
        if moved:
            conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (session_id,))
        return moved
    return holder._execute_write(write)


def recover_session_inputs(holder, *, epoch: int) -> int:
    """Never replay started work. Live worker adoption is a separate explicit operation."""
    def write(conn):
        _epoch(conn, epoch)
        from ..workers._runtime_workers import retire_stale_children
        retire_stale_children(conn, epoch)
        conn.execute("UPDATE worker_executions SET status='unknown' WHERE status IN ('registered','running') AND owner_epoch!=?", (epoch,))
        # A steer handed to a lost claim may or may not have reached its agent: never replayed.
        conn.execute("UPDATE session_admissions SET outcome='interrupted' WHERE status='terminal' AND outcome=? "
                     "AND owner_epoch!=?", (STEERING, epoch))
        return conn.execute("UPDATE session_admissions SET status='unknown' WHERE status='started' AND owner_epoch!=?", (epoch,)).rowcount
    return holder._execute_write(write)


def _canonical_chain(conn, session_id):
    # Same selector as the store's get_compression_chain, on OUR transaction connection.
    from ..conversation._store.compression import _CHAIN_STEP_SQL
    _session(conn, session_id)
    chain = [session_id]
    for _ in range(100):
        child = conn.execute(_CHAIN_STEP_SQL, (chain[-1],)).fetchone()
        if child is None:
            return chain
        if child['id'] in chain:
            raise AuthorityRefusal('admission_conflict')
        chain.append(child['id'])
    raise AuthorityRefusal('admission_conflict')


def resolve_unknown_session_input(holder, *, epoch: int, admission_id: str, generation: int,
                                  notice=None, _captured=None) -> dict:
    """Explicit operator acknowledgement; resolves uncertainty, never requeues it.

    The discarded turn's transcript boundary commits in this same write, so the FIFO can never advance past an
    unclosed lost turn; it carries ``notice`` unless the captured result commits.
    ``_captured=(outcome, stored_result)`` is the exact result this owner captured for the turn
    before its settlement write failed; it commits with the resolution (only for an ``unknown``
    row stamped by this same owner epoch), so the finished answer is not thrown away."""
    from ._runtime import RESULT_PREFIX
    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if (type(generation) is int and row['generation'] == generation and row['status'] == 'terminal'
                and (row['outcome'] == 'interrupted' or conn.execute(
                    'SELECT 1 FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone())):
            return _row(row)
        if type(generation) is not int or row['status'] != 'unknown' or row['generation'] != generation:
            raise AuthorityRefusal('stale_generation')
        outcome = 'interrupted'
        if _captured is not None and row['owner_epoch'] == epoch:
            outcome, stored = _captured
            if outcome not in ('completed', 'interrupted', 'failed'):
                raise AuthorityRefusal('invalid_params')
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                         'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                         (RESULT_PREFIX + admission_id, _json(encode_result(compact_result(
                             stored, user_message=json.loads(row['payload_json']).get('text'))))))
        # ``notice`` says the turn was interrupted: never beside a captured result this resolution commits.
        close_discarded_turn(holder, conn, row, None if (
            _captured is not None and row['owner_epoch'] == epoch) else notice)
        _retire_admission_workers(conn, row, row['owner_epoch'])
        conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?",
                     (outcome, admission_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id))
    return holder._execute_write(write)


def recover_unknown_session_input(holder, *, epoch: int, admission_id: str, generation: int, action: str,
                                  notice=None, _captured=None, owed: str | None = None) -> tuple[dict, bool]:
    """Restart recovery of a person's ``unknown`` messaging turn, decided by the caller from its saved
    transcript (``gateway.session_restart_recovery``). One fenced transaction, like
    ``resolve_unknown_session_input``; ``(row, recovered_now)``:

    * ``completed``: the turn's final answer was saved before the owner died; ``_captured`` is
      ``prepare_result``'s ``(outcome, stored_result)`` and commits as the turn's result.
    * ``requeue``: nothing with an effect ran; the row goes back to ``queued`` in its own FIFO
      place (same seq, requester and payload). The attempt counter (``RECOVERY_ATTEMPTS_PREFIX``) is
      bumped in the same write so a turn that keeps dying is never re-run more than the caller's cap.
    * ``interrupted``: the turn ends.

    ``notice``: the transcript boundary the discarded-turn closer writes in this same write (none
    when ``None``). ``owed`` (``completed`` only): the saved answer still to deliver, recorded in the
    same write (``RECOVERY_OWED_PREFIX``) so it is owed exactly when the turn settled.

    An exact retry of a recovery that already committed (or a row a person's /discard resolved)
    returns the row unchanged with ``recovered_now`` False: this call changed nothing, so its caller
    must not act (e.g. send the answer of a turn the person dropped)."""
    from persistence.authority.admissions import RECOVERY_ATTEMPTS_PREFIX, RECOVERY_OWED_PREFIX
    from ._runtime import RESULT_PREFIX
    if action not in ('completed', 'requeue', 'interrupted'):
        raise AuthorityRefusal('invalid_params')
    attempts_key = RECOVERY_ATTEMPTS_PREFIX + admission_id

    def upsert(conn, key, value):
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) '
                     'ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, value))

    def write(conn):
        _epoch(conn, epoch)
        row = _admission(conn, admission_id)
        if type(generation) is not int:
            raise AuthorityRefusal('stale_generation')
        if row['status'] != 'unknown' or row['generation'] != generation:
            if row['status'] in ('terminal', 'queued', 'started'):
                return _row(row), False
            raise AuthorityRefusal('stale_generation')
        if notice is not None:
            close_discarded_turn(holder, conn, row, notice)
        _retire_admission_workers(conn, row, row['owner_epoch'])
        if action == 'requeue':
            found = conn.execute('SELECT value FROM state_meta WHERE key=?', (attempts_key,)).fetchone()
            try:
                count = int(found[0]) + 1 if found else 1
            except (TypeError, ValueError):
                count = 1_000
            upsert(conn, attempts_key, str(count))
            conn.execute("UPDATE session_admissions SET status='queued',outcome=NULL,generation=NULL "
                         "WHERE admission_id=?", (admission_id,))
        else:
            outcome = 'interrupted'
            if action == 'completed':
                if _captured is None:
                    raise AuthorityRefusal('invalid_params')
                outcome, stored = _captured
                if outcome not in ('completed', 'failed', 'interrupted'):
                    raise AuthorityRefusal('invalid_params')
                upsert(conn, RESULT_PREFIX + admission_id, _json(encode_result(compact_result(
                    stored, user_message=json.loads(row['payload_json']).get('text')))))
                if owed is not None:
                    upsert(conn, RECOVERY_OWED_PREFIX + admission_id, owed)
            conn.execute("UPDATE session_admissions SET status='terminal',outcome=? WHERE admission_id=?",
                         (outcome, admission_id))
        conn.execute('UPDATE sessions SET runtime_revision=runtime_revision+1 WHERE id=?', (row['target_session_id'],))
        return _row(_admission(conn, admission_id)), True
    return holder._execute_write(write)


def recovery_attempts(holder, *, admission_id: str) -> int:
    """The automatic reruns restart recovery committed for the admission; an unreadable count reads 1000."""
    from persistence.authority.admissions import RECOVERY_ATTEMPTS_PREFIX
    with holder._read_ctx() as conn:
        found = conn.execute('SELECT value FROM state_meta WHERE key=?',
                             (RECOVERY_ATTEMPTS_PREFIX + admission_id,)).fetchone()
    try:
        return int(found[0]) if found else 0
    except (TypeError, ValueError):
        return 1_000


def claim_recovery_notice(holder, *, admission_id: str) -> bool:
    """The restart notice for the admission, at most once, durable across restarts: True for the first claim."""
    import time
    from persistence.authority.admissions import RECOVERY_NOTIFIED_PREFIX
    def write(conn):
        return conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                            (RECOVERY_NOTIFIED_PREFIX + admission_id, str(time.time()))).rowcount == 1
    return holder._execute_write(write)


def owed_recovery_answers(holder) -> list[tuple[str, str]]:
    """``(admission_id, owed record)`` of every answer a recovery owes, in key order."""
    from persistence.authority.admissions import RECOVERY_OWED_PREFIX
    with holder._read_ctx() as conn:
        rows = conn.execute("SELECT key, value FROM state_meta WHERE substr(key, 1, ?)=? ORDER BY key",
                            (len(RECOVERY_OWED_PREFIX), RECOVERY_OWED_PREFIX)).fetchall()
    return [(key[len(RECOVERY_OWED_PREFIX):], value) for key, value in rows]


def clear_recovery_owed(holder, *, admission_id: str) -> None:
    from persistence.authority.admissions import RECOVERY_OWED_PREFIX
    holder._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?',
                                                    (RECOVERY_OWED_PREFIX + admission_id,)))


def _recovery_chain(conn, session_id):
    """The session's compression chain, then its local physical target's chain (each id once)."""
    from ._local import local_physical_target
    ids = list(_canonical_chain(conn, session_id))
    for sid in _canonical_chain(conn, local_physical_target(conn, session_id)):
        if sid not in ids:
            ids.append(sid)
    return ids


def lost_turn_transcript(holder, *, session_id: str, event_id: str | None, since: float | None,
                         admission_id: str | None = None) -> dict:
    """One read of what a lost turn saved, over the session's compression and local lineage.

    The anchor is the newest ``user`` row whose ``platform_message_id`` is ``event_id``. Anchored: the rows after it
    in id order. Unanchored: whether an ``assistant`` or ``tool`` row was saved at or after ``since`` and, when
    *admission_id*'s claim stored a transcript mark, above that mark (a row written before the claim is not this
    turn's, whatever its timestamp)."""
    with holder._read_ctx() as conn:
        sessions = _recovery_chain(conn, session_id)
        marks = ','.join('?' * len(sessions))
        anchor = None
        if event_id:
            anchor = conn.execute(
                f"SELECT id, timestamp FROM messages WHERE session_id IN ({marks}) AND role='user' "
                "AND platform_message_id=? ORDER BY id DESC LIMIT 1", (*sessions, event_id)).fetchone()
        if anchor is None:
            claim = conn.execute('SELECT transcript_mark FROM session_admissions WHERE admission_id=?',
                                 (admission_id,)).fetchone() if admission_id else None
            after = claim[0] if claim is not None and claim[0] is not None else -1
            ran = since is not None and conn.execute(
                f"SELECT 1 FROM messages WHERE session_id IN ({marks}) AND timestamp>=? AND id>? "
                "AND role IN ('assistant','tool') LIMIT 1", (*sessions, since, after)).fetchone() is not None
            return {'anchored': False, 'anchor_timestamp': None, 'ran_since': ran, 'rows': ()}
        rows = conn.execute(
            f"SELECT role, content, tool_calls, tool_call_id, display_kind, timestamp, active, finish_reason "
            f"FROM messages WHERE session_id IN ({marks}) AND id>? ORDER BY id", (*sessions, anchor[0])).fetchall()
    return {'anchored': True, 'anchor_timestamp': anchor[1], 'ran_since': False, 'rows': tuple(
        dict(role=r[0], content=r[1], tool_calls=r[2], tool_call_id=r[3], display_kind=r[4], timestamp=r[5],
             active=bool(r[6]), finish_reason=r[7]) for r in rows)}
