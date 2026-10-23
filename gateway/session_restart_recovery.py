"""A person's messaging turn lost across an owner restart continues without /discard or /reset.

Boot recovery fences every started admission ``unknown`` and the session FIFO pauses on it.
Unattended automation already settles itself (``session_unattended_unknown``). A person's
messaging turn used to wait for ``/hermes discard`` or ``/reset``; this module decides from the
saved transcript, never from the prompt wording, what the lost turn actually did:

* ``answered``: its final answer was saved after the turn's own user row (anchored by the
  native ``event.message_id`` = that row's ``platform_message_id``). The turn settles
  ``completed`` with that answer and the answer is sent once. No inference runs again.
* ``unstarted``: no user row for the event exists and no tool call was saved since the event:
  nothing ran, so the row goes back to its own FIFO place and runs (once).
* ``readonly``: at least one tool call was saved and every one is a replay-safe read
  (``REPLAY_SAFE_TOOL_NAMES``): a fresh turn reruns in place, after a transcript boundary (once).
  A turn with its user row and NO saved call is not proof of anything: a runtime that saves its
  calls only when the turn ends (codex app-server) may already have written. It is ``uncertain``.
* ``uncertain``: anything else. Nothing re-runs on its own. The same person's next message
  continues it: the lost turn closes with a boundary naming the steps whose outcome is unknown,
  so the model checks before repeating them. Another person's message closes it with a neutral
  boundary that does not ask the model to continue it. A fresh one gets one short thread notice.

A saved answer is owed in the SAME fenced transaction that settles the turn ``completed`` (a
``state_meta`` marker in the profile store): a /discard that wins, or a write that fails, leaves
nothing owed, so a retry starts clean. After the commit the answer is handed to the delivery
ledger once (deterministic id, deduplicated against a reply the live path or the boot crash-left
pass already ledgered) and the marker cleared; the ledger's sweeps own every later retry. A crash
between the commit and the ledger leaves the marker, which the next boot delivers.

Automatic runs (rerun in place) are capped at one per admission, need a fresh turn
(``HERMES_AUTO_CONTINUE_FRESHNESS``, 1 h) and never run while ``/pause`` (estop) is engaged.
Scope: rows with a ``native_text_v1`` envelope from a person. Local CLI/TUI/Desktop viewers keep
their Discard UI; unattended rows keep GW-10 settlement.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

AUTO_RERUN_CAP = 1
DEFERRED = 'deferred'  # transient storage: the drain backs off and decides again
ANSWER_MAX_AGE_S = 24 * 60 * 60  # the delivery ledger's own stale cutoff
ATTEMPTS_PREFIX = 'restart_recovery_attempts:'
OWED_PREFIX = 'restart_recovery_owed:'
LEDGER_TRIES = 3
LEDGER_BACKOFF_S = 1.0  # doubled per try
# finish_reason of a saved draft the stop gates sent back for more work: not a final answer.
DRAFT_FINISH_REASONS = frozenset({'verification_required', 'verify_hook_continue', 'kanban_terminal_required'})


def replay_safe_tool_names():
    """No-effect tools whose replay is safe. ``browser_console`` runs arbitrary JavaScript (it can
    click, submit or navigate), so it never counts as read-only for an automatic rerun."""
    from agent.tool_result_classification import NO_EFFECT_TOOL_NAMES
    return NO_EFFECT_TOOL_NAMES - {'browser_console'}
NOTIFIED_PREFIX = 'restart_recovery_notified:'


@dataclass(frozen=True)
class Verdict:
    kind: str                      # answered | unstarted | readonly | uncertain
    answer: str | None = None      # answered: the saved final text (sanitized at delivery)
    unsettled: tuple = ()          # tool calls saved without a result: outcome unknown
    completed_effects: tuple = ()  # effect-capable calls that did return
    last_seen: float | None = None
    anchored: bool = False
    started: float | None = None   # the anchored user row's time (the turn start)


def is_recoverable_person(row) -> bool:
    from gateway.session_unattended_unknown import is_unattended
    return (row.get('status') == 'unknown' and 'native_text_v1' in (row.get('payload') or {})
            and not is_unattended(row))


def has_recoverable_person(rows) -> bool:
    return any(is_recoverable_person(row) for row in rows)


def freshness_window() -> float:
    from gateway.session_lifecycle import auto_continue_freshness_window
    return auto_continue_freshness_window()


def _event_time(payload) -> float | None:
    """The person's send time: an epoch-like platform id (Slack ``ts``), else the envelope time."""
    native = payload.get('native_text_v1') or {}
    for value in ((native.get('event') or {}).get('message_id'), (native.get('source') or {}).get('message_id')):
        try:
            stamp = float(str(value))
        except (TypeError, ValueError):
            continue
        if 1e9 < stamp < 1e11:
            return stamp
    from datetime import datetime
    try:
        stamp = datetime.fromisoformat(native['timestamp']).timestamp()
    except (KeyError, TypeError, ValueError):
        return None
    return stamp if stamp > 0 else None


def _chain(conn, session_id):
    from hermes_state_local_lineage import local_physical_target
    from hermes_state_runtime import _canonical_chain
    ids = list(_canonical_chain(conn, session_id))
    for sid in _canonical_chain(conn, local_physical_target(conn, session_id)):
        if sid not in ids:
            ids.append(sid)
    return ids


def _calls(tool_calls):
    try:
        calls = json.loads(tool_calls) if isinstance(tool_calls, str) else (tool_calls or [])
    except ValueError:
        return [('', '?')]
    out = []
    for call in calls if isinstance(calls, list) else []:
        function = (call or {}).get('function') or {}
        out.append(((call or {}).get('id') or (call or {}).get('call_id') or '', function.get('name') or '?'))
    return out


def classify(db, row) -> Verdict:
    """What the lost turn did, read from the saved transcript (read-only)."""
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND
    safe = replay_safe_tool_names()
    payload = row['payload']
    event_id = str(((payload.get('native_text_v1') or {}).get('event') or {}).get('message_id') or '')
    sent_at = _event_time(payload)
    with db._read_ctx() as conn:
        sessions = _chain(conn, row['target_session_id'])
        marks = ','.join('?' * len(sessions))
        anchor = None
        if event_id:
            anchor = conn.execute(
                f"SELECT id, timestamp FROM messages WHERE session_id IN ({marks}) AND role='user' "
                "AND platform_message_id=? ORDER BY id DESC LIMIT 1", (*sessions, event_id)).fetchone()
        if anchor is None:
            # No row of this turn: it never started, unless a tool call was saved after the person
            # sent it (a compaction can drop the anchor). Without a send time nothing is proven.
            if sent_at is None:
                return Verdict('uncertain')
            # Any reply or step saved since then (an answer without tools included) means it ran.
            ran = conn.execute(
                f"SELECT 1 FROM messages WHERE session_id IN ({marks}) AND timestamp>=? "
                "AND role IN ('assistant','tool') LIMIT 1", (*sessions, sent_at - 1)).fetchone()
            return Verdict('uncertain' if ran else 'unstarted', last_seen=sent_at)
        rows = conn.execute(
            f"SELECT role, content, tool_calls, tool_call_id, display_kind, timestamp, active, finish_reason FROM messages "
            f"WHERE session_id IN ({marks}) AND id>? ORDER BY id", (*sessions, anchor[0])).fetchall()
    last_seen = max([anchor[1] or 0] + [r[5] or 0 for r in rows]) or sent_at
    visible = [r for r in rows if r[6] and r[0] not in ('session_meta', 'system')]
    tail = visible[-1] if visible else None
    if (tail is not None and tail[0] == 'assistant' and tail[2] in (None, '', '[]')
            and isinstance(tail[1], str) and tail[1].strip() and tail[4] != FAILED_TURN_DISPLAY_KIND
            and tail[7] not in DRAFT_FINISH_REASONS):
        return Verdict('answered', answer=tail[1], last_seen=last_seen, anchored=True, started=anchor[1])
    calls = [call for r in rows if r[0] == 'assistant' and r[2] not in (None, '', '[]') for call in _calls(r[2])]
    answered_ids = {r[3] for r in rows if r[0] == 'tool' and r[3]}
    unsettled = tuple(dict.fromkeys(name for cid, name in calls if cid not in answered_ids))
    effects = tuple(dict.fromkeys(name for cid, name in calls if cid in answered_ids and name not in safe))
    if calls and all(name in safe for _cid, name in calls):
        return Verdict('readonly', unsettled=unsettled, last_seen=last_seen, anchored=True, started=anchor[1])
    return Verdict('uncertain', unsettled=unsettled, completed_effects=effects, last_seen=last_seen,
                   anchored=True, started=anchor[1])


def _names(names):
    return ', '.join(names)


def continue_notice(verdict: Verdict) -> str:
    """The boundary that closes a lost turn a person's next message continues (model-visible)."""
    if not verdict.anchored:
        return ('Hermes restarted around the previous message, and it may not have run. '
                'Answer the newer message below; ask if the earlier one is still needed.')
    parts = ['Hermes restarted before this request finished.']
    effect_unknown = [n for n in verdict.unsettled]
    if effect_unknown:
        parts.append(f'Steps whose result is unknown: {_names(effect_unknown)}. '
                     'Check what they did before repeating them.')
    if verdict.completed_effects:
        parts.append(f'Steps that already completed: {_names(verdict.completed_effects)}. Do not repeat them.')
    if not effect_unknown and not verdict.completed_effects:
        parts.append('No unfinished step was recorded. Check before repeating anything that changes data.')
    parts.append('Continue the request from here.')
    return ' '.join(parts)


def other_person_notice(verdict: Verdict) -> str:
    """The boundary when someone else's message closes the lost turn: it is not theirs to continue."""
    parts = ['Hermes restarted before an earlier request from another person in this conversation finished.']
    if verdict.unsettled:
        parts.append(f'Steps whose result is unknown: {_names(verdict.unsettled)}.')
    parts.append('Do not continue that request unless the person who asked for it does. Answer the newer message.')
    return ' '.join(parts)


def rerun_notice(_verdict: Verdict) -> str:
    return ('Hermes restarted before this request finished. The steps so far only read information, '
            'so the request is running again.')


def thread_notice(verdict: Verdict) -> str:
    """One short message to the person when their fresh turn cannot continue on its own."""
    steps = (f' These steps may or may not have finished: {_names(verdict.unsettled)}.'
             if verdict.unsettled else '')
    return ('♻️ Hermes restarted while working on this.' + steps
            + ' Reply here (for example "continue") and I will pick it up and check anything that may already have run.')


def attempts(db, admission_id) -> int:
    with db._read_ctx() as conn:
        found = conn.execute('SELECT value FROM state_meta WHERE key=?', (ATTEMPTS_PREFIX + admission_id,)).fetchone()
    try:
        return int(found[0]) if found else 0
    except (TypeError, ValueError):
        return AUTO_RERUN_CAP


def _claim_notice(db, admission_id) -> bool:
    """At most once per admission, durable across restarts."""
    def write(conn):
        return conn.execute('INSERT OR IGNORE INTO state_meta(key,value) VALUES(?,?)',
                            (NOTIFIED_PREFIX + admission_id, str(time.time()))).rowcount == 1
    return db._execute_write(write)


async def send_to_thread(authority, row, text) -> bool:
    """Send ``text`` to the conversation the admission came from (the person's own thread)."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_envelope import restore_native_source
    from gateway.session_ingress import _execution_scope, deliver_response
    native = row['payload']['native_text_v1']
    runner = authority.runner
    source = restore_native_source(row['payload'], runner)
    adapter = runner._adapter_for_source(source)
    if adapter is None:
        return False
    event = MessageEvent(text='', source=source, message_id=(native.get('event') or {}).get('message_id'))
    home = None
    if native.get('provenance') is not None:
        from gateway.session_ingress_context import restore_provenance
        home = restore_provenance(runner, source, native['provenance'])
    with await _execution_scope(authority, home):
        await deliver_response(adapter, event, native.get('route'), text)
    return True


def _profile_label(source, adapter):
    """The ledger's ``adapter_profile``, named exactly as the live send names it
    (``record_obligation``: the adapter's ``_owner_profile`` or ``'default'``), so the boot,
    reconnect and flood sweeps all claim the row. Primary adapters (every single-profile gateway,
    named home included) are ``'default'``; with no adapter, the source's multiplex transport."""
    owner = getattr(adapter, '_owner_profile', None) if adapter is not None else None
    if adapter is None:
        from gateway.session_identity import transport_profile_of
        owner = transport_profile_of(source)
    return str(owner).strip() if owner and str(owner).strip() != 'default' else 'default'


def _ledger_target(authority, row):
    from gateway.session_envelope import restore_native_source
    native = row['payload']['native_text_v1']
    source = restore_native_source(row['payload'], authority.runner)
    try:
        adapter = authority.runner._adapter_for_source(source)
    except Exception:
        adapter = None
    return dict(session_key=native.get('route') or '', platform=str(getattr(source.platform, 'value', source.platform)),
                chat_id=str(source.chat_id), thread_id=getattr(source, 'thread_id', None),
                profile=_profile_label(source, adapter))


def owed_record(authority, row, text, since) -> str:
    """What the commit owes the person: the presented answer and its ledger target (JSON)."""
    return json.dumps({'admission_id': row['admission_id'], 'text': text, 'since': since,
                       **_ledger_target(authority, row)})


def _list_owed(db):
    with db._read_ctx() as conn:
        rows = conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ?', (OWED_PREFIX + '%',)).fetchall()
    out = []
    for key, value in rows:
        try:
            out.append((key[len(OWED_PREFIX):], json.loads(value)))
        except (TypeError, ValueError):
            logger.warning('Unreadable owed restart answer %s', key)
    return out


def _clear_owed(db, admission_id):
    db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (OWED_PREFIX + admission_id,)))


def _ledger_owed(admission_id, owed):
    """Hand the owed answer to the delivery ledger. Returns the obligation id this call must send,
    None when another row already owns it (sent, or being sent by a sweep), or '' when the ledger is
    off (send best-effort, at most once). ``inserted`` is True once per id, so it alone decides who
    sends; clearing the marker is separate (``_clear_owed``) and its failure is harmless."""
    from gateway.delivery_ledger import compute_obligation_id, ledger_enabled, record_recovered_reply
    if not ledger_enabled():
        return ''
    obligation_id = compute_obligation_id(owed['session_key'], 'restart-recovery:' + admission_id, owed['text'])
    inserted = record_recovered_reply(
        obligation_id=obligation_id, session_key=owed['session_key'], platform=owed['platform'],
        chat_id=owed['chat_id'], thread_id=owed['thread_id'], content=owed['text'],
        since=(owed.get('since') or 0) - 1, adapter_profile=owed['profile'])
    return obligation_id if inserted else None


async def _retry_storage(what, admission_id, fn, *args):
    """``fn(*args)`` off the loop, retried on transient storage; raises the last error."""
    import asyncio
    for attempt in range(LEDGER_TRIES):
        try:
            return await asyncio.to_thread(fn, *args)
        except (OSError, sqlite3.OperationalError) as exc:
            if attempt == LEDGER_TRIES - 1:
                raise
            logger.warning('%s for admission %s deferred (%r)', what, admission_id, exc)
            await asyncio.sleep(LEDGER_BACKOFF_S * 2 ** attempt)


async def deliver_owed(authority, admission_id, owed) -> bool:
    """Ledger then send one owed answer. False when the ledger write kept failing: the marker stays
    and the next boot delivers it (never lost, never twice)."""
    try:
        obligation = await _retry_storage('Owed answer ledger', admission_id, _ledger_owed, admission_id, owed)
    except (OSError, sqlite3.OperationalError):
        logger.warning('Owed answer of admission %s left for the next boot', admission_id)
        return False
    try:
        # Ledger on: a marker left behind is harmless (the next boot's insert finds the row and
        # only clears it). Ledger off: the marker is the only once-guard, so no clear, no send.
        await _retry_storage('Owed marker clear', admission_id, _clear_owed, authority.db, admission_id)
    except (OSError, sqlite3.OperationalError):
        if obligation == '':
            logger.warning('Owed answer of admission %s (no ledger) left for the next boot', admission_id)
            return False
        logger.warning('Owed marker of admission %s not cleared; the next boot clears it', admission_id)
    from gateway.delivery_ledger import RECOVERED_MARKER
    if obligation == '':
        await _send_owed(authority, owed, RECOVERED_MARKER + owed['text'])
        return True
    if obligation is None:
        return True
    redeliver = getattr(authority.runner, '_redeliver_claimed_obligations', None)
    claimed = {'obligation_id': obligation, 'session_key': owed['session_key'], 'platform': owed['platform'],
               'chat_id': owed['chat_id'], 'thread_id': owed['thread_id'], 'content': owed['text'],
               'profile': owed['profile'], 'attempts': 1, 'needs_marker': True, 'marker': RECOVERED_MARKER}
    try:
        if callable(redeliver):
            await redeliver([claimed])
        else:  # no ledger-aware runner (bare test runners)
            await _send_owed(authority, owed, RECOVERED_MARKER + owed['text'])
    except Exception as exc:
        # Left 'attempting' under this live process nothing would retry it before the next boot:
        # 'failed' hands it to the reconnect and timer sweeps.
        from gateway.delivery_ledger import mark_failed
        logger.exception('Owed answer of admission %s: send raised', admission_id)
        await asyncio.to_thread(mark_failed, obligation, f'send raised: {type(exc).__name__}')
    return True


async def _send_owed(authority, owed, text):
    """Best-effort direct send (no ledger): to the admission's own thread. At boot the owed record
    carries no row, so it is read back from the store (the settled admission keeps its payload)."""
    import asyncio
    row = owed.get('_row')
    if row is None:
        from hermes_state_runtime import get_session_admission
        row = await asyncio.to_thread(get_session_admission, authority.db, admission_id=owed['admission_id'])
    if row is None or 'native_text_v1' not in (row.get('payload') or {}):
        logger.warning('Owed answer of admission %s has no thread to send to', owed.get('admission_id'))
        return
    await send_to_thread(authority, row, text)


async def deliver_owed_answers(authority) -> int:
    """Boot: deliver answers a previous process committed but never handed to the ledger."""
    import asyncio
    delivered = 0
    for admission_id, owed in await asyncio.to_thread(_list_owed, authority.db):
        try:
            if await deliver_owed(authority, admission_id, owed):
                delivered += 1
        except Exception:
            logger.exception('Owed answer of admission %s not delivered at boot', admission_id)
    return delivered


def _engaged(authority) -> bool:
    """/pause (estop) as the OWNING profile sees it (multiplex: its own ESTOP file)."""
    from agent.estop import is_engaged
    from gateway.session_authorities import owner_scope
    try:
        with owner_scope(authority):
            return is_engaged()
    except Exception:  # fail safe, like is_engaged itself: no automatic rerun
        logger.warning('Could not read /pause for %s; holding automatic reruns', authority.profile_id, exc_info=True)
        return True


def _same_person(head, row) -> bool:
    def who(r):
        source = ((r.get('payload') or {}).get('native_text_v1') or {}).get('source') or {}
        return (r.get('principal_id'), source.get('user_id'))
    mine = who(head)
    return mine[1] is not None and who(row) == mine


def _answer_text(authority, row, verdict):
    """The saved answer as live delivery would present it ('' = nothing to send)."""
    text = verdict.answer or ''
    crash_left = getattr(authority.runner, '_crash_left_reply', None)
    if callable(crash_left):
        try:
            from gateway.session_envelope import restore_native_source
            origin = restore_native_source(row['payload'], authority.runner)
            presented = crash_left([{'role': 'assistant', 'content': text, 'timestamp': time.time()}], 0, origin)
            return presented if isinstance(presented, str) else ''
        except Exception:
            logger.debug('Saved-answer presentation fell back to raw text', exc_info=True)
    return text.strip()


async def _commit(authority, row, action, *, notice=None, stored=None, owed=None):
    from functools import partial

    from gateway.session_results import close_discarded_turn
    from gateway.session_runtime_workers import tracked_write
    from hermes_state_runtime import recover_unknown_session_input
    db = authority.db
    write = None if notice is None else (lambda conn, lost: close_discarded_turn(db, conn, lost, notice))
    return await tracked_write(authority, partial(
        recover_unknown_session_input, db, epoch=authority.epoch, admission_id=row['admission_id'],
        generation=row['generation'], action=action, stored=stored, owed=owed, _terminal_write=write))


async def recover_person_unknown(authority, ref) -> bool:
    """The drain found an ``unknown`` head. True when this changed the FIFO (the drain re-reads),
    False to keep it paused, ``DEFERRED`` after a transient storage error (back off, re-read). Runs at boot recovery (adapters connected) and on every new input."""
    import asyncio

    from gateway.session_unattended_unknown import is_unattended
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions
    rows = await asyncio.to_thread(list_session_admissions, authority.db, session_id=ref.session_id)
    rows = sorted(rows, key=lambda r: r['seq'])
    head = next((r for r in rows if r['status'] == 'unknown'), None)
    if head is None or not is_recoverable_person(head):
        return False
    if head['owner_epoch'] == authority.epoch and head['admission_id'] in authority.pending_results:
        # Fenced by THIS owner after its settlement write failed: the exact captured answer and its
        # retained delivery are here, and the existing resolution commits and sends them once.
        try:
            await authority.resolve_unknown_as_owner(ref, head['admission_id'], head['generation'])
        except RuntimeStoreError as exc:
            logger.warning('Lost turn %s not recovered: %s', head['admission_id'], exc.reason)
            return exc.reason == 'stale_generation'
        return True
    verdict = await asyncio.to_thread(classify, authority.db, head)
    now = time.time()
    window = freshness_window()

    def spoke_since(r):
        # A person's message that continues the lost turn: admitted by THIS owner (sent after the
        # restart) or sent within the freshness window. Older queued messages never trigger
        # automatic execution at boot; they wait, as before, for the person's next message.
        if r['status'] != 'queued' or r['seq'] <= head['seq'] or is_unattended(r):
            return False
        sent = _event_time(r['payload']) if 'native_text_v1' in r['payload'] else None
        return r.get('owner_epoch') == authority.epoch or (sent is not None and now - sent <= window)
    followers = [r for r in rows if spoke_since(r)]
    try:
        if verdict.kind == 'answered':
            from gateway.session_results import prepare_result
            text = _answer_text(authority, head, verdict)
            stored = prepare_result(head, verdict.answer or '', 'completed')
            age = now - (verdict.last_seen or now)
            owed = owed_record(authority, head, text, verdict.started) if text and age <= ANSWER_MAX_AGE_S else None
            # The answer is owed in the same fenced write that settles the turn: a /discard that
            # wins, or a failed write, owes nothing, and the retry starts clean.
            committed = await _commit(authority, head, 'completed', stored=stored, owed=owed)
            if not committed.get('recovered_now'):
                return True  # the person dropped this turn (or it settled elsewhere)
            if owed is not None:
                record = json.loads(owed)
                record['_row'] = head
                await deliver_owed(authority, head['admission_id'], record)
            logger.info('Lost turn %s had a saved answer: settled completed%s', head['admission_id'],
                        '' if owed else ' (nothing sent: empty or older than 24 h)')
            return True
        fresh = verdict.last_seen is not None and now - verdict.last_seen <= window
        # /pause (estop) holds every AUTOMATIC run; a person's own new message still continues.
        may_rerun = (fresh and not await asyncio.to_thread(_engaged, authority)
                     and await asyncio.to_thread(attempts, authority.db, head['admission_id']) < AUTO_RERUN_CAP)
        if may_rerun and (verdict.kind == 'unstarted' or (verdict.kind == 'readonly' and not followers)):
            await _commit(authority, head, 'requeue',
                          notice=rerun_notice(verdict) if verdict.anchored else None)
            logger.info('Lost turn %s (%s) runs again in its FIFO place', head['admission_id'], verdict.kind)
            return True
        if followers:
            # Only the person who asked continues their request; anyone else's message just moves on.
            notice = continue_notice(verdict) if _same_person(head, followers[0]) else other_person_notice(verdict)
            await _commit(authority, head, 'interrupted', notice=notice)
            from gateway.session_ingress_media import release_admission_media
            await asyncio.to_thread(release_admission_media, authority.db, head['admission_id'])
            logger.info('Lost turn %s (%s) closed; the next message runs', head['admission_id'], verdict.kind)
            return True
        if fresh and await asyncio.to_thread(_claim_notice, authority.db, head['admission_id']):
            # Re-check: a /discard may have settled the turn while this decided.
            from hermes_state_runtime import get_session_admission
            current = await asyncio.to_thread(get_session_admission, authority.db, admission_id=head['admission_id'])
            if current is None or current['status'] != 'unknown':
                return True
            try:
                await send_to_thread(authority, head, thread_notice(verdict))
            except Exception:
                logger.warning('Restart notice for admission %s not sent', head['admission_id'], exc_info=True)
        return False
    except RuntimeStoreError as exc:
        logger.warning('Lost turn %s not recovered: %s', head['admission_id'], exc.reason)
        return exc.reason == 'stale_generation'  # resolved elsewhere meanwhile: re-read
    except (OSError, sqlite3.OperationalError) as exc:
        # Transient storage (a writer held past its patience): nothing was committed or owed. The
        # drain backs off (and stops on shutdown) before it decides again.
        logger.warning('Lost turn %s recovery deferred by storage (%r)', head['admission_id'], exc)
        return DEFERRED
