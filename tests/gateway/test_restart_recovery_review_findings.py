"""Review findings on restart recovery of a person's lost messaging turn, each pinned by a test.

Same harness as ``test_restart_recovery_person_turns`` (real SessionStore SQLite, two owner epochs).
"""
import contextlib

import pytest

from hermes_state_runtime import admit_session_input, list_session_admissions
from tests.gateway.test_restart_recovery_person_turns import Harness, _call, _native, _now_id


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    yield h
    h.close()


def _head_row(harness, admission_id):
    return next(r for r in list_session_admissions(harness.db, session_id='s', pending_only=False)
                if r['admission_id'] == admission_id)


# 1. A saved user row with no saved call is not proof of a read-only turn (codex app-server saves
#    its exec/apply_patch calls only when the turn ends).
@pytest.mark.asyncio
async def test_anchored_turn_without_saved_calls_is_never_rerun(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[{'role': 'user', 'content': 'HEAD', 'platform_message_id': eid}])
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)
    assert len(harness.sent) == 1  # one thread notice instead of a blind rerun


# 7. browser_console runs arbitrary JavaScript: never replay-safe.
@pytest.mark.asyncio
async def test_browser_console_is_not_replay_safe(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('a', 'read_file'), _call('b', 'browser_console')]}])
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)


# 2. Another person's message never continues someone else's lost turn.
@pytest.mark.asyncio
async def test_another_persons_message_does_not_continue_the_lost_turn(harness):
    eid = _now_id(-120)
    harness_payload = _native('other person', _now_id(-30), user='U2')
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]}])
    admit_session_input(harness.db, epoch=harness.owner.epoch, principal_id='messaging:person', session_id='s',
                        request_id='u2', payload=harness_payload)
    await harness.drain()
    assert harness.status(head) == ('terminal', 'interrupted')
    assert harness.executed == ['other person']
    (boundary,) = harness.boundaries()
    assert 'another person' in boundary and 'Do not continue' in boundary
    assert 'Continue the request from here' not in boundary


@pytest.mark.asyncio
async def test_same_persons_message_continues_the_lost_turn(harness):
    eid = _now_id(-120)
    _head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]}])
    admit_session_input(harness.db, epoch=harness.owner.epoch, principal_id='messaging:person', session_id='s',
                        request_id='u1', payload=_native('continue', _now_id(), user='U1'))
    await harness.drain()
    (boundary,) = harness.boundaries()
    assert 'Continue the request from here' in boundary


# 3. A /discard that commits while recovery decides wins: the dropped answer is never sent.
@pytest.mark.asyncio
async def test_discard_during_recovery_suppresses_the_saved_answer(harness, monkeypatch):
    from gateway import session_restart_recovery as recovery
    from hermes_state_runtime import resolve_unknown_session_input
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    real = recovery.classify

    def classify_then_discard(db, row):
        verdict = real(db, row)
        resolve_unknown_session_input(db, epoch=harness.owner.epoch, admission_id=row['admission_id'],
                                      generation=row['generation'])
        return verdict
    monkeypatch.setattr(recovery, 'classify', classify_then_discard)
    await harness.drain()
    assert harness.sent == []
    assert harness.status(head) == ('terminal', 'interrupted')
    assert _ledger_states() == [] and _owed(harness) == []  # nothing was ever owed


def test_store_reports_whether_this_call_recovered(tmp_path, monkeypatch):
    import asyncio
    from hermes_state_runtime import recover_unknown_session_input
    h = Harness(tmp_path, monkeypatch)
    try:
        head, _ = asyncio.run(h.lose_head(_now_id(-30)))
        row = _head_row(h, head)
        first = recover_unknown_session_input(h.db, epoch=h.owner.epoch, admission_id=head,
                                              generation=row['generation'], action='interrupted')
        again = recover_unknown_session_input(h.db, epoch=h.owner.epoch, admission_id=head,
                                              generation=row['generation'], action='interrupted')
        assert first['recovered_now'] is True and again['recovered_now'] is False
    finally:
        h.close()


# 5. The saved answer is ledgered: a reply the boot crash-left pass already ledgered is not sent
#    twice, and a ledgered answer survives a failed send.
@pytest.mark.asyncio
async def test_saved_answer_already_ledgered_is_not_sent_twice(harness):
    from gateway.delivery_ledger import record_crash_left_reply
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    record_crash_left_reply(obligation_id='boot', session_key='route-s', platform='slack', chat_id='C1',
                            thread_id='T1', content='THE SAVED ANSWER', since=0)
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_saved_answer_goes_through_the_ledger_redelivery(harness):
    from gateway.delivery_ledger import RECOVERED_MARKER, _connect
    claimed = []

    async def redeliver(rows):
        claimed.extend(rows)
        return 0  # the send failed: the row must stay in the ledger for the sweeps
    harness.runner._redeliver_claimed_obligations = redeliver
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    (row,) = claimed
    assert row['content'] == 'THE SAVED ANSWER' and row['marker'] == RECOVERED_MARKER and row['thread_id'] == 'T1'
    with contextlib.closing(_connect()) as conn:
        assert conn.execute('SELECT content FROM delivery_obligations WHERE obligation_id=?',
                            (row['obligation_id'],)).fetchone()[0] == 'THE SAVED ANSWER'


# Low: a draft the stop gates sent back for verification is not a final answer.
@pytest.mark.asyncio
async def test_draft_awaiting_verification_is_not_settled_as_the_answer(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]},
        {'role': 'tool', 'content': 'ok', 'tool_call_id': 'd', 'tool_name': 'terminal'},
        {'role': 'assistant', 'content': 'DRAFT', 'finish_reason': 'verification_required'}])
    await harness.drain()
    assert harness.status(head) == ('unknown', None)
    assert all('DRAFT' not in text for _id, text in harness.sent)


# Low: without an anchor, a reply saved since the send means the turn ran.
@pytest.mark.asyncio
async def test_unanchored_turn_with_a_saved_reply_is_not_unstarted(harness):
    head, _ = await harness.lose_head(_now_id(-60), transcript=[
        {'role': 'assistant', 'content': 'answered without tools'}])
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)


# 6. /pause is read in the owning profile's scope.
@pytest.mark.asyncio
async def test_estop_is_checked_in_the_owner_profile_scope(harness, monkeypatch):
    seen = []

    @contextlib.contextmanager
    def scope(authority):
        seen.append(authority.profile_id)
        yield
    monkeypatch.setattr('gateway.session_authorities.owner_scope', scope)
    _head, _ = await harness.lose_head(_now_id(-30))
    await harness.drain()
    assert seen == ['p'] and harness.executed == ['HEAD']


# 4. A message admitted while recovery decides is not stranded behind the old pause.
@pytest.mark.asyncio
async def test_message_admitted_during_recovery_is_not_stranded(harness, monkeypatch):
    from gateway import session_restart_recovery as recovery
    eid = _now_id(-120)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]}])
    real = recovery.recover_person_unknown
    calls = []

    async def racing(authority, ref):
        calls.append(1)
        if len(calls) == 1:  # the person speaks after recovery read the FIFO
            result = await real(authority, ref)
            admit_session_input(harness.db, epoch=authority.epoch, principal_id='messaging:person', session_id='s',
                                request_id='late', payload=_native('continue', _now_id()))
            authority._schedule(ref)
            return result
        return await real(authority, ref)
    monkeypatch.setattr(recovery, 'recover_person_unknown', racing)
    await harness.drain()
    assert harness.status(head) == ('terminal', 'interrupted')
    assert harness.executed == ['continue']


# Re-review: a transient commit failure must not lose the saved answer.
@pytest.mark.asyncio
async def test_saved_answer_survives_a_transient_commit_failure(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    from gateway.delivery_ledger import _connect
    claimed = []

    async def redeliver(rows):
        claimed.extend(rows)
        return len(rows)
    harness.runner._redeliver_claimed_obligations = redeliver
    real = recovery._commit
    failures = []

    async def flaky(authority, row, action, **kw):
        if action == 'completed' and not failures:
            failures.append(1)
            raise sqlite3.OperationalError('database is locked')
        return await real(authority, row, action, **kw)
    monkeypatch.setattr(recovery, '_commit', flaky)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    await harness.drain()
    assert failures == [1]
    assert harness.status(head) == ('terminal', 'completed')
    assert [r['content'] for r in claimed] == ['THE SAVED ANSWER']  # sent once, after the retry
    with contextlib.closing(_connect()) as conn:
        states = [r[0] for r in conn.execute('SELECT state FROM delivery_obligations')]
    assert states == ['attempting']  # owned by this process until the (stubbed) send marks it


def test_ledger_profile_matches_the_live_send_naming():
    from types import SimpleNamespace
    from gateway.session_restart_recovery import _profile_label
    primary = SimpleNamespace(_owner_profile=None)  # every primary adapter, named home included
    assert _profile_label(SimpleNamespace(), primary) == 'default'
    assert _profile_label(SimpleNamespace(), SimpleNamespace(_owner_profile='scout')) == 'scout'
    assert _profile_label(SimpleNamespace(), None) == 'default'


def test_a_ledgered_recovery_row_is_never_re_owned(tmp_path, monkeypatch):
    """Round 3 M1: a row a sweep claimed (and may be sending) is never taken over again."""
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.delivery_ledger import mark_failed, record_recovered_reply
    kw = dict(obligation_id='o1', session_key='k', platform='slack', chat_id='C', thread_id='T',
              content='A', since=0, adapter_profile='default')
    assert record_recovered_reply(**kw) is True
    assert record_recovered_reply(**kw) is False
    mark_failed('o1', 'flood')
    assert record_recovered_reply(**kw) is False  # the flood timer owns it, with its wait
    assert record_recovered_reply(**{**kw, 'obligation_id': 'o2'}) is False


def _ledger_states():
    from gateway.delivery_ledger import _connect
    with contextlib.closing(_connect()) as conn:
        return [r[0] for r in conn.execute('SELECT state FROM delivery_obligations')]


def _owed(harness):
    from gateway.session_restart_recovery import _list_owed
    return _list_owed(harness.db)


def _answered(harness, eid):
    return [{'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
            {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}]


# Round 3 M2: a transient commit failure, then a /discard in the retry window: nothing is sent.
@pytest.mark.asyncio
async def test_discard_after_a_transient_commit_failure_sends_nothing(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    from hermes_state_runtime import resolve_unknown_session_input
    real = recovery._commit
    failed = []

    async def flaky(authority, row, action, **kw):
        if action == 'completed' and not failed:
            failed.append(1)
            resolve_unknown_session_input(harness.db, epoch=authority.epoch, admission_id=row['admission_id'],
                                          generation=row['generation'])  # the person's /discard lands
            raise sqlite3.OperationalError('database is locked')
        return await real(authority, row, action, **kw)
    monkeypatch.setattr(recovery, '_commit', flaky)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert failed == [1] and harness.sent == []
    assert harness.status(head) == ('terminal', 'interrupted')
    assert _ledger_states() == [] and _owed(harness) == []


# Round 3 M3: the ledger write failing never loses the answer.
@pytest.mark.asyncio
async def test_ledger_write_failure_keeps_the_answer_owed_until_boot(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    import gateway.delivery_ledger as ledger
    real = ledger.record_recovered_reply

    def locked(**kw):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(ledger, 'record_recovered_reply', locked)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == [] and [aid for aid, _ in _owed(harness)] == [head]  # still owed
    # Next boot: the owed answer is delivered once, and only once.
    monkeypatch.setattr(ledger, 'record_recovered_reply', real)
    from gateway.session_restart_recovery import deliver_owed_answers
    harness.runner._redeliver_claimed_obligations = _capture(harness)
    assert await deliver_owed_answers(harness.owner) == 1
    assert await deliver_owed_answers(harness.owner) == 0
    assert [t for _a, t in harness.sent] == ['THE SAVED ANSWER'] and _owed(harness) == []


@pytest.mark.asyncio
async def test_boot_owed_answer_already_in_the_ledger_is_not_sent_again(harness):
    """Crash after the ledger insert but before the marker clear: the sweep owns the row."""
    from gateway.session_restart_recovery import deliver_owed_answers
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    harness.runner._redeliver_claimed_obligations = _capture(harness)
    await harness.drain()
    assert [t for _a, t in harness.sent] == ['THE SAVED ANSWER']
    import json as _json
    harness.db._execute_write(lambda c: c.execute(  # simulate the marker surviving the crash
        'INSERT INTO state_meta(key,value) VALUES(?,?)', ('restart_recovery_owed:' + head, _json.dumps(
            {'admission_id': head, 'text': 'THE SAVED ANSWER', 'since': 0, 'session_key': 'route-s',
             'platform': 'slack', 'chat_id': 'C1', 'thread_id': 'T1', 'profile': 'default'}))))
    await deliver_owed_answers(harness.owner)
    assert len(harness.sent) == 1 and _owed(harness) == []


def _capture(harness):
    async def redeliver(rows):
        for row in rows:
            harness.sent.append((row['obligation_id'], row['content']))
        return len(rows)
    return redeliver


# Round 4: the ledger insert landed but clearing the marker failed: the answer is still sent once.
@pytest.mark.asyncio
async def test_failed_marker_clear_after_the_ledger_insert_still_sends_once(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    real = recovery._clear_owed
    calls = []

    def flaky(db, admission_id):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError('database is locked')
        return real(db, admission_id)
    monkeypatch.setattr(recovery, '_clear_owed', flaky)
    harness.runner._redeliver_claimed_obligations = _capture(harness)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [t for _a, t in harness.sent] == ['THE SAVED ANSWER'] and _owed(harness) == []
    # A marker that survived anyway (clear never succeeded) is cleared at boot without a resend.
    from gateway.session_restart_recovery import deliver_owed_answers
    await deliver_owed_answers(harness.owner)
    assert len(harness.sent) == 1


@pytest.mark.asyncio
async def test_marker_that_cannot_be_cleared_never_resends(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)

    def locked(db, admission_id):
        raise sqlite3.OperationalError('database is locked')
    real = recovery._clear_owed
    monkeypatch.setattr(recovery, '_clear_owed', locked)
    harness.runner._redeliver_claimed_obligations = _capture(harness)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert [t for _a, t in harness.sent] == ['THE SAVED ANSWER']
    assert [aid for aid, _ in _owed(harness)] == [head]  # left behind
    monkeypatch.setattr(recovery, '_clear_owed', real)
    from gateway.session_restart_recovery import deliver_owed_answers
    await deliver_owed_answers(harness.owner)  # next boot: insert finds the row, clears, no send
    assert len(harness.sent) == 1 and _owed(harness) == []


# Round 5: ledger off, the marker is the only once-guard: a failed clear sends nothing now.
@pytest.mark.asyncio
async def test_ledger_off_failed_clear_never_sends_twice(harness, monkeypatch):
    import sqlite3
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    monkeypatch.setattr('gateway.delivery_ledger.ledger_enabled', lambda *a, **k: False)
    real = recovery._clear_owed
    locked = [True]

    def clear(db, admission_id):
        if locked[0]:
            raise sqlite3.OperationalError('database is locked')
        return real(db, admission_id)
    monkeypatch.setattr(recovery, '_clear_owed', clear)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == [] and [aid for aid, _ in _owed(harness)] == [head]
    from gateway.session_restart_recovery import deliver_owed_answers
    await deliver_owed_answers(harness.owner)  # still locked: still nothing sent
    assert harness.sent == []
    locked[0] = False
    await deliver_owed_answers(harness.owner)  # next boot, store healthy: sent once
    await deliver_owed_answers(harness.owner)
    assert len(harness.sent) == 1 and _owed(harness) == []


# KC review 2026-10-22: a redelivery that raises must hand the obligation to the sweeps
# (mark it failed), not crash with NameError (asyncio was not imported in deliver_owed).
@pytest.mark.asyncio
async def test_redelivery_that_raises_marks_the_obligation_failed(harness, monkeypatch):
    import gateway.delivery_ledger as ledger
    failed = []
    monkeypatch.setattr(ledger, 'mark_failed', lambda oid, error='': failed.append((oid, error)))

    async def boom(rows):
        raise RuntimeError('slack down')
    harness.runner._redeliver_claimed_obligations = boom
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert len(failed) == 1 and failed[0][1] == 'send raised: RuntimeError'


# The model-visible notices must not claim more than the evidence shows: an unanchored turn is
# "may not have run", not "was not run"; no recorded calls is not "no step was left half-done".
def test_restart_notices_do_not_overclaim():
    from gateway.session_restart_recovery import Verdict, continue_notice
    unanchored = continue_notice(Verdict('unstarted'))
    assert 'was not run' not in unanchored
    assert 'may not have' in unanchored
    anchored = continue_notice(Verdict('readonly', anchored=True))
    assert 'No step was left half-done' not in anchored
    assert 'Check' in anchored
