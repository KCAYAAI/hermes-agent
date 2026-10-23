"""Review findings on restart recovery of a person's lost messaging turn, each pinned by a test.

Same harness as ``test_restart_recovery_person_turns`` (real SessionStore SQLite, two owner epochs).
"""
import contextlib

import pytest

from persistence.core.errors import StoreContention
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.test_restart_recovery_person_turns import Harness, _call, _native, _now_id


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    yield h
    h.close()


def _head_row(harness, admission_id):
    return harness.row(admission_id)


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
    harness.admit(harness.owner.epoch, 'u2', harness_payload)
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
    harness.admit(harness.owner.epoch, 'u1', _native('continue', _now_id(), user='U1'))
    await harness.drain()
    (boundary,) = harness.boundaries()
    assert 'Continue the request from here' in boundary


# 3. A /discard that commits while recovery decides wins: the dropped answer is never sent.
@pytest.mark.asyncio
async def test_discard_during_recovery_suppresses_the_saved_answer(harness, monkeypatch):
    from gateway import session_restart_recovery as recovery
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    real = recovery.classify

    def classify_then_discard(store, row):
        verdict = real(store, row)
        store.admissions.resolve_unknown(epoch=harness.owner.epoch, admission_id=row['admission_id'],
                                         generation=row['generation'])
        return verdict
    monkeypatch.setattr(recovery, 'classify', classify_then_discard)
    await harness.drain()
    assert harness.sent == []
    assert harness.status(head) == ('terminal', 'interrupted')
    assert _ledger_states() == [] and _owed(harness) == []  # nothing was ever owed


def test_store_reports_whether_this_call_recovered(tmp_path, monkeypatch):
    import asyncio
    h = Harness(tmp_path, monkeypatch)
    try:
        head, _ = asyncio.run(h.lose_head(_now_id(-30)))
        row = _head_row(h, head)
        _first, first_recovered = h.db.admissions.recover_unknown(
            epoch=h.owner.epoch, admission_id=head, generation=row['generation'], action='interrupted')
        _again, again_recovered = h.db.admissions.recover_unknown(
            epoch=h.owner.epoch, admission_id=head, generation=row['generation'], action='interrupted')
        assert first_recovered is True and again_recovered is False
    finally:
        h.close()


# 5. The saved answer is ledgered: a reply this turn already ledgered is not sent twice (see the r35
#    tests below: the live send keyed to the turn, or to a later message it answered), and a ledgered
#    answer survives a failed send.
#    r35: the boot adoption of a crash marker is ledgered with the turn tag its token carries
#    (``mark_turn_active(turn=...)``). The adoption of THIS turn's marker (a parked profile's boot
#    adopts it before recovery runs) is not sent twice; another turn's, or an untagged marker's, does
#    not swallow this turn's answer, however recent.
def _crash_left(text, token):
    from gateway.delivery_ledger import compute_obligation_id, record_crash_left_reply
    from gateway.session_restart_recovery import turn_of_marker
    oid = compute_obligation_id('route-s', f'crash:{token}', text)
    record_crash_left_reply(obligation_id=oid, session_key='route-s', platform='slack', chat_id='C1', thread_id='T1',
                            content=text, since=0, turn=turn_of_marker(token))
    return oid


def _set_created(oid, created_at, *, untag=False):
    from gateway import delivery_ledger
    with delivery_ledger._store() as store:
        raw_holder(store)._write_sql('UPDATE delivery_obligations SET created_at = ?' + (', turn = NULL' if untag else '')
                                     + ' WHERE obligation_id = ?', (created_at, oid))


@pytest.mark.asyncio
async def test_another_turns_crash_left_row_does_not_suppress_the_saved_answer(harness):
    # Pass-4 review (Astra F2, Fable F1): a LATER turn F's marker adopted on a second boot, while this
    # turn's owed answer is still pending, is F's row, not this turn's.
    eid, fid = _now_id(-60), _now_id(-59)
    head, followers = await harness.lose_head(eid, followers=[('FOLLOWER', fid)], transcript=_answered(harness, eid))
    _crash_left('THE SAVED ANSWER', 'tok-f~' + _turn(followers[0]))
    _crash_left('THE SAVED ANSWER', 'tok-untagged')
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [text.endswith('THE SAVED ANSWER') for _a, text in harness.sent][:1] == [True]


@pytest.mark.asyncio
async def test_this_turns_crash_left_adoption_is_not_sent_again(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    _crash_left('THE SAVED ANSWER', 'tok-h~' + _turn(head))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


def test_the_marker_token_carries_the_turn_tag(tmp_path):
    from gateway.session_restart_recovery import turn_of_marker
    assert turn_of_marker('abc~admission:p:a1') == 'admission:p:a1'
    assert turn_of_marker('abc') == '' and turn_of_marker(None) == ''


@pytest.mark.asyncio
async def test_saved_answer_goes_through_the_ledger_redelivery(harness):
    from gateway.delivery_ledger import RECOVERED_MARKER
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
    assert _ledger_rows('content', 'obligation_id=?', (row['obligation_id'],)) == ['THE SAVED ANSWER']


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


# Astra r32 pass 8: an idle cache-expiry compaction can save a fresh assistant summary just before the
# person sends. Saved before the claim, it is not this turn running: the turn runs once.
@pytest.mark.asyncio
async def test_unanchored_turn_ignores_a_reply_saved_before_its_claim(harness):
    _head, _ = await harness.lose_head(_now_id(-60), before_claim=[
        {'role': 'assistant', 'content': '[CONTEXT COMPACTION] summary of the earlier conversation'}])
    await harness.drain()
    assert harness.executed == ['HEAD']


# The same bound never hides a step this turn saved after its claim.
@pytest.mark.asyncio
async def test_unanchored_turn_with_a_tool_row_after_its_claim_is_not_rerun(harness):
    head, _ = await harness.lose_head(_now_id(-60), before_claim=[
        {'role': 'assistant', 'content': '[CONTEXT COMPACTION] summary of the earlier conversation'}], transcript=[
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]},
        {'role': 'tool', 'content': 'ok', 'tool_call_id': 'd', 'tool_name': 'terminal'}])
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)


# 6. /pause is read in the owning profile's scope.
@pytest.mark.asyncio
async def test_estop_is_checked_in_the_owner_profile_scope(harness, monkeypatch):
    # What is asserted is the scope active when /pause is READ, not how often owner_scope is entered:
    # core enters it on its own paths too (startup local-session recovery, once per authority epoch).
    active, checked = [], []

    @contextlib.contextmanager
    def scope(authority):
        active.append(authority.profile_id)
        try:
            yield
        finally:
            active.pop()

    def is_engaged():
        checked.append(tuple(active))
        return False
    monkeypatch.setattr('gateway.session_authorities.owner_scope', scope)
    monkeypatch.setattr('agent.estop.is_engaged', is_engaged)
    _head, _ = await harness.lose_head(_now_id(-30))
    await harness.drain()
    assert checked == [('p',)] and harness.executed == ['HEAD']


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
            harness.admit(authority.epoch, 'late', _native('continue', _now_id()))
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
    from gateway import session_restart_recovery as recovery
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
            raise StoreContention('database is locked')
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
    assert _ledger_states() == ['attempting']  # owned by this process until the (stubbed) send marks it


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


def _ledger_rows(column, where='1', params=()):
    """One column of the delivery ledger's rows, read raw from the ledger's own store."""
    from gateway import delivery_ledger
    with delivery_ledger._store() as store, raw_holder(store)._read_ctx() as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'delivery_obligations'").fetchone() is None:
            return []  # the ledger creates its table on its first use of the store
        return [r[0] for r in conn.execute(f'SELECT {column} FROM delivery_obligations WHERE {where}', params)]


def _ledger_states():
    return _ledger_rows('state')


def _owed(harness):
    from gateway.session_restart_recovery import _list_owed
    return _list_owed(harness.db)


def _answered(harness, eid):
    return [{'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
            {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}]


# Round 3 M2: a transient commit failure, then a /discard in the retry window: nothing is sent.
@pytest.mark.asyncio
async def test_discard_after_a_transient_commit_failure_sends_nothing(harness, monkeypatch):
    from gateway import session_restart_recovery as recovery
    real = recovery._commit
    failed = []

    async def flaky(authority, row, action, **kw):
        if action == 'completed' and not failed:
            failed.append(1)
            harness.db.admissions.resolve_unknown(epoch=authority.epoch, admission_id=row['admission_id'],
                                                  generation=row['generation'])  # the person's /discard lands
            raise StoreContention('database is locked')
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
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    import gateway.delivery_ledger as ledger
    real = ledger.record_recovered_reply

    def locked(**kw):
        raise StoreContention('database is locked')
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
    raw_holder(harness.db)._execute_write(lambda c: c.execute(  # simulate the marker surviving the crash
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
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    real = recovery._clear_owed
    calls = []

    def flaky(db, admission_id):
        calls.append(1)
        if len(calls) == 1:
            raise StoreContention('database is locked')
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
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)

    def locked(db, admission_id):
        raise StoreContention('database is locked')
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
    from gateway import session_restart_recovery as recovery
    monkeypatch.setattr(recovery, 'LEDGER_BACKOFF_S', 0.0)
    monkeypatch.setattr('gateway.delivery_ledger.ledger_enabled', lambda *a, **k: False)
    real = recovery._clear_owed
    locked = [True]

    def clear(db, admission_id):
        if locked[0]:
            raise StoreContention('database is locked')
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


# r35 review (Astra BLOCK 1, pass 3 Astra/Fable): only a ledger row of THIS turn suppresses the recovered
# answer. Each live send is tagged with its turn (``_kc_ledger_turn``), whatever inbound id keyed it: a
# queued chain's last message, a busy redirect's message that is never an admission, or the head itself.
def _live_send(session_key, message_id, text, turn=None):
    """The row the live final send ledgers (base_delivery_ledger.record_final_obligation)."""
    from gateway.delivery_ledger import compute_obligation_id, record_obligation
    oid = compute_obligation_id(session_key, message_id, text)
    record_obligation(obligation_id=oid, session_key=session_key, platform='slack', chat_id='C1', thread_id='T1',
                      content=text, turn=turn)
    return oid


def _turn(admission_id):
    from gateway.session_restart_recovery import ledger_turn
    return ledger_turn('p', admission_id)


@pytest.mark.asyncio
async def test_an_earlier_turns_identical_reply_does_not_suppress_the_recovered_answer(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    _live_send('route-s', _now_id(-61), 'THE SAVED ANSWER', _turn('earlier-turn'))  # turn A's reply, same text
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [text.endswith('THE SAVED ANSWER') for _a, text in harness.sent] == [True]


@pytest.mark.asyncio
async def test_the_lost_turns_own_live_send_is_not_sent_again(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    _live_send('route-s', eid, 'THE SAVED ANSWER', _turn(head))  # the dying process sent it before the crash
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_a_send_keyed_to_a_message_that_is_no_admission_is_not_sent_again(harness):
    # A busy redirect's final is ledgered under the redirect's message id (MessageEvent.ledger_message_id),
    # which is never an admission of its own; a queued chain's under its last message's id.
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=_answered(harness, eid))
    _live_send('route-s', 'redirect-message-id', 'THE SAVED ANSWER', _turn(head))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_a_followers_own_reply_does_not_suppress_the_heads_answer(harness):
    # A follower admitted after the crash answered with the same text in its own turn.
    eid, fid = _now_id(-60), _now_id(-59)
    head, followers = await harness.lose_head(eid, followers=[('FOLLOWER', fid)], transcript=_answered(harness, eid))
    _live_send('route-s', fid, 'THE SAVED ANSWER', _turn(followers[0]))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [text.endswith('THE SAVED ANSWER') for _a, text in harness.sent][:1] == [True]


# Install boundary: a row r34 ledgered has no turn (NULL). It counts as this turn's only by the exact
# id one of this turn's own r34 sends had (the head's message id, or r34's own recovery row), for the
# text r34 presented. Another turn's NULL row with the same text, however close in time, never counts.
def _r34_row(message_ref, text):
    import time as _time
    oid = _live_send('route-s', message_ref, text)
    _set_created(oid, _time.time(), untag=True)
    return oid


async def _lose_two_line(harness, answer='Done\n[ SILENT ]'):
    _real_presentation(harness)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': answer, 'finish_reason': 'stop'}])
    return eid, head


@pytest.mark.asyncio
async def test_a_row_r34_ledgered_with_the_raw_text_is_not_sent_again(harness):
    eid, head = await _lose_two_line(harness)
    _r34_row(eid, 'Done\n[ SILENT ]')
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_r34s_own_recovery_row_is_not_sent_again(harness):
    _eid, head = await _lose_two_line(harness)
    _r34_row('restart-recovery:' + head, 'Done\n[ SILENT ]')
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_a_row_r34_ledgered_without_media_directives_is_not_sent_again(harness):
    # Pass-4 review (Astra F4): r34 stripped the media directive and kept the padded marker.
    eid, head = await _lose_two_line(harness, '[[as_document]]Done\n[ SILENT ]')
    from gateway.platforms.base import _strip_media_directives
    _r34_row(eid, _strip_media_directives('[[as_document]]Done\n[ SILENT ]').strip())
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_another_turns_r34_row_with_the_same_text_does_not_suppress_the_answer(harness):
    # Pass-4 review (Astra F1, Fable F2): r34 delivered turn A's "Done" just before this turn started.
    _eid, head = await _lose_two_line(harness)
    _r34_row(_now_id(-61), 'Done')
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [text.rsplit('\n', 1)[-1] for _a, text in harness.sent] == ['Done']


# r35 review (Astra BLOCK 2, Fable F1): the saved answer is presented with the turn's own
# reply_expected. An unaddressed turn (reply_expected False) that saved a bare silence marker owes
# nothing; before, the presentation saw no user row and owed the "unexpected silence" notice.
def _real_presentation(harness):
    from types import MethodType
    from gateway.run_startup import GatewayStartupMixin
    harness.runner._crash_left_reply = MethodType(GatewayStartupMixin._crash_left_reply, harness.runner)


@pytest.mark.asyncio
async def test_an_unaddressed_turns_saved_silence_owes_nothing(harness, monkeypatch):
    _real_presentation(harness)
    eid = _now_id(-60)
    payload = _native('HEAD', eid)
    payload['native_text_v1']['event']['reply_expected'] = False
    original = harness.admit

    def admit(epoch, request_id, p):
        return original(epoch, request_id, payload if request_id == 'head' else p)
    monkeypatch.setattr(harness, 'admit', admit)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': '[审核]', 'finish_reason': 'stop'}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert harness.sent == []


@pytest.mark.asyncio
async def test_the_saved_answer_is_presented_by_the_real_presentation(harness):
    # Proves the presentation above really ran: a trailing marker line is stripped, raw text would keep it.
    _real_presentation(harness)
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': 'Done\n[ SILENT ]', 'finish_reason': 'stop'}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert [text.rsplit('\n', 1)[-1] for _a, text in harness.sent] == ['Done']


@pytest.mark.asyncio
async def test_an_addressed_message_folded_into_an_unaddressed_turn_still_presents_the_notice(harness, monkeypatch):
    # The dead process steered a message into the unaddressed head's turn (saved as a user row, never an
    # admission; absorb_reply_expected may have made the turn addressed): a bare marker gets the notice.
    from agent.prompt_builder import steer_user_row
    _real_presentation(harness)
    eid = _now_id(-60)
    payload = _native('HEAD', eid)
    payload['native_text_v1']['event']['reply_expected'] = False
    original = harness.admit

    def admit(epoch, request_id, p):
        return original(epoch, request_id, payload if request_id == 'head' else p)
    monkeypatch.setattr(harness, 'admit', admit)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        steer_user_row('@bot are you there?'),
        {'role': 'assistant', 'content': '[审核]', 'finish_reason': 'stop'}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert any('silence marker' in text for _a, text in harness.sent)


def test_turn_reply_expected_is_unknown_once_input_was_folded():
    from gateway.session_restart_recovery import Verdict, turn_reply_expected

    def head(value):
        return {'payload': {'native_text_v1': {'event': {} if value is None else {'reply_expected': value}}}}
    plain, folded = Verdict('answered', answer='x'), Verdict('answered', answer='x', folded=True)
    assert turn_reply_expected(head(False), plain) is False
    assert turn_reply_expected(head(False), folded) is None
    assert turn_reply_expected(head(True), folded) is True
    assert turn_reply_expected(head(None), plain) is None
    # The fold's durable record, for this turn only.
    raised = {'turn': 'admission:p:h', 'reply_expected': True}
    assert turn_reply_expected(head(False), plain, raised, 'admission:p:h') is True
    assert turn_reply_expected(head(False), plain, raised, 'admission:p:other') is False
    assert turn_reply_expected(head(False), plain, {'turn': 'admission:p:h', 'reply_expected': None},
                               'admission:p:h') is None
    assert turn_reply_expected(head(False), plain, {'turn': 'admission:p:h', 'reply_expected': False},
                               'admission:p:h') is False


@pytest.mark.asyncio
async def test_an_accepted_steer_not_yet_saved_still_presents_the_notice(harness, monkeypatch):
    # Pass-4 review (Astra F5, Fable F4): the fold recorded the raised expectation; the steer row was
    # never saved. A bare marker still gets the live notice.
    from types import SimpleNamespace
    from gateway.session_restart_recovery import TURN_REPLY_EXPECTED_KEY, record_folded_reply_expected
    _real_presentation(harness)
    saved = {}
    store = harness.runner.session_store
    monkeypatch.setattr(store, 'set_session_metadata',
                        lambda key, name, value: saved.__setitem__((key, name), value) or True, raising=False)
    monkeypatch.setattr(store, 'get_session_metadata',
                        lambda key, name, default=None: saved.get((key, name), default), raising=False)
    eid = _now_id(-60)
    payload = _native('HEAD', eid)
    payload['native_text_v1']['event']['reply_expected'] = False
    original = harness.admit

    def admit(epoch, request_id, p):
        return original(epoch, request_id, payload if request_id == 'head' else p)
    monkeypatch.setattr(harness, 'admit', admit)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': '[审核]', 'finish_reason': 'stop'}])
    running = SimpleNamespace(_kc_ledger_turn=_turn(head), reply_expected=True)
    record_folded_reply_expected(store, payload['native_text_v1']['route'], running)
    assert saved[(payload['native_text_v1']['route'], TURN_REPLY_EXPECTED_KEY)]['turn'] == _turn(head)
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    assert any('silence marker' in text for _a, text in harness.sent)


def test_the_fold_records_the_raised_expectation_before_it_returns(monkeypatch):
    from types import SimpleNamespace
    from gateway.platforms.event import MessageEvent
    from gateway.run_busy import GatewayBusySessionMixin
    from gateway.session_restart_recovery import TURN_REPLY_EXPECTED_KEY
    monkeypatch.setattr('gateway.run_plugin_injection.person_input_joined_turn', lambda *_a: None)
    saved = {}
    opener = MessageEvent(text='hi', source=SimpleNamespace(), reply_expected=False)
    opener._kc_ledger_turn = 'admission:p:h'
    agent = object()
    turn = SimpleNamespace(agent=agent, event=opener, ctx=None, running_event=lambda: opener)
    runner = SimpleNamespace(
        _session_state=lambda _k: SimpleNamespace(turn=turn),
        session_store=SimpleNamespace(set_session_metadata=lambda k, n, v: saved.__setitem__((k, n), dict(v))))
    steer = MessageEvent(text='@bot?', source=SimpleNamespace(), reply_expected=True)
    assert GatewayBusySessionMixin._fold_into_running_turn(runner, agent, 'route-s', steer) is turn
    assert saved[('route-s', TURN_REPLY_EXPECTED_KEY)] == {'turn': 'admission:p:h', 'reply_expected': True}


@pytest.mark.asyncio
async def test_the_live_send_carries_its_admissions_turn_tag(tmp_path, monkeypatch):
    """Both events that can send a turn's reply carry its tag: the executed copy and the waiting caller's."""
    from functools import partial
    from types import SimpleNamespace
    import gateway.platforms.base_delivery_ledger as bdl
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore
    from gateway.session_restart_recovery import turn_of_marker
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_ingress import admit_message
    from gateway.session_restart_recovery import ledger_turn
    from tests.gateway.test_prompt_attachments import _authority
    executed = []

    queued_lane = []

    class _Adapter:
        async def send_final_ledgered(self, event, *_a, **_k):
            queued_lane.append(getattr(event, '_kc_ledger_turn', None))
            return SimpleNamespace(success=True), None
    monkeypatch.setattr('gateway.run_notifications.BasePlatformAdapter', _Adapter)

    async def answer(event):
        executed.append(getattr(event, '_kc_ledger_turn', None))
        # The real durable marker: its token carries the tag.
        await GatewayRunner._mark_durable_active_turn(runner, event, key)
        tokens.append(store._entries[key].active_turn_token)
        # The queued first-response lane builds a fresh event inside the turn: tagged from the drain.
        from gateway.run_notifications import GatewayNotificationsMixin
        await GatewayNotificationsMixin._send_queued_final_text(
            SimpleNamespace(), _Adapter(), source, 'first', None, None, 'route-s', 'inbound-1')
        return 'reply'
    tokens = []
    authority = await _authority(tmp_path, monkeypatch, answer)
    store = authority.runner.session_store
    key = store.get_or_create_session(authority.sessions['s'].source).session_key
    runner = SimpleNamespace(async_session_store=AsyncSessionStore(store))
    runner._clear_durable_active_turn = partial(GatewayRunner._clear_durable_active_turn, runner)
    authority.runner._clear_durable_active_turn = runner._clear_durable_active_turn
    source = authority.sessions['s'].source
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    receipts = []

    async def admit_native(event):
        receipt = await authority.submit(actor, Submission(event.text, SessionRef('p', 's'), {'text': event.text}, 'queue'))
        receipts.append(receipt)
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    event = MessageEvent(text='one', source=source)
    assert await admit_message(authority, event) == 'reply'
    tag = ledger_turn(authority.profile_id, receipts[0].admission_id)
    assert executed == [tag]
    assert event._kc_ledger_turn == tag
    assert [turn_of_marker(tok) for tok in tokens] == [tag]
    assert queued_lane == [tag]
    # record_final_obligation ledgers the tag.
    seen = {}
    monkeypatch.setattr('gateway.delivery_ledger.ledger_enabled', lambda: True)
    monkeypatch.setattr('gateway.delivery_ledger.record_obligation', lambda **kw: seen.update(kw))
    monkeypatch.setattr('gateway.delivery_ledger.mark_attempting', lambda _oid: None)
    adapter = type('A', (), {'typed_command_prefix': '!'})()
    assert await bdl.record_final_obligation(adapter, event, 'route-s', 'reply', adapter, False)
    assert seen['turn'] == tag
