"""A person's messaging turn lost across an owner restart continues without /discard or /reset.

Real ``SessionStore`` SQLite and two real owner epochs. The decision comes from the saved
transcript (anchored by the native event id), never from the prompt text.
"""
import json
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef
from hermes_state_runtime import admit_session_input, claim_session_input, list_session_admissions
from tests.waiting import async_wait_until


def _native(text, event_id, *, user='U1'):
    return {'text': text, 'native_text_v1': {
        'event': {'message_id': event_id}, 'route': 'route-s', 'timestamp': '2026-10-19T16:06:00',
        'source': {'platform': 'slack', 'chat_id': 'C1', 'thread_id': 'T1', 'user_id': user,
                   'chat_type': 'group', 'is_bot': False, 'message_id': event_id}}}


def _runner(tmp_path, monkeypatch, executed, sent):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    return store, runner


class Harness:
    """Owner 1 admits a person's head (and optional followers), claims it, writes what the lost
    turn saved, then dies. Owner 2 starts on the same store."""

    def __init__(self, tmp_path, monkeypatch):
        self.executed, self.sent = [], []
        self.store, self.runner = _runner(tmp_path, monkeypatch, self.executed, self.sent)
        self.db = self.store._db
        self.db.create_session('s', source='slack')
        self.ref = SessionRef('p', 's')
        sent = self.sent

        async def send(authority, row, text):
            sent.append((row['admission_id'], text))
            return True
        monkeypatch.setattr('gateway.session_restart_recovery.send_to_thread', send)
        executed = self.executed

        async def route_ok(runner, payload, session_id, available_source, adapter):
            return available_source, 'route-s'

        async def execute(owner, ref, row):  # the turn itself (a model call) is not under test
            executed.append(row['payload']['text'])
            return 'ACK_' + row['payload']['text']
        monkeypatch.setattr('gateway.session_envelope.check_native_route', route_ok)
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)

    async def lose_head(self, event_id, *, sent_at=None, followers=(), transcript=()):
        first = await initialize_session_authority(self.runner, profile_id='p', instance_id='owner-1')
        first.sessions['s'] = LiveSession(SessionSource(platform=Platform.SLACK, chat_id='C1'), 's')
        first._schedule = lambda ref: None
        head = admit_session_input(self.db, epoch=first.epoch, principal_id='messaging:person', session_id='s',
                                   request_id='head', payload=_native('HEAD', event_id))
        follower_ids = []
        for i, (text, fid) in enumerate(followers):
            row = admit_session_input(self.db, epoch=first.epoch, principal_id='messaging:person', session_id='s',
                                      request_id=f'f{i}', payload=_native(text, fid))
            follower_ids.append(row['admission_id'])
        claimed = claim_session_input(self.db, epoch=first.epoch, session_id='s')
        assert claimed['admission_id'] == head['admission_id']
        for message in transcript:
            self.db.append_message('s', **message)
        self.owner = await initialize_session_authority(self.runner, profile_id='p', instance_id='owner-2')
        self.owner.sessions['s'] = LiveSession(SessionSource(platform=Platform.SLACK, chat_id='C1'), 's')
        assert self.status(head['admission_id'])[0] == 'unknown'
        return head['admission_id'], follower_ids

    def status(self, admission_id):
        rows = {r['admission_id']: (r['status'], r['outcome'])
                for r in list_session_admissions(self.db, session_id='s', pending_only=False)}
        return rows[admission_id]

    async def drain(self):
        self.owner._schedule(self.ref)
        live = self.owner.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')

    def boundaries(self):
        return [m['content'] for m in self.db.get_messages('s') if m.get('display_kind') == 'failed_turn']

    def close(self):
        self.store.close_all_db_handles()


def _now_id(offset=0.0):
    return f'{time.time() + offset:.6f}'


def _call(cid, name):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    yield h
    h.close()


# ------------------------------------------------------------------ answered (Will's thread)
@pytest.mark.asyncio
async def test_saved_answer_settles_completed_and_is_sent_once_without_inference(harness):
    eid = _now_id(-60)
    head, (follower,) = await harness.lose_head(eid, followers=[('FOLLOWER', _now_id(-30))], transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('c1', 'terminal')]},
        {'role': 'tool', 'content': 'ok', 'tool_call_id': 'c1', 'tool_name': 'terminal'},
        {'role': 'assistant', 'content': 'THE SAVED ANSWER', 'finish_reason': 'stop'}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'completed')
    from gateway.delivery_ledger import RECOVERED_MARKER
    assert harness.sent == [(head, RECOVERED_MARKER + 'THE SAVED ANSWER')]
    assert harness.executed == ['FOLLOWER']  # the head never re-ran
    assert harness.status(follower) == ('terminal', 'completed')
    assert not harness.boundaries()
    from gateway.session_results import admission_result
    assert admission_result(harness.db, head)['result']['final_response'] == 'THE SAVED ANSWER'
    # A later boot finds nothing to recover and sends nothing again.
    await harness.drain()
    assert len(harness.sent) == 1


# ------------------------------------------------------------------ uncertain (Sam's thread)
@pytest.mark.asyncio
async def test_effect_capable_step_without_result_waits_then_next_message_continues(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('a', 'skill_view'), _call('b', 'skill_view')]},
        {'role': 'tool', 'content': 'x', 'tool_call_id': 'a', 'tool_name': 'skill_view'},
        {'role': 'tool', 'content': 'y', 'tool_call_id': 'b', 'tool_name': 'skill_view'},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('c', 'skill_view'), _call('d', 'terminal')]}])
    await harness.drain()
    # Nothing re-ran on its own; one short notice went to the thread, no /discard advice.
    assert harness.executed == [] and harness.status(head) == ('unknown', None)
    assert len(harness.sent) == 1 and 'terminal' in harness.sent[0][1]
    assert '/discard' not in harness.sent[0][1] and '/reset' not in harness.sent[0][1]
    await harness.drain()
    assert len(harness.sent) == 1  # deduplicated across drains (and restarts: it is durable)
    # The person says "continue" after the restart: the lost turn closes with a boundary that
    # names the unknown step, and their message runs.
    admit_session_input(harness.db, epoch=harness.owner.epoch, principal_id='messaging:person', session_id='s',
                        request_id='cont', payload=_native('continue', _now_id()))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'interrupted')
    assert harness.executed == ['continue']
    (boundary,) = harness.boundaries()
    assert 'terminal' in boundary and 'unknown' in boundary.lower() and 'restart' in boundary.lower()


@pytest.mark.asyncio
async def test_messages_queued_before_the_restart_continue_a_fresh_lost_turn(harness):
    """Nathan's "continue" was already queued behind Sam's lost turn: it runs once, in order."""
    eid = _now_id(-120)
    head, followers = await harness.lose_head(
        eid, followers=[('continue', _now_id(-60)), ('again', _now_id(-30))], transcript=[
            {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
            {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')]}])
    await harness.drain()
    assert harness.status(head) == ('terminal', 'interrupted')
    assert harness.executed == ['continue', 'again']
    assert [harness.status(f) for f in followers] == [('terminal', 'completed')] * 2
    assert harness.sent == []


@pytest.mark.asyncio
async def test_stale_queued_messages_never_run_on_their_own(harness, monkeypatch):
    """Days-old messages behind an old lost turn wait for the person's next message."""
    old = _now_id(-5 * 86400)
    head, (follower,) = await harness.lose_head(old, followers=[('old follow-up', _now_id(-5 * 86400 + 60))], transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': old, 'timestamp': float(old)},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('d', 'terminal')], 'timestamp': float(old) + 5}])
    await harness.drain()
    assert harness.executed == [] and harness.sent == []
    assert harness.status(head) == ('unknown', None) and harness.status(follower) == ('queued', None)
    admit_session_input(harness.db, epoch=harness.owner.epoch, principal_id='messaging:person', session_id='s',
                        request_id='new', payload=_native('new message', _now_id()))
    await harness.drain()
    assert harness.status(head) == ('terminal', 'interrupted')
    assert harness.executed == ['old follow-up', 'new message']  # FIFO order kept


# ------------------------------------------------------------------ unstarted
@pytest.mark.asyncio
async def test_unstarted_turn_runs_once_in_its_own_place(harness):
    head, (follower,) = await harness.lose_head(_now_id(-30), followers=[('FOLLOWER', _now_id(-20))])
    await harness.drain()
    assert harness.executed == ['HEAD', 'FOLLOWER']
    assert harness.status(follower) == ('terminal', 'completed')
    assert harness.status(head) == ('terminal', 'completed')
    assert not harness.boundaries()


@pytest.mark.asyncio
async def test_rerun_is_capped_when_the_turn_keeps_dying(harness, monkeypatch):
    from gateway import session_restart_recovery as recovery
    head, _ = await harness.lose_head(_now_id(-30))
    harness.db._execute_write(lambda c: c.execute(
        'INSERT INTO state_meta(key,value) VALUES(?,?)', (recovery.ATTEMPTS_PREFIX + head, '1')))
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)
    assert len(harness.sent) == 1  # one notice instead of a rerun loop


# ------------------------------------------------------------------ readonly
@pytest.mark.asyncio
async def test_fresh_readonly_turn_reruns_once_with_a_boundary(harness):
    eid = _now_id(-60)
    head, _ = await harness.lose_head(eid, transcript=[
        {'role': 'user', 'content': 'HEAD', 'platform_message_id': eid},
        {'role': 'assistant', 'content': None, 'tool_calls': [_call('a', 'read_file'), _call('b', 'web_search')]},
        {'role': 'tool', 'content': 'x', 'tool_call_id': 'a', 'tool_name': 'read_file'}])
    await harness.drain()
    assert harness.executed == ['HEAD']
    assert harness.status(head) == ('terminal', 'completed')
    (boundary,) = harness.boundaries()
    assert 'only read' in boundary


@pytest.mark.asyncio
async def test_estop_holds_automatic_reruns(harness, monkeypatch):
    monkeypatch.setattr('agent.estop.is_engaged', lambda: True)
    head, _ = await harness.lose_head(_now_id(-30))
    await harness.drain()
    assert harness.executed == [] and harness.status(head) == ('unknown', None)


# ------------------------------------------------------------------ scope
@pytest.mark.asyncio
async def test_non_messaging_person_turn_keeps_the_discard_pause(harness):
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    first = await initialize_session_authority(harness.runner, profile_id='p', instance_id='owner-1')
    first.sessions['s'] = LiveSession(SessionSource(platform=Platform.SLACK, chat_id='C1'), 's')
    first._schedule = lambda ref: None

    class Peer:
        def write(self, frame):
            return True
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    head = await first.submit(actor, Submission('h', harness.ref, {'text': 'HEAD'}, 'queue'))
    claim_session_input(harness.db, epoch=first.epoch, session_id='s')
    harness.owner = await initialize_session_authority(harness.runner, profile_id='p', instance_id='owner-2')
    harness.owner.sessions['s'] = LiveSession(SessionSource(platform=Platform.SLACK, chat_id='C1'), 's')
    await harness.drain()
    assert harness.status(head.admission_id) == ('unknown', None) and harness.sent == []


def test_classifier_reads_the_compression_chain_and_ignores_prompt_wording(tmp_path, monkeypatch):
    from gateway.session_restart_recovery import classify
    h = Harness(tmp_path, monkeypatch)
    try:
        eid = _now_id(-60)
        row = {'target_session_id': 's', 'payload': _native('please only read the sheet, do not write', eid)}
        h.db.append_message('s', 'user', 'x', platform_message_id=eid)
        h.db.append_message('s', 'assistant', None, tool_calls=[_call('z', 'patch')])
        assert classify(h.db, row).kind == 'uncertain'  # wording says read-only; the evidence does not
        assert classify(h.db, row).unsettled == ('patch',)
    finally:
        h.close()


def test_recover_unknown_session_input_is_fenced_and_idempotent(tmp_path, monkeypatch):
    import asyncio
    from hermes_state_runtime import RuntimeStoreError, recover_unknown_session_input
    h = Harness(tmp_path, monkeypatch)
    try:
        head, _ = asyncio.run(h.lose_head(_now_id(-30)))
        row = next(r for r in list_session_admissions(h.db, session_id='s') if r['admission_id'] == head)
        with pytest.raises(RuntimeStoreError):
            recover_unknown_session_input(h.db, epoch=h.owner.epoch - 1, admission_id=head,
                                          generation=row['generation'], action='requeue')
        first = recover_unknown_session_input(h.db, epoch=h.owner.epoch, admission_id=head,
                                              generation=row['generation'], action='requeue')
        again = recover_unknown_session_input(h.db, epoch=h.owner.epoch, admission_id=head,
                                              generation=row['generation'], action='requeue')
        assert first['status'] == again['status'] == 'queued'
        assert first['seq'] == row['seq']
        with h.db._read_ctx() as conn:
            assert conn.execute("SELECT value FROM state_meta WHERE key=?",
                                ('restart_recovery_attempts:' + head,)).fetchone()[0] == '1'
    finally:
        h.close()
