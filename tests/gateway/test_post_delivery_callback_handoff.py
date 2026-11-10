"""Every turn's post-delivery callbacks run once, after ITS reply, when a follow-up is waiting.

Live 2026-11-10 (KC r23): Slack working cards stayed in a thread after their final reply was
delivered whenever the person had written again during the turn. 22 leftover cards were read back
from live threads; 12 of them came from this hand-off.

A turn registers its callbacks (background-review release when the agent starts, progress-card
cleanup when it ends) in the adapter's one slot for the chat. The session drain runs the chat's
next turn as soon as this one settles, while the adapter task that waited for it is still sending
the reply:

* the waiting follow-up's turn registered under a newer generation and replaced the finished
  turn's entry before that entry had run (the finished turn's card was never removed);
* the finished turn's task read the generation off the shared interrupt Event only when it fired,
  after the follow-up had stamped its own, so its generation-checked pop matched nothing;
* a busy follow-up's reply is sent by ``dispatch_shared_busy``, which never ran the slot at all
  (the follow-up's own card was never removed).

Real GatewayRunner, adapter, SessionAuthority, drain and ``execute_admission`` over SQLite. Only the
agent turn is a stand-in: it binds its run generation and registers its callbacks where the real
turn does.
"""
import asyncio

import pytest

from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses, _wait_for


class _SlowFirstSend(Adapter):
    """The first reply takes a moment to send, as a platform call does."""

    def __init__(self, first_send_s: float):
        super().__init__()
        self._first_send_s = first_send_s

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content == 'reply to one' and self._first_send_s:
            await asyncio.sleep(self._first_send_s)
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


async def _owner(monkeypatch, adapter):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter.gateway_runner = runner  # as GatewayRunner._create_adapter binds every adapter
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    return runner, authority, source


def _turn(runner, adapter, fired, gates, *, second_turn_s=0.0, registers_nothing=(), entered=None):
    """Stand-in for ``_handle_message_with_agent``: the three things a real turn does to the
    adapter's post-delivery slot, at the points the real turn does them. ``entered`` collects each
    message whose turn is running (``execute_admission`` has read its waiter by then)."""
    from gateway.session_results import execution_result

    async def turn(event, source, quick_key, run_generation):
        runner._bind_adapter_run_generation(adapter, quick_key, run_generation)  # _hmwa_prepare_turn
        text = event.text
        if entered is not None:
            entered.append(text)
        if text in registers_nothing:  # answered with a notice before any agent ran
            execution_result.get()['result'] = {'final_response': 'reply to ' + text, 'messages': []}
            return 'reply to ' + text
        adapter.register_post_delivery_callback(  # TurnRunner: background-review release
            quick_key, lambda: fired.append((text, 'release')), generation=run_generation)
        gate = gates.get(text)
        assert isinstance(text, str)
        if gate is not None:
            await gate.wait()
        elif second_turn_s:
            await asyncio.sleep(second_turn_s)
        adapter.register_post_delivery_callback(  # _run_agent_schedule_bubble_cleanup
            quick_key,
            lambda: fired.append((text, 'cleanup', adapter.last_delivery_complete(quick_key, generation=run_generation))),
            generation=run_generation)
        execution_result.get()['result'] = {'final_response': 'reply to ' + text, 'messages': []}
        return 'reply to ' + text

    return turn


async def _let_callbacks_run(fired, expected: int) -> None:
    """Give the callbacks a moment, without failing here: the assertions below name what is missing."""
    deadline = asyncio.get_running_loop().time() + 1.5
    while len(fired) < expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # nothing fires twice


def _expected(*texts):
    return sorted(row for text in texts for row in ((text, 'release'), (text, 'cleanup', True)))


@pytest.mark.asyncio
@pytest.mark.parametrize('first_send_s, second_turn_s', [
    (0.15, 0.0),    # the follow-up starts AND ends while the first reply is still being sent
    (0.15, 0.4),    # the follow-up starts during the first send and ends after it
    (0.0, 0.05),    # the first reply is out before the follow-up registers anything
])
async def test_a_turn_and_its_busy_follow_up_each_run_their_own_callbacks(monkeypatch, first_send_s, second_turn_s):
    from gateway.platforms.event import MessageEvent

    adapter = _SlowFirstSend(first_send_s)
    runner, authority, source = await _owner(monkeypatch, adapter)
    key = runner.session_store._generate_session_key(source)
    fired, gate = [], asyncio.Event()
    monkeypatch.setattr(runner, '_handle_message_with_agent',
                        _turn(runner, adapter, fired, {'one': gate}, second_turn_s=second_turn_s))

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and 'reply to one' in adapter.sent
                    and not adapter._background_tasks and key not in adapter._session_tasks, 15)
    await _let_callbacks_run(fired, 4)

    assert sorted(adapter.sent) == ['reply to one', 'reply to two']
    # Each turn's callbacks ran exactly once, and each cleanup saw its OWN reply delivered.
    assert sorted(fired) == _expected('one', 'two')
    assert adapter._post_delivery_callbacks == {}  # nothing left to pin a finished turn


@pytest.mark.asyncio
async def test_three_messages_in_a_row_each_run_their_own_callbacks(monkeypatch):
    from gateway.platforms.event import MessageEvent

    adapter = _SlowFirstSend(0.1)
    runner, authority, source = await _owner(monkeypatch, adapter)
    key = runner.session_store._generate_session_key(source)
    fired, gate = [], asyncio.Event()
    monkeypatch.setattr(runner, '_handle_message_with_agent',
                        _turn(runner, adapter, fired, {'one': gate}, second_turn_s=0.05))

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    for n, text in enumerate(('two', 'three'), start=2):
        await asyncio.wait_for(adapter.handle_message(MessageEvent(text=text, source=source, message_id=f'm{n}')), 5)
        await _wait_for(lambda: text in _statuses(authority))
    gate.set()
    await _wait_for(lambda: len(adapter.sent) >= 3 and not adapter._background_tasks
                    and key not in adapter._session_tasks, 15)
    await _let_callbacks_run(fired, 6)

    assert sorted(adapter.sent) == ['reply to one', 'reply to three', 'reply to two']
    assert sorted(fired) == _expected('one', 'two', 'three')
    assert adapter._post_delivery_callbacks == {}


@pytest.mark.asyncio
async def test_a_follow_up_whose_reply_fails_keeps_its_card(monkeypatch):
    """The cleanup still runs after a failed send, and is told the reply was not delivered: the
    working card stays as the breadcrumb (``last_delivery_complete`` is False, not unknown)."""
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent

    class _SecondReplyFails(_SlowFirstSend):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            if 'reply to two' in content:  # the plain-text fallback fails too
                self.sent.append('reply to two')
                return SendResult(success=False, error='boom')
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    adapter = _SecondReplyFails(0.0)
    runner, authority, source = await _owner(monkeypatch, adapter)
    key = runner.session_store._generate_session_key(source)
    fired, gate = [], asyncio.Event()
    monkeypatch.setattr(runner, '_handle_message_with_agent', _turn(runner, adapter, fired, {'one': gate}))

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and not adapter._background_tasks
                    and key not in adapter._session_tasks, 15)
    await _let_callbacks_run(fired, 4)

    assert ('one', 'cleanup', True) in fired
    assert ('two', 'cleanup', False) in fired
    assert adapter._post_delivery_callbacks == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('late', ['one', 'two'])
async def test_a_waiter_that_registers_after_its_turn_started_runs_the_callbacks(monkeypatch, late):
    """``admit_message`` can resume only after the drain already claimed and started its row (the
    drain was running). That waiter still sends the reply, so it runs the turn's callbacks: for a
    first message through the adapter task, for a busy follow-up through its observer."""
    from gateway.platforms.event import MessageEvent

    adapter = _SlowFirstSend(0.0)
    runner, authority, source = await _owner(monkeypatch, adapter)
    key = runner.session_store._generate_session_key(source)
    fired, entered, gates = [], [], {'one': asyncio.Event(), 'two': asyncio.Event()}
    monkeypatch.setattr(runner, '_handle_message_with_agent', _turn(runner, adapter, fired, gates, entered=entered))
    admit, texts = authority.admit_native, ['one'] if late == 'one' else ['one', 'two']

    async def admit_then_let_the_drain_start_it(event):
        receipt = await admit(event)
        if event.text == late:
            if late == 'two':
                gates['one'].set()  # the first turn ends and the drain claims the follow-up
            # ``admit_message`` registers its waiter only after this returns, and the turn is already
            # running by then: the drain read "no waiter" before it started the turn.
            await _wait_for(lambda: late in entered)
        return receipt
    monkeypatch.setattr(authority, 'admit_native', admit_then_let_the_drain_start_it)

    for n, text in enumerate(texts, start=1):
        await asyncio.wait_for(adapter.handle_message(MessageEvent(text=text, source=source, message_id=f'm{n}')), 5)
        await _wait_for(lambda: _statuses(authority).get(text) == 'started')
    await _wait_for(lambda: bool(getattr(authority, 'marker_adopters', None)))  # the late waiter is in place
    for gate in gates.values():
        gate.set()
    await _wait_for(lambda: len(adapter.sent) == len(texts) and not adapter._background_tasks
                    and key not in adapter._session_tasks, 15)
    await _let_callbacks_run(fired, 2 * len(texts))

    assert sorted(fired) == _expected(*texts)
    assert adapter._post_delivery_callbacks == {}


@pytest.mark.asyncio
async def test_a_turn_that_registered_nothing_does_not_run_an_earlier_turns_callbacks(monkeypatch):
    """An earlier turn's entry can still be in the slot when the next turn runs (its own sender was
    cancelled, or has not run it yet). The next turn's sender must not run it: it would tell that
    turn's cleanup that THIS reply was delivered and remove a card kept as a breadcrumb."""
    from gateway.platforms.event import MessageEvent

    adapter = _SlowFirstSend(0.0)
    runner, _authority, source = await _owner(monkeypatch, adapter)
    key = runner.session_store._generate_session_key(source)
    fired = []
    earlier = (0, lambda: fired.append(('earlier turn', adapter.last_delivery_complete(key, generation=0))))
    adapter._post_delivery_callbacks[key] = earlier
    monkeypatch.setattr(runner, '_handle_message_with_agent',
                        _turn(runner, adapter, fired, {}, registers_nothing={'one'}))

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: 'reply to one' in adapter.sent and not adapter._background_tasks
                    and key not in adapter._session_tasks, 15)
    await _let_callbacks_run(fired, 1)

    assert fired == []
    assert adapter._post_delivery_callbacks.get(key) is earlier  # left for the turn that owns it


class _EveryReplyFails(Adapter):
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        self.sent.append(content)
        return SendResult(success=False, error='boom')


@pytest.mark.asyncio
@pytest.mark.parametrize('adapter_cls, delivered', [(Adapter, True), (_EveryReplyFails, False)])
async def test_a_turn_no_adapter_task_waits_for_tells_its_cleanup_the_delivery_outcome(
        monkeypatch, adapter_cls, delivered):
    """A turn admitted with no waiting adapter task (a recovered turn, a completion notice) is sent
    by the drain, which then runs its callbacks. The cleanup is told whether the whole reply went
    out, so a failed send keeps the working card instead of removing it as an unknown outcome."""
    from gateway.platforms.event import MessageEvent
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home

    adapter = adapter_cls()
    runner, authority, source = await _owner(monkeypatch, adapter)
    fired = []
    monkeypatch.setattr(runner, '_handle_message_with_agent', _turn(runner, adapter, fired, {}))

    event = MessageEvent(text='one', source=source, message_id='m1')
    with native_callback(runner, event, get_hermes_home()):
        await authority.admit_native(event)
    await _wait_for(lambda: _statuses(authority).get('one') == 'terminal' and adapter.sent, 15)
    await _let_callbacks_run(fired, 2)

    assert sorted(fired) == sorted([('one', 'release'), ('one', 'cleanup', delivered)])
    assert adapter._post_delivery_callbacks == {}
    assert not getattr(authority, 'reply_complete', None)  # nothing kept after the callbacks ran
