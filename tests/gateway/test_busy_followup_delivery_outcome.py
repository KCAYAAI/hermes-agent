"""KC: a busy follow-up's progress card survives an undelivered or partial reply.

Upstream's busy follow-up observer (``dispatch_shared_busy``) runs the turn's post-delivery
callbacks after its own send. KC's rule (``last_delivery_complete``) is that the temp-bubble cleanup
keeps the progress card unless the whole final reply was delivered. The observer has to hand that
outcome to the callbacks, or the cleanup sees "unknown lane" and deletes.
"""
import asyncio

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from tests.gateway.test_busy_followup_own_post_delivery import _AuthorityAdapter, _register_cleanup
from tests.gateway.test_shared_busy_receive_loop import _statuses, _wait_for


class _FlakyAdapter(_AuthorityAdapter):
    """Every send of the follow-up's reply (the first try and any plain-text retry) returns
    *second* (a SendResult) or raises it (an exception)."""

    def __init__(self, second):
        super().__init__()
        self.second = second

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content != 'reply to one':
            self.sent.append('reply to two')
            if isinstance(self.second, Exception):
                raise self.second
            return self.second
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


@pytest.mark.parametrize('second, kept', [
    (SendResult(success=True, message_id='ok'), False),
    (SendResult(success=False, error='refused', retryable=False), True),
    (SendResult(success=True, message_id='some', error_kind='partial'), True),
    (RuntimeError('send blew up'), True),
    (None, False),  # the follow-up has nothing to send: complete, as on the first message's lane
], ids=['delivered', 'failed', 'partial', 'raised', 'empty'])
@pytest.mark.asyncio
async def test_busy_follow_up_keeps_its_progress_card_unless_the_whole_reply_arrived(monkeypatch, second, kept):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = _FlakyAdapter(second)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    key = adapter._event_session_key(MessageEvent(text='', source=source))
    gate, first_task_done = asyncio.Event(), asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        if event.text == 'two':
            await asyncio.wait_for(first_task_done.wait(), 5)
        event._hermes_run_generation = run_generation
        runner._bind_adapter_run_generation(adapter, key, run_generation)
        if event.text == 'one':
            await gate.wait()
        _register_cleanup(adapter, run_generation, ['bubble-' + event.text], key=key, chat_id=source.chat_id)
        reply = '' if event.text == 'two' and second is None else 'reply to ' + event.text
        execution_result.get()['result'] = {'final_response': reply, 'messages': []}
        return reply
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    finish = adapter._finish_session_task

    def finish_and_signal(session_key, interrupt_event):
        finish(session_key, interrupt_event)
        first_task_done.set()
    adapter._finish_session_task = finish_and_signal

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: (second is None or 'reply to two' in adapter.sent) and not adapter._background_tasks
                    and key not in adapter._post_delivery_callbacks
                    and (kept or 'bubble-two' in adapter.deleted), 30)
    await asyncio.sleep(0.2)  # a kept card must still be there after the callbacks have had time to run
    assert 'bubble-one' in adapter.deleted
    assert ('bubble-two' not in adapter.deleted) is kept
