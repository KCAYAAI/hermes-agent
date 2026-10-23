"""KC: every delivery-ledger row of an admitted turn carries that turn's tag (restart recovery).

Producer-side proof for ``session_restart_recovery``'s same-turn check: the tag is written by the
real code paths, not seeded by a fixture. The boot crash-left pass reads it from the marker token;
a busy follow-up (``dispatch_shared_busy``) ledgers on a copy of its event; a ``pre_gateway_dispatch``
rewrite keeps the adapter's original event tagged.
"""
import asyncio

import pytest

from tests.fixtures.storage_holder import raw_holder


def _ledger_turns():
    from gateway import delivery_ledger
    with delivery_ledger._store() as store, raw_holder(store)._read_ctx() as conn:
        return {content: turn for content, turn in conn.execute('SELECT content, turn FROM delivery_obligations')}


@pytest.mark.asyncio
@pytest.mark.parametrize(('tag', 'expected'), [('admission:p:h1', 'admission:p:h1'), (None, '')])
async def test_the_boot_crash_left_pass_ledgers_the_marker_tokens_turn(tmp_path, tag, expected):
    from tests.gateway.test_active_turn_recovery import _close_store_handles, _db_runner, _make_source
    runner, store = _db_runner(tmp_path)
    entry = store.get_or_create_session(_make_source('tagged'))
    token = store.mark_turn_active(entry.session_key, turn=tag)
    assert token.endswith(f'~{tag}') if tag else '~' not in token
    store.append_to_transcript(entry.session_id, {'role': 'user', 'content': 'question'})
    store.append_to_transcript(entry.session_id, {'role': 'assistant', 'content': 'the stored answer'})
    assert await runner._recover_unclean_sessions() == (0, 1)
    assert _ledger_turns() == {'the stored answer': expected}
    _close_store_handles(store)


@pytest.mark.asyncio
async def test_a_busy_followups_reply_is_ledgered_with_its_own_admissions_tag(monkeypatch):
    # Same real path as test_busy_followup_marker_handoff: GatewayRunner, adapter, SessionAuthority,
    # drain and execute_admission over SQLite; the follow-up goes through dispatch_shared_busy.
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_restart_recovery import turn_of_marker
    from gateway.session_results import execution_result
    from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses, _wait_for

    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', 'fixture-user')
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='busy-chat', chat_type='dm', user_id='fixture-user')
    key = runner.session_store._generate_session_key(source)
    gate = asyncio.Event()

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        if event.text == 'one':
            await gate.wait()
        execution_result.get()['result'] = {'final_response': 'reply to ' + event.text, 'messages': []}
        return 'reply to ' + event.text
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    at_ledger = {}
    record = adapter._record_delivery_obligation

    async def ledger(event, session_key, text, *args, **kwargs):
        at_ledger[text] = (getattr(event, '_kc_ledger_turn', None),
                           turn_of_marker(runner.session_store._entries[key].active_turn_token))
        return await record(event, session_key, text, *args, **kwargs)
    monkeypatch.setattr(adapter, '_record_delivery_obligation', ledger)

    await adapter.handle_message(MessageEvent(text='one', source=source, message_id='m1'))
    await _wait_for(lambda: _statuses(authority).get('one') == 'started')
    await asyncio.wait_for(adapter.handle_message(MessageEvent(text='two', source=source, message_id='m2')), 5)
    gate.set()
    await _wait_for(lambda: 'reply to two' in adapter.sent and not adapter._background_tasks, 15)
    one, two = at_ledger['reply to one'], at_ledger['reply to two']
    assert one[0] and one[0].startswith('admission:default:') and one[0] == one[1]
    assert two[0] and two[0].startswith('admission:default:') and two[0] == two[1]
    assert one[0] != two[0]


def test_a_rewritten_event_and_the_event_it_was_made_from_are_both_tagged():
    from types import SimpleNamespace
    from gateway.session_ingress import tag_ledger_turn
    original = SimpleNamespace()
    rewritten = SimpleNamespace(_rewritten_from=original)
    assert tag_ledger_turn(SimpleNamespace(profile_id='p'), rewritten, 'a1') == 'admission:p:a1'
    assert rewritten._kc_ledger_turn == original._kc_ledger_turn == 'admission:p:a1'
