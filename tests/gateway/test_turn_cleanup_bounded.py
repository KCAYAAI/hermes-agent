"""A finished turn's reply must never wait forever on a stuck cleanup step.

Incident (2026-10-20): a Slack turn logged ``Turn ended`` with its answer persisted, but never
logged ``response ready``; the reply was never sent and later messages queued behind the open
admission. Every post-agent cleanup await was unbounded, and an aiohttp transport can convert
the cleanup ``CancelledError`` into ``TimeoutError`` so a task never honours its cancel.

These tests drive the real ``_run_agent_cleanup_turn_tasks``, the real native task-card loop,
and the real Slack adapter with a transport that swallows cancellation.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import turn_cleanup
from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _fast_budget(monkeypatch):
    monkeypatch.setattr(turn_cleanup, "TURN_TASK_CLEANUP_TIMEOUT_S", 0.2)
    monkeypatch.setattr(turn_cleanup, "TURN_STALL_WATCHDOG_S", 0.3)


async def _swallow_cancel_forever(started: asyncio.Event | None = None, *, for_s: float = 2.0):
    """Ignore every cancel for ``for_s`` seconds (far past the 0.2 s test budget), then end, so
    test teardown never waits on it."""
    if started:
        started.set()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + for_s
    while loop.time() < deadline:
        try:
            await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()


def _cleanup_runner():
    from gateway.run_turn import GatewayTurnMixin

    runner = object.__new__(GatewayTurnMixin)
    runner._draining = False
    runner.released = []
    runner._release_running_agent_state = lambda key, run_generation=None: runner.released.append(key)
    runner._update_runtime_status = lambda *_a, **_k: None
    runner._delivery_adapter_for = lambda _src: None
    return runner


def _turn_ctx():
    from gateway.turn_context import TurnContext

    return TurnContext(
        source=SimpleNamespace(chat_id="C1"), session_key="agent:x:slack:C1:1", run_generation=1,
    )


async def _done():
    return None


# ── cleanup joins are bounded ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_finishes_when_progress_task_swallows_cancel():
    runner, ctx = _cleanup_runner(), _turn_ctx()
    started = asyncio.Event()
    progress = asyncio.create_task(_swallow_cancel_forever(started))
    await started.wait()
    others = [asyncio.create_task(_done()) for _ in range(4)]

    await asyncio.wait_for(
        runner._run_agent_cleanup_turn_tasks(
            ctx, progress_task=progress, log_task=None, interrupt_monitor=others[0],
            _notify_task=others[1], tracking_task=others[2], stream_task=None,
        ),
        timeout=1.5,
    )

    assert runner.released == [ctx.session_key]
    assert ctx._progress_stop.is_set()
    assert progress in turn_cleanup.detached_turn_tasks()


@pytest.mark.asyncio
async def test_cleanup_finishes_when_stream_consumer_swallows_cancel():
    runner, ctx = _cleanup_runner(), _turn_ctx()
    ctx.stream_consumer_holder[0] = object()  # a consumer exists: the 5 s flush path runs
    stream = asyncio.create_task(_swallow_cancel_forever())
    tracking = asyncio.create_task(_done())
    import gateway.run_turn as rt

    rt._STREAM_FLUSH_TIMEOUT_S = 0.1
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        await runner._run_agent_cleanup_turn_tasks(
            ctx, progress_task=None, log_task=None, interrupt_monitor=None,
            _notify_task=None, tracking_task=tracking, stream_task=stream,
        )
    finally:
        rt._STREAM_FLUSH_TIMEOUT_S = 5.0
    # Flush budget (0.1) + bounded join (0.2), far below the 2 s the consumer ignores cancels.
    assert loop.time() - t0 < 1.0
    assert runner.released == [ctx.session_key]


@pytest.mark.asyncio
async def test_cleanup_still_swallows_real_task_errors():
    runner, ctx = _cleanup_runner(), _turn_ctx()

    async def boom():
        raise RuntimeError("transport died")

    failed = asyncio.create_task(boom())
    await asyncio.sleep(0)
    await runner._run_agent_cleanup_turn_tasks(
        ctx, progress_task=failed, log_task=None, interrupt_monitor=None,
        _notify_task=None, tracking_task=asyncio.create_task(_done()), stream_task=None,
    )
    assert runner.released == [ctx.session_key]


@pytest.mark.asyncio
async def test_parent_cancellation_is_forwarded_to_the_child():
    child_cancelled = asyncio.Event()

    async def child():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            child_cancelled.set()
            raise

    task = asyncio.create_task(child())
    await asyncio.sleep(0)
    joiner = asyncio.create_task(turn_cleanup.join_cancelled_task(task, label="t"))
    await asyncio.sleep(0.01)
    joiner.cancel()
    assert await asyncio.wait_for(joiner, 2) is True
    assert child_cancelled.is_set() and task.cancelled()


@pytest.mark.asyncio
async def test_detached_task_is_supervised_until_it_ends():
    release = asyncio.Event()

    async def stubborn():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()
            raise ValueError("late failure")

    task = asyncio.create_task(stubborn())
    await asyncio.sleep(0)
    task.cancel()
    assert await turn_cleanup.join_cancelled_task(task, label="t", timeout=0.05) is False
    assert task in turn_cleanup.detached_turn_tasks()
    release.set()
    await asyncio.sleep(0.05)
    assert task not in turn_cleanup.detached_turn_tasks()


@pytest.mark.asyncio
async def test_run_bounded_detaches_a_hung_typing_clear():
    assert await turn_cleanup.run_bounded(_swallow_cancel_forever(), label="typing", timeout=0.05) is False


@pytest.mark.asyncio
async def test_stop_typing_for_turn_is_bounded():
    from gateway.run_turn import GatewayTurnMixin

    class _Adapter:
        async def _stop_typing_with_metadata(self, chat_id, metadata):
            await _swallow_cancel_forever()

    runner = object.__new__(GatewayTurnMixin)
    runner._delivery_adapter_for = lambda _src: _Adapter()
    runner._event_thread_metadata = lambda _e, _s: {}
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await runner._hmwa_stop_typing_for_turn(SimpleNamespace(), SimpleNamespace(chat_id="C1"))
    assert loop.time() - t0 < 1.0  # bounded at 0.2 s; the hung clear runs for 2 s


@pytest.mark.asyncio
async def test_stall_watchdog_logs_the_await_chain(caplog):
    caplog.set_level("WARNING", logger="gateway.run")
    with turn_cleanup.stall_watchdog("probe", "sess", seconds=0.05):
        await asyncio.sleep(0.15)
    assert any("Turn stalled: probe" in r.getMessage() for r in caplog.records)


# ── native task-card loop ────────────────────────────────────────────────────────────────────


def _card_runner(adapter, *, current=lambda: True):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1", platform="slack"), progress_queue=queue.Queue(),
        _progress_reply_to=None, _progress_metadata={"thread_id": "1.0"}, _cleanup_progress=False,
        _cleanup_msg_ids=[], _run_still_current=current, tool_progress_enabled=False,
        _progress_stop=threading.Event(), _progress_owner="turn-A",
    )
    runner._agent_interrupted = lambda: False
    return runner


class _FakeCardAdapter:
    def __init__(self):
        self.sends, self.stops, self.texts = [], [], []
        self.hang = False

    def native_task_card_destination_supported(self, *_a, **_k):
        return True

    async def send_native_task_card_progress(self, **kw):
        self.sends.append(kw)
        if self.hang:
            await _swallow_cancel_forever()
        return SendResult(success=True, message_id="CARD")

    async def stop_native_task_card_progress(self, chat_id, *, reply_to=None, metadata=None, owner=None):
        self.stops.append(owner)

    async def send(self, *a, **k):
        self.texts.append(a)
        return SendResult(success=True, message_id="TXT")


@pytest.mark.asyncio
async def test_card_loop_exits_on_stop_flag_even_if_cancel_is_swallowed():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    task = asyncio.create_task(runner._send_native_task_card_progress(adapter))
    await asyncio.sleep(0.05)
    runner._ctx._progress_stop.set()
    await asyncio.wait_for(task, 2)
    assert adapter.stops == ["turn-A"]


@pytest.mark.asyncio
async def test_cancelled_publish_keeps_final_state_for_one_terminal_flush():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    gate = asyncio.Event()
    calls = []

    async def slow_send(**kw):
        calls.append([t["status"] for t in kw["tasks"]])
        if len(calls) == 2:
            await gate.wait()  # the completion publish is cut off by the cancel
        return SendResult(success=True, message_id="CARD")

    adapter.send_native_task_card_progress = slow_send
    q = runner._ctx.progress_queue
    task = asyncio.create_task(runner._send_native_task_card_progress(adapter))
    q.put({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    await asyncio.sleep(0.25)
    q.put({"type": "tool.completed", "tool_call_id": "1", "tool_name": "terminal"})
    await asyncio.sleep(0.25)
    task.cancel()
    await asyncio.wait_for(task, 2)
    assert calls[-1] == ["complete"], calls  # the final state was re-published once
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_no_new_text_fallback_once_closing():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    runner._ctx._progress_stop.set()
    st = runner._TaskCardState(adapter)
    st.native_failed = True
    sent = []

    async def _send_progress_text(_st, text):
        sent.append(text)
        return SendResult(success=True, message_id="TXT")

    runner._send_progress_text = _send_progress_text
    await runner._task_card_send_or_edit_fallback(st)
    await runner._task_card_send_or_edit_fallback(st)
    assert sent == []


@pytest.mark.asyncio
async def test_existing_fallback_still_gets_its_final_edit_when_closing():
    adapter = _FakeCardAdapter()
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    runner = _card_runner(adapter)
    runner._ctx._progress_stop.set()
    st = runner._TaskCardState(adapter)
    st.native_failed, st.fallback_msg_id = True, "TXT"
    await runner._task_card_send_or_edit_fallback(st)
    adapter.edit_message.assert_awaited_once()


def test_missing_stop_flag_and_owner_are_compatible():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    del runner._ctx._progress_stop
    del runner._ctx._progress_owner
    assert runner._progress_stopped() is False
    assert runner._task_card_owner_kwargs(adapter.stop_native_task_card_progress) == {}


def test_owner_kwarg_only_for_adapters_that_accept_it():
    runner = _card_runner(_FakeCardAdapter())

    async def legacy_stop(chat_id, *, reply_to=None, metadata=None):
        return None

    assert runner._task_card_owner_kwargs(legacy_stop) == {}

    async def forwarding_wrapper(*args, **kwargs):  # forwards to a parent without ``owner``
        return None

    assert runner._task_card_owner_kwargs(forwarding_wrapper) == {}
    assert runner._task_card_owner_kwargs(_FakeCardAdapter().stop_native_task_card_progress) == {"owner": "turn-A"}


# ── Slack adapter: cancellation and ownership ────────────────────────────────────────────────


def _slack_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    a = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_adapter_reraises_cancel_converted_to_timeout():
    adapter = _slack_adapter()
    entered = asyncio.Event()

    async def api_call(method, *, json):
        if method == "chat.startStream":
            return {"ts": "S1"}
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError as exc:
            # A transport that turns the caller's cancel into an ordinary error while the task is
            # still being cancelled (cancelling() > 0).
            raise TimeoutError() from exc

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    task = asyncio.create_task(adapter.send_native_task_card_progress(
        "C1", [{"id": "1", "title": "x", "status": "in_progress"}], metadata=meta))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_old_turn_stop_cannot_close_newer_turns_card():
    adapter = _slack_adapter()
    adapter._app.client.api_call.side_effect = _api
    meta = {"thread_id": "1.0"}
    tasks = [{"id": "1", "title": "x", "status": "in_progress"}]
    assert (await adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="new")).success

    await adapter.stop_native_task_card_progress("C1", metadata=meta, owner="old")
    assert len(adapter._native_task_card_streams) == 1  # the newer card survives

    await adapter.stop_native_task_card_progress("C1", metadata=meta, owner="new")
    assert adapter._native_task_card_streams == {}


@pytest.mark.asyncio
async def test_unowned_stop_keeps_legacy_behavior():
    adapter = _slack_adapter()
    adapter._app.client.api_call.side_effect = _api
    meta = {"thread_id": "1.0"}
    await adapter.send_native_task_card_progress("C1", [{"id": "1", "title": "x"}], metadata=meta, owner="t")
    await adapter.stop_native_task_card_progress("C1", metadata=meta)
    assert adapter._native_task_card_streams == {}


@pytest.mark.asyncio
async def test_abandon_retires_stuck_card_without_network_and_next_turn_opens_fresh():
    adapter = _slack_adapter()
    starts = []

    async def api_call(method, *, json):
        if method == "chat.startStream":
            starts.append(1)
            return {"ts": f"S{len(starts)}"}
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    tasks = [{"id": "1", "title": "x"}]
    await adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="A")
    old = next(iter(adapter._native_task_card_streams.values()))
    await old.lock.acquire()  # A's updater is stuck holding the stream lock
    calls_before = adapter._app.client.api_call.await_count
    adapter.abandon_native_task_card_progress("C1", metadata=meta, owner="A")
    assert adapter._app.client.api_call.await_count == calls_before
    assert old.stopped and adapter._native_task_card_streams == {}
    r = await asyncio.wait_for(adapter.send_native_task_card_progress(
        "C1", tasks, metadata=meta, owner="B"), 2)
    assert r.success and len(starts) == 2


async def _api(method, *, json):
    return {"ts": "S1"} if method == "chat.startStream" else {"ok": True}


# ── queued follow-up handoff ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_followup_handoff_closes_progress_before_it_cancels_the_senders(monkeypatch, tmp_path):
    """The queued-follow-up handoff cancels the progress sender before turn cleanup runs. It must
    close the turn first, as cleanup does: the card loop's cancel-time flush would otherwise open
    a NEW text fallback after the cancel, a bubble that no turn's cleanup list is sure to own."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class FailedCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.text_attempts = 0
            self.send_started = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            return SendResult(success=False, error="card lane down")

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            if content not in ("first", "second"):
                self.text_attempts += 1
                if self.text_attempts == 1:
                    self.send_started.set()
                    await asyncio.Event().wait()  # in flight until the handoff cancels it
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                return {"final_response": "second", "messages": [], "api_calls": 1}
            # Native cards take the ID-bearing start callback the runner attaches to the agent.
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.send_started.is_set, timeout=5, what="text fallback send in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = FailedCardAdapter()
    runner = harness._make_runner(adapter)
    gateway_run = harness._install_fakes(
        monkeypatch, FollowupAgent, cleanup_on=True, cleanup_platform=Platform.SLACK)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:slack:group:C1"
    FollowupAgent.adapter, FollowupAgent.session_key = adapter, session_key

    result = await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.SLACK, chat_id="C1"),
        session_id="sess-1", session_key=session_key,
    )
    for _ in range(20):
        await asyncio.sleep(0)

    assert result["final_response"] == "second"
    assert adapter.send_started.is_set()
    assert adapter.text_attempts == 1  # the cancelled send only: no new bubble after the handoff


@pytest.mark.asyncio
async def test_followup_handoff_opens_no_new_native_card(monkeypatch, tmp_path):
    """The same handoff while the FIRST native card publish is still in flight (no card shown yet):
    the cancel-time flush must not start a new card. A card opened after the cancel can finish
    after the handoff stops waiting, and then no cleanup list ever holds its id."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class SlowStartCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.card_attempts = 0
            self.start_in_flight = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            self.card_attempts += 1
            if self.card_attempts == 1:
                self.start_in_flight.set()
                await asyncio.Event().wait()  # the stream start, in flight until the handoff cancels it
            return SendResult(success=True, message_id=f"CARD{self.card_attempts}")

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.start_in_flight.is_set, timeout=5, what="first card publish in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = SlowStartCardAdapter()
    runner = harness._make_runner(adapter)
    gateway_run = harness._install_fakes(
        monkeypatch, FollowupAgent, cleanup_on=True, cleanup_platform=Platform.SLACK)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:slack:group:C1"
    FollowupAgent.adapter, FollowupAgent.session_key = adapter, session_key

    result = await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.SLACK, chat_id="C1"),
        session_id="sess-1", session_key=session_key,
    )
    for _ in range(20):
        await asyncio.sleep(0)

    assert result["final_response"] == "second"
    assert adapter.card_attempts == 1  # the cancelled start only
    assert [m["content"] for m in adapter.sent] == ["first"]  # and no text bubble in its place


@pytest.mark.asyncio
async def test_handoff_opens_no_new_card_but_a_live_card_gets_its_final_state():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    runner._ctx._progress_stop.set()
    runner._ctx._progress_handoff = True
    await runner._task_card_publish(st)
    assert adapter.sends == [] and adapter.texts == []  # nothing was shown: nothing is opened
    st.card_live = True
    await runner._task_card_publish(st)
    assert len(adapter.sends) == 1  # a card that is already up still gets its final state


@pytest.mark.asyncio
async def test_normal_close_may_still_open_the_card_with_its_final_state():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    runner._ctx._progress_stop.set()  # turn cleanup, no handoff
    await runner._task_card_publish(st)
    assert len(adapter.sends) == 1
