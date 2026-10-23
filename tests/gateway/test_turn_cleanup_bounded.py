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
        self.sends, self.stops, self.texts, self.edits = [], [], [], []
        self.hang, self.edit_ok = False, True

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
        self.texts.append(a or k)
        return SendResult(success=True, message_id="TXT")

    async def edit_message(self, **kw):
        self.edits.append(kw)
        return SendResult(success=self.edit_ok, message_id=kw.get("message_id"),
                          error=None if self.edit_ok else "message_not_found")


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


class _SlackShapedError(Exception):
    """slack_sdk.SlackApiError's shape (``exc.response["error"]``); the adapter reads only that."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"ok": False, "error": code}


@pytest.mark.asyncio
async def test_adapter_reports_a_started_card_whose_first_frame_failed():
    """chat.startStream answered, the first chat.appendStream failed: the gateway got no card id,
    but the card exists. The adapter reports its ts to the owning turn only."""
    adapter = _slack_adapter()

    async def api_call(method, *, json):
        if method == "chat.startStream":
            return {"ts": "S1"}
        raise RuntimeError("append failed")

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="A") == []  # no stream yet
    r = await adapter.send_native_task_card_progress("C1", [{"id": "1", "title": "x"}], metadata=meta, owner="A")
    assert not r.success and r.retryable and not r.message_id
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="A") == ["S1"]
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="B") == []  # another turn's card
    assert adapter.abandon_native_task_card_progress("C1", metadata=meta, owner="A") == ["S1"]
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="A") == []


@pytest.mark.asyncio
async def test_adapter_reports_an_expired_card_whose_replacement_was_cut_off():
    """The first frame finds its card S1 already sealed, so the adapter starts S2; the cancel lands
    on S2's first frame, before the adapter deleted S1 and before any publish returned success.
    Both cards exist and neither id reached the gateway: the adapter reports both."""
    adapter = _slack_adapter()
    in_flight, starts, deletes = asyncio.Event(), [], []

    async def api_call(method, *, json):
        if method == "chat.startStream":
            starts.append(1)
            return {"ts": f"S{len(starts)}"}
        if method == "chat.appendStream" and json["ts"] == "S1":
            raise _SlackShapedError("message_not_in_streaming_state")
        if method == "chat.appendStream":
            in_flight.set()
            await asyncio.Event().wait()
        if method == "chat.delete":
            deletes.append(json["ts"])
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    task = asyncio.create_task(adapter.send_native_task_card_progress(
        "C1", [{"id": "1", "title": "x"}], metadata=meta, owner="A"))
    await in_flight.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert deletes == []  # the adapter never reached its own delete of the sealed card
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="A") == ["S1", "S2"]


@pytest.mark.asyncio
async def test_stop_hands_over_a_card_another_sender_started_while_it_waited():
    """Two senders on one stream (two sessions in one thread). B's update finds the shared card
    reaped and holds the stream while its replacement starts. Owner A closes meanwhile: its early
    look at the stream sees only S1, and its stop waits for B. B's start answers S2 and its first
    frame fails. The stop then holds the stream, and must hand S2 to A before it seals and drops
    the stream, or neither turn ever learns S2."""
    adapter = _slack_adapter()
    start_held, release, starts, appends = asyncio.Event(), asyncio.Event(), [], []

    async def api_call(method, *, json):
        if method == "chat.startStream":
            starts.append(1)
            if len(starts) == 2:
                start_held.set()
                await release.wait()
            return {"ts": f"S{len(starts)}"}
        if method == "chat.appendStream":
            appends.append(json["ts"])
            if len(appends) == 2:
                raise _SlackShapedError("message_not_found")  # S1 was reaped
            if len(appends) == 3:
                raise RuntimeError("transient")  # S2's first frame
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    tasks = [{"id": "1", "title": "x"}]
    assert (await adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="A")).success
    b_update = asyncio.create_task(adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="B"))
    await start_held.wait()
    assert adapter.native_task_card_message_ids("C1", metadata=meta, owner="A") == ["S1"]  # A's early look
    handed = []
    a_stop = asyncio.create_task(adapter.stop_native_task_card_progress(
        "C1", metadata=meta, owner="A", on_card_ids=handed.append))
    await asyncio.sleep(0)
    assert handed == []  # the stop is waiting for the stream
    release.set()
    b_result = await asyncio.wait_for(b_update, 2)
    await asyncio.wait_for(a_stop, 2)
    assert not b_result.success and b_result.retryable
    assert handed == [["S1", "S2"]]
    assert adapter._native_task_card_streams == {}


async def _disposals_done(adapter):
    await asyncio.wait_for(asyncio.gather(*list(getattr(adapter, "_task_card_disposals", ()))), 2)


@pytest.mark.asyncio
async def test_taking_a_retired_card_down_survives_the_cancel_of_its_sender():
    """The take-down runs in its own task: the sender that made the late start can be cancelled
    (the follow-up finished and turn cleanup cancels it again) without skipping the delete."""
    adapter = _slack_adapter()
    start_entered, start_release = asyncio.Event(), asyncio.Event()
    stop_entered, stop_release, calls = asyncio.Event(), asyncio.Event(), []

    async def api_call(method, *, json):
        calls.append((method, json.get("ts")))
        if method == "chat.startStream":
            start_entered.set()
            await start_release.wait()
            return {"ts": "S1"}
        if method == "chat.stopStream":
            stop_entered.set()
            await stop_release.wait()
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}

    async def sender():
        await adapter.send_native_task_card_progress("C1", [{"id": "1", "title": "x"}], metadata=meta, owner="A")
        await asyncio.Event().wait()  # the sender task lives on, as a real one does

    task = asyncio.create_task(sender())
    await start_entered.wait()
    adapter.abandon_native_task_card_progress("C1", metadata=meta, owner="A")
    start_release.set()
    await asyncio.wait_for(stop_entered.wait(), 2)  # the take-down's stopStream is in flight
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    stop_release.set()
    await _disposals_done(adapter)
    assert calls == [("chat.startStream", None), ("chat.stopStream", "S1"), ("chat.delete", "S1")]


@pytest.mark.asyncio
async def test_start_that_returns_after_abandon_takes_its_card_down():
    """A chat.startStream that was in flight when its stream was abandoned returns late. Nothing
    tracks or stops that card any more, so the adapter ends and deletes it and sends no frame."""
    adapter = _slack_adapter()
    entered, release, calls = asyncio.Event(), asyncio.Event(), []

    async def api_call(method, *, json):
        calls.append((method, json.get("ts")))
        if method == "chat.startStream":
            entered.set()
            await release.wait()
            return {"ts": "S1"}
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    task = asyncio.create_task(adapter.send_native_task_card_progress(
        "C1", [{"id": "1", "title": "x"}], metadata=meta, owner="A"))
    await entered.wait()
    assert adapter.abandon_native_task_card_progress("C1", metadata=meta, owner="A") == []  # no card yet
    release.set()
    r = await asyncio.wait_for(task, 2)
    assert not r.success and not r.retryable
    await _disposals_done(adapter)
    assert calls == [("chat.startStream", None), ("chat.stopStream", "S1"), ("chat.delete", "S1")]


def test_abandon_keeps_the_retired_cards_id_for_cleanup():
    runner, ctx = _cleanup_runner(), _turn_ctx()
    ctx._cleanup_progress = True
    adapter = SimpleNamespace(abandon_native_task_card_progress=lambda *_a, **_k: ["S8", "S9"])
    runner._delivery_adapter_for = lambda _src: adapter
    runner._run_agent_abandon_task_card(ctx)
    runner._run_agent_abandon_task_card(ctx)
    assert ctx._progress_abandoned and ctx._cleanup_msg_ids == ["S8", "S9"]


@pytest.mark.asyncio
async def test_sender_takes_the_ids_its_stop_hands_over():
    """The stop of an adapter that names ``on_card_ids`` gets the sender's sink, and what it hands
    over lands in the turn's cleanup list. An adapter that does not name it is called as before."""
    adapter = _FakeCardAdapter()

    async def stop(chat_id, *, reply_to=None, metadata=None, owner=None, on_card_ids=None):
        on_card_ids(["S1", "S2"])

    adapter.stop_native_task_card_progress = stop
    runner = _card_runner(adapter)
    runner._ctx._cleanup_progress = True
    runner._ctx._progress_stop.set()
    await asyncio.wait_for(runner._send_native_task_card_progress(adapter), 2)
    assert runner._ctx._cleanup_msg_ids == ["S1", "S2"]

    plain = _FakeCardAdapter()
    runner = _card_runner(plain)
    runner._ctx._progress_stop.set()
    await asyncio.wait_for(runner._send_native_task_card_progress(plain), 2)
    assert plain.stops == ["turn-A"]


@pytest.mark.asyncio
async def test_sender_tracks_the_started_card_even_when_its_stop_is_cancelled():
    """The sender learns the card ids from the adapter before it awaits the stop, so a second
    cancel that cuts the stop short cannot leave a card out of the cleanup list."""
    adapter = _FakeCardAdapter()
    stop_entered = asyncio.Event()

    def card_ids(chat_id, *, reply_to=None, metadata=None, owner=None):
        return ["S6", "S7"] if owner == "turn-A" else []

    async def hung_stop(chat_id, *, reply_to=None, metadata=None, owner=None):
        stop_entered.set()
        await asyncio.Event().wait()

    adapter.native_task_card_message_ids = card_ids
    adapter.stop_native_task_card_progress = hung_stop
    runner = _card_runner(adapter)
    runner._ctx._cleanup_progress = True
    task = asyncio.create_task(runner._send_native_task_card_progress(adapter))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.wait_for(stop_entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert runner._ctx._cleanup_msg_ids == ["S6", "S7"]


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
async def test_followup_handoff_sends_no_card_update_after_the_cancel(monkeypatch, tmp_path):
    """The same handoff while an UPDATE of a card that is already shown is in flight. A flush after
    the cancel would send the update again; if Slack expired the card meanwhile, that update
    reopens a new card, which can land after the handoff stops waiting and is then never cleaned
    up. So nothing is sent to the card lane after the cancel."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class LiveCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.card_attempts = 0
            self.card_live = threading.Event()
            self.update_in_flight = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            self.card_attempts += 1
            if self.card_attempts == 1:
                self.card_live.set()
                return SendResult(success=True, message_id="CARD1")
            if self.card_attempts == 2:
                self.update_in_flight.set()
                await asyncio.Event().wait()  # the update, in flight until the handoff cancels it
            return SendResult(success=True, message_id="CARD2")  # an expired card reopened as a new one

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.card_live.is_set, timeout=5, what="card shown")
            self.tool_start_callback("call-2", "terminal", {"command": "ls"})
            wait_until(cls.adapter.update_in_flight.is_set, timeout=5, what="card update in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = LiveCardAdapter()
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
    assert adapter.card_attempts == 2  # the shown card and its cancelled update: nothing after the cancel
    assert {"chat_id": "C1", "message_id": "CARD1"} in adapter.deleted  # the shown card is still cleaned up


@pytest.mark.asyncio
async def test_followup_handoff_sends_no_replacement_text_after_the_cancel(monkeypatch, tmp_path):
    """The same handoff on the text lane: an EDIT of the existing fallback bubble is in flight. A
    flush after the cancel would edit again; if the bubble is gone that edit fails and a
    replacement bubble is sent, which can land untracked. So the text lane is not touched either."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class FallbackEditAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.text_attempts = 0
            self.edit_attempts = 0
            self.fallback_shown = threading.Event()
            self.edit_in_flight = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            return SendResult(success=False, error="card lane down")

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            result = await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
            if content not in ("first", "second"):
                self.text_attempts += 1
                self.fallback_shown.set()
            return result

        async def edit_message(self, chat_id, message_id, content, metadata=None, **_kw) -> SendResult:
            self.edit_attempts += 1
            if self.edit_attempts == 1:
                self.edit_in_flight.set()
                await asyncio.Event().wait()  # the edit, in flight until the handoff cancels it
            return SendResult(success=False, error="message_not_found")  # the bubble is gone

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.fallback_shown.is_set, timeout=5, what="text fallback shown")
            self.tool_start_callback("call-2", "terminal", {"command": "ls"})
            wait_until(cls.adapter.edit_in_flight.is_set, timeout=5, what="fallback edit in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = FallbackEditAdapter()
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
    assert adapter.edit_attempts == 1  # the cancelled edit only
    assert adapter.text_attempts == 1  # the bubble that was shown: no replacement after the cancel


@pytest.mark.asyncio
async def test_followup_handoff_cleans_up_a_card_slack_started_but_never_confirmed(monkeypatch, tmp_path):
    """Slack started the card (the adapter holds its ts, and that of an expired predecessor) but
    the first frame failed with a retryable error: no publish returned success, so the gateway
    tracked no id, and a resend is pending. At the handoff nothing more is published, and those
    cards must still be deleted with the turn's other bubbles before the follow-up starts."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class StartedCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.card_attempts = 0
            self.card_owner = None
            self.first_frame_failed = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, owner=None, **_kw):
            self.card_attempts += 1
            self.card_owner = owner
            self.first_frame_failed.set()
            return SendResult(success=False, error="append failed", retryable=True)

        def native_task_card_message_ids(self, chat_id, *, reply_to=None, metadata=None, owner=None):
            # An expired predecessor the adapter had not deleted yet, and the started card.
            return ["EXPIRED1", "STARTED1"] if self.card_owner and owner == self.card_owner else []

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

    class FollowupAgent(harness.QueuedFollowupAgent):
        deleted_at_followup = None

        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                cls.deleted_at_followup = [d["message_id"] for d in cls.adapter.deleted]
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.first_frame_failed.is_set, timeout=5, what="first card frame failed")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = StartedCardAdapter()
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

    assert result["final_response"] == "second"
    assert adapter.card_attempts == 1  # the failed first frame only: no resend after the handoff
    assert sorted(FollowupAgent.deleted_at_followup) == ["EXPIRED1", "STARTED1"]


@pytest.mark.asyncio
async def test_followup_handoff_sends_no_replacement_text_when_the_cancelled_edit_returns_a_failure(
    monkeypatch, tmp_path
):
    """The text lane when the transport turns the handoff's cancel into an ordinary failure of the
    in-flight edit. The publish that was under way continues after that await: it must stop there
    and not send a replacement bubble."""
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class FailsOnCancelEditAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.text_attempts = 0
            self.edit_attempts = 0
            self.fallback_shown = threading.Event()
            self.edit_in_flight = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            return SendResult(success=False, error="card lane down")

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            result = await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
            if content not in ("first", "second"):
                self.text_attempts += 1
                self.fallback_shown.set()
            return result

        async def edit_message(self, chat_id, message_id, content, metadata=None, **_kw) -> SendResult:
            self.edit_attempts += 1
            self.edit_in_flight.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
            return SendResult(success=False, error="timeout")

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.fallback_shown.is_set, timeout=5, what="text fallback shown")
            self.tool_start_callback("call-2", "terminal", {"command": "ls"})
            wait_until(cls.adapter.edit_in_flight.is_set, timeout=5, what="fallback edit in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = FailsOnCancelEditAdapter()
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
    assert adapter.edit_attempts == 1
    assert adapter.text_attempts == 1  # the bubble that was shown: no replacement after the failed edit


@pytest.mark.asyncio
async def test_normal_close_cleans_up_a_card_slack_started_but_never_confirmed(monkeypatch, tmp_path):
    """Without a follow-up the sender is joined before the turn's cleanup is registered, so the
    started-but-unconfirmed card is in the list there too and goes with a delivered reply."""
    from gateway.config import Platform
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class StartedCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.card_owner = None
            self.first_frame_failed = threading.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, owner=None, **_kw):
            self.card_owner = owner
            self.first_frame_failed.set()
            return SendResult(success=False, error="startStream ok, append failed", retryable=False)

        def native_task_card_message_ids(self, chat_id, *, reply_to=None, metadata=None, owner=None):
            return ["STARTED1"] if self.card_owner and owner == self.card_owner else []

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

    class OneTurnAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(type(self).adapter.first_frame_failed.is_set, timeout=5, what="first card frame failed")
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter = StartedCardAdapter()
    runner = harness._make_runner(adapter)
    gateway_run = harness._install_fakes(
        monkeypatch, OneTurnAgent, cleanup_on=True, cleanup_platform=Platform.SLACK)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:slack:group:C1"
    OneTurnAgent.adapter, OneTurnAgent.session_key = adapter, session_key

    result = await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.SLACK, chat_id="C1"),
        session_id="sess-1", session_key=session_key,
    )
    assert result["final_response"] == "done"
    await harness._fire_post_delivery_cb(adapter.pop_post_delivery_callback(session_key))
    assert "STARTED1" in {d["message_id"] for d in adapter.deleted}


@pytest.mark.asyncio
async def test_followup_handoff_ends_at_once_when_the_cancelled_card_start_returns_a_failure(monkeypatch, tmp_path):
    """The same handoff when the transport turns the cancel into an ordinary failure result: the
    sender must leave on the stop flag (not idle until the settle bound cancels it again) and open
    neither a second card nor a text bubble."""
    import time

    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    class FailsOnCancelCardAdapter(harness.CleanupCaptureAdapter):
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
            self.start_in_flight.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                asyncio.current_task().uncancel()
                return SendResult(success=False, error="startStream outcome unknown: timeout")

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

    adapter = FailsOnCancelCardAdapter()
    runner = harness._make_runner(adapter)
    gateway_run = harness._install_fakes(
        monkeypatch, FollowupAgent, cleanup_on=True, cleanup_platform=Platform.SLACK)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    session_key = "agent:main:slack:group:C1"
    FollowupAgent.adapter, FollowupAgent.session_key = adapter, session_key

    started = time.monotonic()
    result = await runner._run_agent(
        message="hello", context_prompt="", history=[],
        source=SessionSource(platform=Platform.SLACK, chat_id="C1"),
        session_id="sess-1", session_key=session_key,
    )
    elapsed = time.monotonic() - started
    for _ in range(20):
        await asyncio.sleep(0)

    assert result["final_response"] == "second"
    assert elapsed < 5  # well under the 10 s settle bound: the sender left on the stop flag
    assert adapter.card_attempts == 1  # the cancelled start only
    assert [m["content"] for m in adapter.sent] == ["first"]  # and no text bubble in its place


@pytest.mark.asyncio
async def test_followup_handoff_retires_a_stuck_card_sender_before_the_followup(monkeypatch, tmp_path):
    """A card update that swallows the handoff's cancel (the transport hangs on) outlives the settle
    bound. The handoff then retires that card stream, as turn cleanup does, so the follow-up does
    not queue behind it; and when the stuck call finally gives the cancel back, the old sender
    publishes nothing: its stream is stopped, so a publish would open a new card nobody cleans up."""
    from gateway import run_turn_bubble_cleanup as bubble_cleanup
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    from tests.gateway import test_run_cleanup_progress as harness
    from tests.waiting import wait_until

    monkeypatch.setattr(bubble_cleanup, "DELETE_TIMEOUT_SECONDS", 0.1)

    class StuckUpdateCardAdapter(harness.CleanupCaptureAdapter):
        def __init__(self):
            super().__init__(Platform.SLACK)
            self.card_attempts = 0
            self.abandoned = 0
            self.abandoned_when_followup_started = None
            self.card_live = threading.Event()
            self.update_stuck = threading.Event()
            self.update_returned = asyncio.Event()

        def native_task_cards_enabled(self):
            return True

        def native_task_card_destination_supported(self, *_a, **_k):
            return True

        async def send_native_task_card_progress(self, **_kw):
            self.card_attempts += 1
            if self.card_attempts == 1:
                self.card_live.set()
                return SendResult(success=True, message_id="CARD1")
            if self.card_attempts == 2:
                self.update_stuck.set()
                await _swallow_cancel_forever(for_s=0.7)  # survives both handoff cancels and cleanup's
                self.update_returned.set()
                raise asyncio.CancelledError()  # the transport gives the cancel back at last
            return SendResult(success=True, message_id=f"CARD{self.card_attempts}")

        async def stop_native_task_card_progress(self, *_a, **_k):
            return None

        def abandon_native_task_card_progress(self, *_a, **_k):
            self.abandoned += 1

    class FollowupAgent(harness.QueuedFollowupAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
            cls = type(self)
            if message == "follow-up":
                cls.adapter.abandoned_when_followup_started = cls.adapter.abandoned
                return {"final_response": "second", "messages": [], "api_calls": 1}
            self.tool_start_callback("call-1", "terminal", {"command": "pwd"})
            wait_until(cls.adapter.card_live.is_set, timeout=5, what="card shown")
            self.tool_start_callback("call-2", "terminal", {"command": "ls"})
            wait_until(cls.adapter.update_stuck.is_set, timeout=5, what="card update in flight")
            cls.adapter._pending_messages[cls.session_key] = MessageEvent(
                text="follow-up", message_type=MessageType.TEXT,
                source=SessionSource(platform=Platform.SLACK, chat_id="C1"))
            return {"final_response": "first", "messages": [], "api_calls": 1}

    adapter = StuckUpdateCardAdapter()
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
    assert result["final_response"] == "second"
    assert adapter.abandoned_when_followup_started == 1  # retired at the handoff, before the follow-up ran

    await asyncio.wait_for(adapter.update_returned.wait(), timeout=5)
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert adapter.card_attempts == 2  # the shown card and its stuck update: no third, orphan card
    assert [m["content"] for m in adapter.sent] == ["first"]  # and no text bubble
    assert "CARD1" in {d["message_id"] for d in adapter.deleted}  # the shown card was still cleaned up


@pytest.mark.asyncio
async def test_an_abandoned_sender_publishes_nothing_more():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    st.card_live = True
    runner._ctx._progress_abandoned = True
    await runner._task_card_publish(st)
    assert adapter.sends == [] and adapter.texts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shown", ["nothing", "card", "fallback"])
async def test_handoff_publishes_nothing_more_whatever_is_on_screen(shown):
    """After the handoff the sender creates and updates nothing: not a first card, not an update
    that could reopen an expired card, not an edit that could end in a replacement bubble."""
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    if shown == "card":
        st.card_live = True
    elif shown == "fallback":
        st.native_failed, st.fallback_msg_id = True, "TXT"
    runner._ctx._progress_stop.set()
    runner._ctx._progress_handoff = True
    await runner._task_card_publish(st)
    assert adapter.sends == [] and adapter.texts == [] and adapter.edits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [True, False])
async def test_closing_never_replaces_a_fallback_that_could_not_be_edited(closing):
    """Normal turn close (no handoff): the closing flush may edit the existing text fallback, but
    when that edit fails it must not send a NEW bubble in its place. Mid-turn it still does."""
    adapter = _FakeCardAdapter()
    adapter.edit_ok = False
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    st.native_failed, st.fallback_msg_id = True, "TXT"
    if closing:
        runner._ctx._progress_stop.set()
    await runner._task_card_publish(st)
    assert len(adapter.edits) == 1
    assert len(adapter.texts) == (0 if closing else 1)


@pytest.mark.asyncio
async def test_normal_close_may_still_open_the_card_with_its_final_state():
    adapter = _FakeCardAdapter()
    runner = _card_runner(adapter)
    st = runner._TaskCardState(adapter)
    st.apply_event({"type": "tool.started", "tool_call_id": "1", "tool_name": "terminal"})
    runner._ctx._progress_stop.set()  # turn cleanup, no handoff
    await runner._task_card_publish(st)
    assert len(adapter.sends) == 1
