"""The working card is removed after a delivered reply even when its stream was never closed.

Live 2026-11-10 (KC r23): 9 of 22 leftover working cards followed "Turn cleanup: progress did not
finish 5.0s after cancel". The progress task was stuck in ``chat.stopStream`` (the TCP connect to
Slack stalled), so bounded cleanup detached it and retired the card stream locally, with no network
call. The final reply went out and the cleanup ran ``chat.delete`` on a card whose stream was still
open. Slack refuses that with ``cant_delete_message`` (probed with Dev's bot the same day: delete
while streaming is refused, delete after ``chat.stopStream`` succeeds, a second delete answers
``message_not_found``). The refusal was swallowed at debug level and never retried, so the card
stayed, and Slack later sealed it as ``completed``.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class _SlackError(Exception):
    """slack_sdk's SlackApiError shape: the structured code is ``response['error']``."""

    def __init__(self, code: str):
        super().__init__(f"The request to the Slack API failed. (error: {code})")
        self.response = {"ok": False, "error": code}


def _slack_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    return adapter


def _slack(adapter, *, gone=(), stop_fails=False, delete_always=None):
    """A Slack that behaves as probed: an open stream cannot be deleted; a stopped one can."""
    calls, streaming, missing = [], set(), set(gone)

    async def chat_delete(*, channel, ts):
        calls.append(("chat.delete", ts))
        if delete_always:
            raise _SlackError(delete_always)
        if ts in missing:
            raise _SlackError("message_not_found")
        if ts in streaming:
            raise _SlackError("cant_delete_message")
        missing.add(ts)
        return {"ok": True}

    async def api_call(method, *, json):
        if method == "chat.startStream":
            streaming.add("CARD")
            return {"ts": "CARD"}
        if method == "chat.appendStream":
            return {"ok": True}
        calls.append((method, json.get("ts")))
        if method == "chat.stopStream":
            if stop_fails:
                raise _SlackError("internal_error")
            if json["ts"] not in streaming:
                raise _SlackError("message_not_in_streaming_state")
            streaming.discard(json["ts"])
        return {"ok": True}

    adapter._app.client.chat_delete.side_effect = chat_delete
    adapter._app.client.api_call.side_effect = api_call
    return calls


_THREAD = {"thread_id": "1.0"}


async def _open_card_then_detach_its_stop(adapter):
    """The live sequence: the turn opened its card, then bounded cleanup gave up on a stuck
    stopStream and retired the stream locally. Slack still holds the stream open."""
    tasks = [{"id": "1", "title": "x", "status": "in_progress"}]
    result = await adapter.send_native_task_card_progress("C1", tasks, metadata=_THREAD, owner="A")
    assert result.success and result.message_id == "CARD"
    adapter.abandon_native_task_card_progress("C1", metadata=_THREAD, owner="A")


@pytest.mark.asyncio
async def test_a_card_whose_stop_was_detached_is_closed_then_deleted():
    adapter = _slack_adapter()
    calls = _slack(adapter)
    await _open_card_then_detach_its_stop(adapter)

    assert await adapter.delete_message("C1", "CARD") is True

    assert calls == [("chat.delete", "CARD"), ("chat.stopStream", "CARD"), ("chat.delete", "CARD")]


@pytest.mark.asyncio
async def test_a_card_whose_stop_completed_is_deleted_in_one_call():
    adapter = _slack_adapter()
    calls = _slack(adapter)
    tasks = [{"id": "1", "title": "x", "status": "in_progress"}]
    await adapter.send_native_task_card_progress("C1", tasks, metadata=_THREAD, owner="A")
    await adapter.stop_native_task_card_progress("C1", metadata=_THREAD, owner="A")
    calls.clear()

    assert await adapter.delete_message("C1", "CARD") is True

    assert calls == [("chat.delete", "CARD")]


@pytest.mark.asyncio
async def test_a_message_that_is_already_gone_counts_as_deleted():
    """A sealed card is deleted when its replacement opens, and stays on the turn's cleanup list."""
    adapter = _slack_adapter()
    calls = _slack(adapter, gone={"SEALED"})

    assert await adapter.delete_message("C1", "SEALED") is True

    assert calls == [("chat.delete", "SEALED")]


@pytest.mark.asyncio
async def test_a_refused_card_delete_is_not_retried_forever():
    """One close, one more delete, then report it."""
    adapter = _slack_adapter()
    calls = _slack(adapter, delete_always="cant_delete_message")
    await _open_card_then_detach_its_stop(adapter)

    assert await adapter.delete_message("C1", "CARD") is False

    assert calls == [("chat.delete", "CARD"), ("chat.stopStream", "CARD"), ("chat.delete", "CARD")]


@pytest.mark.asyncio
async def test_a_failed_close_still_reports_the_failed_delete():
    adapter = _slack_adapter()
    calls = _slack(adapter, stop_fails=True)
    await _open_card_then_detach_its_stop(adapter)

    assert await adapter.delete_message("C1", "CARD") is False

    assert calls == [("chat.delete", "CARD"), ("chat.stopStream", "CARD"), ("chat.delete", "CARD")]


@pytest.mark.asyncio
async def test_a_message_that_is_not_a_working_card_is_never_closed():
    """``cant_delete_message`` also means "not yours". Only a card this adapter opened is closed."""
    adapter = _slack_adapter()
    calls = _slack(adapter, delete_always="cant_delete_message")

    assert await adapter.delete_message("C1", "M1") is False

    assert calls == [("chat.delete", "M1")]


@pytest.mark.asyncio
async def test_other_delete_errors_are_reported_without_a_close():
    adapter = _slack_adapter()
    calls = _slack(adapter, delete_always="ratelimited")
    await _open_card_then_detach_its_stop(adapter)

    assert await adapter.delete_message("C1", "CARD") is False

    assert calls == [("chat.delete", "CARD")]


def test_the_record_of_opened_cards_is_bounded():
    adapter = _slack_adapter()
    for n in range(adapter._NATIVE_TASK_CARD_TS_KEPT + 40):
        adapter._remember_native_task_card("C1", f"{n}.0")

    assert len(adapter._native_task_card_ts) == adapter._NATIVE_TASK_CARD_TS_KEPT
    assert ("C1", "0.0") not in adapter._native_task_card_ts
    assert ("C1", f"{adapter._NATIVE_TASK_CARD_TS_KEPT + 39}.0") in adapter._native_task_card_ts


@pytest.mark.asyncio
async def test_a_retired_stream_does_not_reopen_a_fresh_card():
    """Bounded cleanup abandoned the card while an update was in flight. Slack then reports the
    stream sealed. The turn is over: a fresh card opened now would come after its cleanup was
    scheduled, and nothing would ever remove or close it."""
    adapter = _slack_adapter()
    starts, appends, release = [], [], asyncio.Event()

    async def api_call(method, *, json):
        if method == "chat.startStream":
            starts.append(json)
            return {"ts": f"S{len(starts)}"}
        if method == "chat.appendStream":
            appends.append(json["ts"])
            if len(appends) > 1:  # the update that was in flight when cleanup gave up on the card
                await release.wait()
                raise _SlackError("message_not_in_streaming_state")
        return {"ok": True}

    adapter._app.client.api_call.side_effect = api_call
    meta = {"thread_id": "1.0"}
    tasks = [{"id": "1", "title": "x", "status": "in_progress"}]
    assert (await adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="A")).success

    update = asyncio.create_task(adapter.send_native_task_card_progress("C1", tasks, metadata=meta, owner="A"))
    for _ in range(200):
        if len(appends) > 1:
            break
        await asyncio.sleep(0.005)
    adapter.abandon_native_task_card_progress("C1", metadata=meta, owner="A")
    release.set()
    result = await asyncio.wait_for(update, 2)

    assert not result.success and not result.retryable
    assert len(starts) == 1, "a second card was opened for a turn that had already ended"


# ── gateway: the post-delivery cleanup ────────────────────────────────────────────────────


class _CleanupAdapter:
    """Records delete_message calls; ``outcomes`` gives each message's successive results."""

    def __init__(self, outcomes=None):
        self.deleted: list[str] = []
        self._outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.callback = None

    def register_post_delivery_callback(self, session_key, callback, *, generation=None):
        self.callback = callback

    def last_delivery_complete(self, session_key, *, generation=None):
        return True

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)
        queue = self._outcomes.get(message_id)
        outcome = queue.pop(0) if queue else True
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _schedule(adapter, ids):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    ctx = SimpleNamespace(_cleanup_progress=True, _cleanup_msg_ids=ids, session_key="k", run_generation=1,
                          source=SimpleNamespace(chat_id="C1"))
    runner._run_agent_schedule_bubble_cleanup({"final_response": "done"}, adapter, ctx)
    assert callable(adapter.callback)
    return ctx


async def _settle(predicate, timeout=3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.03)


@pytest.fixture
def _fast_retries(monkeypatch):
    from gateway import run_turn

    monkeypatch.setattr(run_turn, "_CLEANUP_DELETE_RETRY_PAUSES_S", (0.01, 0.02))


@pytest.mark.asyncio
async def test_cleanup_deletes_a_card_that_joined_the_list_after_it_was_scheduled(_fast_retries):
    """A detached progress task reopened the card while the turn was closing: the new card is
    tracked after the cleanup was registered, and must still be removed."""
    adapter = _CleanupAdapter()
    ctx = _schedule(adapter, ["CARD1"])
    ctx._cleanup_msg_ids.append("CARD2")

    adapter.callback()
    await _settle(lambda: len(adapter.deleted) >= 2)

    assert adapter.deleted == ["CARD1", "CARD2"]


@pytest.mark.asyncio
async def test_cleanup_retries_a_delete_that_failed(_fast_retries, caplog):
    adapter = _CleanupAdapter({"CARD1": [False, True], "B2": [RuntimeError("timeout"), True]})
    _schedule(adapter, ["CARD1", "B2", "OK3"])

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        adapter.callback()
        await _settle(lambda: len(adapter.deleted) >= 5)

    assert adapter.deleted == ["CARD1", "B2", "OK3", "CARD1", "B2"]  # only the failed ones again
    assert "Could not remove" not in caplog.text


@pytest.mark.asyncio
async def test_cleanup_reports_a_card_it_could_not_remove(_fast_retries, caplog):
    """Bounded, then loud: a card left in a thread is visible in the log, not only in Slack."""
    adapter = _CleanupAdapter({"CARD1": [False, False, False, False]})
    _schedule(adapter, ["CARD1", "OK2"])

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        adapter.callback()
        await _settle(lambda: "Could not remove" in caplog.text)

    assert adapter.deleted == ["CARD1", "OK2", "CARD1", "CARD1"]  # first pass, then two retries
    warnings = [r.getMessage() for r in caplog.records if "Could not remove" in r.getMessage()]
    assert len(warnings) == 1
    assert "CARD1" in warnings[0] and "C1" in warnings[0] and "OK2" not in warnings[0]
