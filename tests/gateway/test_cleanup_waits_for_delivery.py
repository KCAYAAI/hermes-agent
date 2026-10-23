"""KC review 2026-10-22: the temporary progress card/bubbles are deleted only after the final
reply was confirmed delivered. Post-delivery callbacks fire from ``finally`` whatever the
outcome, so a failed send used to delete the breadcrumb too."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self, *, send_ok: bool):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self._send_ok = send_ok

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if self._send_ok:
            return SendResult(success=True, message_id="m-final")
        return SendResult(success=False, error="boom")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    await (stop_event.wait() if stop_event is not None else asyncio.Event().wait())


def _event():
    return MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="m1",
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"))


@pytest.mark.asyncio
@pytest.mark.parametrize("send_ok", [True, False])
async def test_delivery_outcome_is_visible_to_post_delivery_callbacks(send_ok):
    adapter = _Adapter(send_ok=send_ok)
    adapter._keep_typing = _hold_typing
    event = _event()
    key = build_session_key(event.source)
    seen = []

    async def handler(_event):
        adapter.register_post_delivery_callback(
            key, lambda: seen.append(adapter.last_delivery_complete(key)))
        return "the answer"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert seen == [send_ok]  # callbacks still fire (bg-review release), now knowing the outcome


@pytest.mark.asyncio
@pytest.mark.parametrize("send_ok", [True, False])
async def test_bubble_cleanup_fired_by_the_adapter_follows_the_delivery_outcome(send_ok):
    """The real cleanup callback, fired the way the adapter fires it (bounded ``wait_for`` after
    the final send): it still sees this turn's outcome and deletes only after a delivered reply."""
    from types import SimpleNamespace

    from gateway.run_turn_bubble_cleanup import register_bubble_cleanup

    deleted = []

    class _Deleting(_Adapter):
        async def delete_message(self, chat_id, message_id) -> bool:
            deleted.append(str(message_id))
            return True

    adapter = _Deleting(send_ok=send_ok)
    adapter._keep_typing = _hold_typing
    event = _event()
    key = build_session_key(event.source)

    async def handler(_event):
        register_bubble_cleanup({"final_response": "the answer"}, SimpleNamespace(
            _cleanup_adapter=adapter, _cleanup_msg_ids=["b1", "b2"], session_key=key,
            _cleanup_progress=True, source=event.source, run_generation=None))
        return "the answer"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert deleted == (["b1", "b2"] if send_ok else [])


def test_unknown_or_stale_outcome_is_none():
    from gateway.platforms.base import _DELIVERY_OUTCOME
    adapter = _Adapter(send_ok=True)
    assert adapter.last_delivery_complete("nope") is None
    token = _DELIVERY_OUTCOME.set(("k", 3, True))
    try:
        assert adapter.last_delivery_complete("k", generation=3) is True
        assert adapter.last_delivery_complete("k", generation=4) is None
        assert adapter.last_delivery_complete("other") is None
    finally:
        _DELIVERY_OUTCOME.reset(token)
    assert adapter.last_delivery_complete("k") is None


@pytest.mark.asyncio
async def test_overlapping_runs_of_one_session_keep_their_own_outcome():
    """Review: an older run's finalizer must not clear or overwrite a newer run's outcome. The two
    runs genuinely overlap: the old one resets while the new one is still waiting to read."""
    from gateway.platforms.base import _DELIVERY_OUTCOME
    adapter = _Adapter(send_ok=True)
    new_set, old_reset = asyncio.Event(), asyncio.Event()
    seen = {}

    async def old_run():
        token = _DELIVERY_OUTCOME.set(("k", None, True))
        await new_set.wait()
        seen["old"] = adapter.last_delivery_complete("k")
        _DELIVERY_OUTCOME.reset(token)
        old_reset.set()

    async def new_run():
        token = _DELIVERY_OUTCOME.set(("k", None, False))
        new_set.set()
        await old_reset.wait()
        seen["new"] = adapter.last_delivery_complete("k")
        _DELIVERY_OUTCOME.reset(token)

    await asyncio.gather(asyncio.create_task(old_run()), asyncio.create_task(new_run()))
    assert seen == {"old": True, "new": False}


class _TextFailsMediaOk(_Adapter):
    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kw):
        return SendResult(success=True, message_id="m-img")


@pytest.mark.asyncio
async def test_failed_text_with_delivered_attachment_is_not_complete(tmp_path):
    """Review: one successful attachment must not mark a reply whose text failed as delivered."""
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter = _TextFailsMediaOk(send_ok=False)
    adapter._keep_typing = _hold_typing
    event = _event()
    key = build_session_key(event.source)
    seen = []

    async def handler(_event):
        adapter.register_post_delivery_callback(
            key, lambda: seen.append(adapter.last_delivery_complete(key)))
        return f"the answer\nMEDIA:{img}"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert seen == [False]


class _TwoImagesOneFails(_Adapter):
    async def send_image(self, chat_id, image_url, caption=None, reply_to=None, metadata=None):
        if "bad" in image_url:
            return SendResult(success=False, error="upload failed")
        return SendResult(success=True, message_id="m-img")


@pytest.mark.asyncio
async def test_image_batch_with_a_failed_image_is_partial():
    adapter = _TwoImagesOneFails(send_ok=True)
    result = await adapter.send_multiple_images(
        "1", [("https://example.com/ok.png", ""), ("https://example.com/bad.png", "")])
    assert result.success is True and result.error_kind == "partial"
    clean = await adapter.send_multiple_images("1", [("https://example.com/ok.png", "")])
    assert clean.success is True and clean.error_kind is None


@pytest.mark.asyncio
async def test_text_delivered_but_an_image_failed_is_not_complete():
    """Review round 2: text ok + one image failed + one image ok must not count as delivered."""
    adapter = _TwoImagesOneFails(send_ok=True)
    adapter._keep_typing = _hold_typing
    event = _event()
    key = build_session_key(event.source)
    seen = []

    async def handler(_event):
        adapter.register_post_delivery_callback(
            key, lambda: seen.append(adapter.last_delivery_complete(key)))
        return "the answer\n![a](https://example.com/ok.png)\n![b](https://example.com/bad.png)"

    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, key)
    assert seen == [False]

