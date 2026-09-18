"""Guarded root-document recovery through the real Slack inbound adapter."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def scenario():
    client = AsyncMock()
    client.users_info = AsyncMock(return_value={
        "ok": True, "user": {"id": "U_ROOT", "is_bot": False,
                              "profile": {"display_name": "Synthetic author"}},
    })
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="synthetic"))
    adapter._app = MagicMock()
    adapter._app.client = client
    adapter._team_clients = {"T_TEST": client}
    adapter._team_bot_user_ids = {"T_TEST": "U_BOT"}
    adapter._bot_user_id = "U_BOT"
    adapter._running = True
    adapter._user_name_cache = {("T_TEST", "U_ROOT"): "Synthetic author",
                                ("T_TEST", "U_HUMAN"): "Synthetic requester"}
    adapter._user_is_bot_cache = {("T_TEST", "U_HUMAN"): False}
    adapter._has_active_session_for_thread = MagicMock(return_value=False)
    adapter._register_mentioned_thread = MagicMock()
    adapter.handle_message = AsyncMock()
    adapter.set_authorization_check(lambda *args, **kwargs: True)
    pdf = b"%PDF-1.7\nsynthetic-root-document\n"
    adapter._download_slack_file_bytes = AsyncMock(return_value=pdf)
    root = {"ts": "123.000", "user": "U_ROOT", "text": "Synthetic test document", "files": [{
        "id": "F_TEST", "name": "synthetic.pdf", "mimetype": "application/pdf",
        "size": len(pdf), "url_private_download": "https://files.slack.com/synthetic.pdf",
    }]}
    event = {"text": "<@U_BOT> Review the document", "user": "U_HUMAN", "channel": "C_TEST",
             "ts": "123.456", "thread_ts": "123.000", "channel_type": "channel", "team": "T_TEST"}
    client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": [root, event]})
    return adapter, client, root, event, pdf


@pytest.mark.asyncio
async def test_authorized_root_pdf_reaches_document_pipeline(scenario):
    adapter, _, _, event, pdf = scenario
    await adapter._handle_slack_message(event)
    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.media_types == ["application/pdf"]
    assert delivered.message_type == MessageType.DOCUMENT
    assert len(delivered.media_urls) == 1
    assert Path(delivered.media_urls[0]).read_bytes() == pdf


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_human", [False, True], ids=["cold", "cached-human"])
@pytest.mark.parametrize("workflow,expected_identity", [
    pytest.param(False, False, id="valid-human"),
    pytest.param(True, True, id="valid-bot"),
    pytest.param(None, None, id="null"),
    pytest.param(0, None, id="numeric-zero"),
    pytest.param(1, None, id="numeric-one"),
    pytest.param("true", None, id="string-true"),
    pytest.param("false", None, id="string-false"),
])
async def test_root_sdk_identity_requires_verified_status(
    scenario, workflow, expected_identity, cached_human,
):
    """Unknown fresh identity cannot authorize files.info, download or caching."""
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

    adapter, client, root, event, pdf = scenario
    payload = {"ok": True, "user": {
        "id": "U_ROOT", "is_bot": False, "is_workflow_bot": workflow,
    }}

    async def users_info(*, user):
        assert user == "U_ROOT"
        return AsyncSlackResponse(
            client=client, http_verb="GET",
            api_url="https://slack.invalid/api/users.info", req_args={},
            data=payload, headers={}, status_code=200,
        ).validate()

    client.users_info.side_effect = users_info
    assert await adapter._resolve_user_is_bot("U_ROOT", "C_TEST", "T_TEST") is expected_identity
    if expected_identity is None:
        assert ("T_TEST", "U_ROOT") not in adapter._user_is_bot_cache
    adapter._user_is_bot_cache.pop(("T_TEST", "U_ROOT"), None)
    if cached_human:
        adapter._user_is_bot_cache[("T_TEST", "U_ROOT")] = False
    client.users_info.reset_mock()
    resolved = dict(root["files"][0])
    root["files"] = [{"id": resolved["id"], "file_access": "check_file_info"}]
    client.files_info.return_value = {"ok": True, "file": resolved}
    authorizations = []

    def authorize(user, chat_type, channel_id, **kwargs):
        if user == "U_HUMAN":
            return True
        if kwargs.get("thread_id") == event["thread_ts"]:
            authorizations.append(kwargs)
        return kwargs.get("is_bot") is not True

    adapter.set_authorization_check(authorize)
    adapter._cache_slack_document = AsyncMock(wraps=adapter._cache_slack_document)
    await adapter._handle_slack_message(event)
    client.users_info.assert_awaited_once_with(user="U_ROOT")
    adapter.handle_message.assert_awaited_once()
    delivered = adapter.handle_message.await_args.args[0]
    if expected_identity is False:
        client.files_info.assert_awaited_once_with(file=resolved["id"])
        adapter._cache_slack_document.assert_awaited_once()
        adapter._download_slack_file_bytes.assert_awaited_once()
        assert len(delivered.media_urls) == 1
        assert Path(delivered.media_urls[0]).read_bytes() == pdf
        assert len(authorizations) == 1
    else:
        client.files_info.assert_not_awaited()
        adapter._cache_slack_document.assert_not_awaited()
        adapter._download_slack_file_bytes.assert_not_awaited()
        assert delivered.media_urls == []
        assert "not loaded" in delivered.channel_context
        if expected_identity is None:
            assert authorizations == []
        else:
            assert len(authorizations) == 1
            assert authorizations[0]["is_bot"] is True


@pytest.mark.asyncio
async def test_root_document_rejects_actual_bytes_over_limit(scenario):
    adapter, _, _, event, _ = scenario
    adapter._download_slack_file_bytes.return_value = b"x" * (20 * 1024 * 1024 + 1)
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.media_urls == []
    assert "not loaded" in delivered.channel_context


@pytest.mark.asyncio
async def test_root_download_stops_reading_at_limit(scenario, monkeypatch):
    import httpx
    import tools.url_safety as safety

    adapter, _, _, event, _ = scenario
    del adapter._download_slack_file_bytes
    adapter._resolve_download_token = lambda *args: "synthetic"
    consumed = []
    closed = []

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(6):
                consumed.append(i)
                yield b"x" * (8 * 1024 * 1024)

        async def aclose(self):
            closed.append(True)

    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, stream=Chunks())))
    monkeypatch.setattr(safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(safety, "create_ssrf_safe_async_client", lambda **kwargs: client)
    await adapter._handle_slack_message(event)
    assert len(consumed) <= 3
    assert closed
    assert adapter.handle_message.await_args.args[0].media_urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["deny", "raise", "missing", "unknown", "truthy"])
async def test_root_authorization_must_be_positive(scenario, mode):
    adapter, client, root, event, _ = scenario
    # A Connect stub proves denial precedes even files.info.
    root["files"] = [{"id": "F_STUB", "file_access": "check_file_info"}]
    def authorize(user, *args, **kwargs):
        if user == "U_HUMAN":
            return True
        if mode == "raise":
            raise RuntimeError("synthetic authorization failure")
        return {"deny": False, "unknown": None, "truthy": "yes"}.get(mode)
    adapter.set_authorization_check(None if mode == "missing" else authorize)
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()
    client.files_info.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"ok": False, "error": "not_allowed"}, {"ok": True},
    {"ok": True, "user": {}},
    {"ok": True, "user": {"id": "U_OTHER", "is_bot": False}},
    {"ok": True, "user": {"id": "U_ROOT"}},
    {"ok": True, "user": {"id": "U_ROOT", "is_bot": "false"}},
])
async def test_malformed_identity_cannot_inherit_human_cache(scenario, payload):
    adapter, client, _, event, _ = scenario
    adapter._user_is_bot_cache[("T_TEST", "U_ROOT")] = False
    client.users_info.return_value = payload
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("markers", [
    {"app_id": "A_TEST"}, {"bot_id": "B_TEST"}, {"bot_profile": {"id": "B_TEST"}},
    {"subtype": "bot_message"}, {"user_profile": {"is_bot": True}},
])
async def test_root_bot_shapes_use_canonical_classifier(scenario, markers):
    adapter, _, root, event, _ = scenario
    root.update(markers)
    observed = []
    def authorize(user, chat_type, channel, **kwargs):
        if user == "U_HUMAN":
            return True
        observed.append((chat_type, channel, kwargs))
        return False
    adapter.set_authorization_check(authorize)
    await adapter._handle_slack_message(event)
    assert any(kwargs.get("is_bot") is True and kwargs.get("thread_id") == "123.000"
               for _, _, kwargs in observed)
    adapter._download_slack_file_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_workspace_never_downloads_with_primary_client(scenario):
    adapter, client, _, event, _ = scenario
    adapter._team_clients = {}
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()
    client.files_info.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("known_by", ["declared", "cache"])
async def test_known_bot_is_authorized_before_thread_document_downloads(scenario, known_by):
    adapter, client, _, event, _ = scenario
    adapter.config.extra["allow_bots"] = "all"
    event["user"] = "U_PEER_BOT"
    if known_by == "declared":
        event["bot_id"] = "B_PEER"
    else:
        adapter._user_is_bot_cache[("T_TEST", "U_PEER_BOT")] = True
    authorizations = []

    def authorize(user, *_args, **_kwargs):
        authorizations.append(user)
        return user != "U_PEER_BOT"

    adapter.set_authorization_check(authorize)

    await adapter._handle_slack_message(event)

    assert authorizations == ["U_PEER_BOT"]
    client.conversations_replies.assert_not_awaited()
    adapter._download_slack_file_bytes.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_self_bot_has_no_authorization_bypass(scenario):
    adapter, _, root, event, _ = scenario
    root.update(user="U_BOT", bot_id="B_SELF")
    adapter.set_authorization_check(lambda user, *args, **kwargs: user == "U_HUMAN")
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_connect_stub_uses_own_workspace_and_real_document_cache(scenario):
    adapter, client, root, event, pdf = scenario
    original = dict(root["files"][0])
    root["files"] = [{"id": "F_TEST", "file_access": "check_file_info"}]
    client.files_info.return_value = {"ok": True, "file": original}
    await adapter._handle_slack_message(event)
    client.files_info.assert_awaited_once_with(file="F_TEST")
    delivered = adapter.handle_message.await_args.args[0]
    assert Path(delivered.media_urls[0]).read_bytes() == pdf


@pytest.mark.asyncio
async def test_same_file_on_current_reply_has_priority(scenario):
    adapter, _, root, event, _ = scenario
    event["files"] = root["files"]
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_awaited_once()
    assert len(adapter.handle_message.await_args.args[0].media_urls) == 1
    # Direct attachment path retains its original uncapped call contract.
    assert "max_bytes" not in adapter._download_slack_file_bytes.await_args.kwargs


@pytest.mark.asyncio
async def test_duplicate_root_file_ids_deliver_once(scenario):
    adapter, _, root, event, _ = scenario
    root["files"] *= 2
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_awaited_once()
    assert len(adapter.handle_message.await_args.args[0].media_urls) == 1


@pytest.mark.asyncio
async def test_active_thread_does_not_redownload_root(scenario):
    adapter, _, _, event, _ = scenario
    adapter._has_active_session_for_thread.return_value = True
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_recovery_limit(scenario):
    adapter, _, root, event, _ = scenario
    root["files"] = [dict(root["files"][0], id=f"F_{i}") for i in range(6)]
    await adapter._handle_slack_message(event)
    assert adapter._download_slack_file_bytes.await_count == 4
    assert "attachment limit" in adapter.handle_message.await_args.args[0].channel_context


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [None, 0, -1, True, "10", 20 * 1024 * 1024 + 1])
async def test_invalid_declared_size_does_not_download(scenario, size):
    adapter, _, root, event, _ = scenario
    root["files"][0]["size"] = size
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_notices_do_not_rewrite_commands_or_leak_urls(scenario):
    adapter, _, root, event, _ = scenario
    root["files"][0]["url_private_download"] += "?private-marker=do-not-leak"
    adapter._download_slack_file_bytes.side_effect = ValueError("private-marker=do-not-leak")
    event["text"] = "<@U_BOT> /status"
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.text.strip() == "/status"
    assert "not loaded" in delivered.channel_context
    assert "private-marker" not in delivered.channel_context + delivered.text


@pytest.mark.asyncio
async def test_missing_filename_uses_supported_mime(scenario):
    adapter, _, root, event, _ = scenario
    root["files"][0]["name"] = None
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.media_types == ["application/pdf"]
    assert delivered.media_urls[0].endswith("document.pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize("stub_mime", [None, "image/png"])
async def test_authorized_connect_image_still_delivers(scenario, stub_mime):
    adapter, client, root, event, _ = scenario
    root["files"] = [{"id": "F_IMAGE", "file_access": "check_file_info", "mimetype": stub_mime}]
    client.files_info.return_value = {"ok": True, "file": {
        "id": "F_IMAGE", "mimetype": "image/png", "name": "image.png",
        "url_private_download": "https://files.slack.com/image.png",
    }}
    adapter._download_slack_file = AsyncMock(return_value="/synthetic/image.png")
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.media_types == ["image/png"]
    client.files_info.assert_awaited_once_with(file="F_IMAGE")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "html", "declared_oversize", "unsafe_host"])
async def test_bounded_download_transport_contract(scenario, monkeypatch, mode):
    import httpx
    import tools.url_safety as safety

    adapter, _, root, event, pdf = scenario
    del adapter._download_slack_file_bytes
    adapter._resolve_download_token = lambda *args: "synthetic"
    calls = []
    def handler(request):
        calls.append(request)
        headers = {"content-type": "text/html" if mode == "html" else "application/pdf"}
        if mode == "declared_oversize":
            headers["content-length"] = str(21 * 1024 * 1024)
        return httpx.Response(200, headers=headers, content=pdf)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(safety, "create_ssrf_safe_async_client", lambda **kwargs: client)
    if mode == "unsafe_host":
        root["files"][0]["url_private_download"] = "https://untrusted.example/file.pdf"
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    if mode == "success":
        assert Path(delivered.media_urls[0]).read_bytes() == pdf
    else:
        assert delivered.media_urls == []
    assert len(calls) == (0 if mode == "unsafe_host" else 1)
    await client.aclose()


@pytest.mark.asyncio
async def test_known_images_keep_priority_and_share_root_limit(scenario):
    adapter, _, root, event, _ = scenario
    docs = [dict(root["files"][0], id=f"F_DOC{i}") for i in range(4)]
    root["files"] = [{"id": "F_IMAGE", "mimetype": "image/png", "name": "image.png",
                      "url_private_download": "https://files.slack.com/image.png"}, *docs]
    adapter._download_slack_file = AsyncMock(return_value="/synthetic/image.png")
    await adapter._handle_slack_message(event)
    delivered = adapter.handle_message.await_args.args[0]
    assert delivered.media_types == ["image/png", *(["application/pdf"] * 3)]
    assert adapter._download_slack_file_bytes.await_count == 3


@pytest.mark.asyncio
async def test_root_audio_video_are_not_downloaded(scenario):
    adapter, _, root, event, _ = scenario
    root["files"] = [dict(root["files"][0], id="F_AUDIO", mimetype="audio/mpeg"),
                     dict(root["files"][0], id="F_VIDEO", mimetype="video/mp4")]
    await adapter._handle_slack_message(event)
    adapter._download_slack_file_bytes.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == []
