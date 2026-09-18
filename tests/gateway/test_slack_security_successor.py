"""Display-name lookups must not grant strict clarification authority."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from slack_sdk.web.async_slack_response import AsyncSlackResponse

from gateway.config import PlatformConfig
from gateway.run_inbound import GatewayInboundMixin
from plugins.platforms.slack.adapter import SlackAdapter
from tools import clarify_gateway


TEAM, CHANNEL, USER, BOT = "T12345678", "C12345678", "U12345678", "U87654321"


def make_adapter(payload):
    adapter = SlackAdapter(PlatformConfig(
        enabled=True, token="synthetic-test-only", extra={
            "allowed_channels": [CHANNEL], "require_mention": True,
            "strict_mention": True, "thread_require_mention": True,
            "reply_in_thread": True, "allow_bots": "all",
        },
    ))
    client = SimpleNamespace(users_info=AsyncMock(return_value=payload))
    adapter._app = SimpleNamespace(client=client)
    adapter._team_clients = {TEAM: client}
    adapter._team_bot_user_ids = {TEAM: BOT}
    adapter._bot_user_id = BOT
    adapter._running = True
    adapter.set_authorization_check(lambda *_args, **_kwargs: True)
    adapter._session_store = SimpleNamespace(config=SimpleNamespace(
        group_sessions_per_user=True, thread_sessions_per_user=False),
        _resolve_profile_for_key=lambda source: None)
    adapter._has_active_session_for_thread = lambda **_kwargs: False
    adapter._fetch_thread_context = AsyncMock(return_value="")
    adapter._fetch_thread_parent_text = AsyncMock(return_value="")
    adapter._resolve_channel_name = AsyncMock(return_value="Synthetic room")
    adapter.handle_message = AsyncMock()
    return adapter, client


async def answer_clarify(adapter, team=TEAM):
    key = adapter._build_thread_session_key(CHANNEL, "100.000", USER, team_id=team)
    entry = clarify_gateway.register("identity-cache-test", key, "Choose a route", [])
    try:
        await adapter._handle_slack_message({
            "type": "message", "channel": CHANNEL, "channel_type": "channel",
            "team": team, "user": USER, "text": "custom answer", "ts": "101.000",
            "thread_ts": "100.000", "client_msg_id": "synthetic-message-id",
        })
        if adapter.handle_message.await_count:
            captured = adapter.handle_message.await_args.args[0]
            assert captured.metadata["_hermes_clarify_response_only"] == "identity-cache-test"
            runner = SimpleNamespace(
                _pending_event_audio_paths=lambda ev: [],
                _prepare_clarify_reply_text=AsyncMock(return_value=captured.text),
                _adapter_for_source=lambda source: None,
            )
            await GatewayInboundMixin._hm_clarify_reply(runner, captured, captured.source, key)
        return entry.event.is_set(), entry.response
    finally:
        clarify_gateway.clear_session(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "name-warmed"])
@pytest.mark.parametrize("flags, expected_status", [
    pytest.param({}, None, id="missing-flags"),
    pytest.param({"is_bot": None}, None, id="null-bot"),
    pytest.param({"is_bot": 0}, None, id="numeric-bot"),
    pytest.param({"is_bot": ""}, None, id="empty-bot"),
    pytest.param({"is_bot": "false"}, None, id="string-bot"),
    pytest.param({"is_bot": []}, None, id="list-bot"),
    pytest.param({"is_bot": False, "is_workflow_bot": None}, None, id="null-workflow"),
    pytest.param({"is_bot": False, "is_workflow_bot": 0}, None, id="numeric-workflow"),
    pytest.param({"is_workflow_bot": False}, None, id="workflow-false-only"),
    pytest.param({"is_bot": False}, False, id="human"),
    pytest.param({"is_bot": False, "is_workflow_bot": False}, False, id="human-both-flags"),
    pytest.param({"is_bot": True}, True, id="bot"),
    pytest.param({"is_bot": True, "is_workflow_bot": True}, True, id="workflow-bot"),
    pytest.param({"is_bot": False, "is_workflow_bot": True}, True, id="workflow-overrides-human"),
    pytest.param({"is_bot": True, "profile": {"bot_id": "B12345678"}}, True, id="profile-bot"),
    pytest.param({"api_failure": True}, None, id="api-failure"),
    pytest.param({"api_exception": True}, None, id="api-exception"),
    pytest.param({"invalid_user": True}, None, id="invalid-user"),
])
async def test_only_verified_humans_can_answer_clarify(warm, flags, expected_status):
    user = {"id": USER, "profile": {"display_name": "Synthetic identity"}, **flags}
    payload = {"ok": True, "user": user}
    expected_name = "Synthetic identity"
    if "profile" in flags:
        expected_name = USER
    if flags.get("api_failure"):
        payload = {"ok": False, "user": {"is_bot": False}}
        expected_name = USER
    if flags.get("invalid_user"):
        payload["user"] = []
        expected_name = USER
    adapter, client = make_adapter(payload)
    if flags.get("api_exception"):
        client.users_info.side_effect = RuntimeError("synthetic lookup failure")
        expected_name = USER
    if warm:
        assert await adapter._humanize_user_mentions(
            f"Discuss with <@{USER}>", CHANNEL, TEAM,
        ) == f"Discuss with @{expected_name}"
    resolved, response = await answer_clarify(adapter)
    assert resolved is (expected_status is False)
    if expected_status is False:
        assert response == "custom answer"
        adapter.handle_message.assert_awaited_once()
    else:
        assert response is None
        adapter.handle_message.assert_not_awaited()
    assert adapter._user_is_bot_cache.get((TEAM, USER)) is expected_status
    if expected_status is None:
        assert (TEAM, USER) not in adapter._user_is_bot_cache
        # A cached display name must not prevent later identity recovery.
        client.users_info.side_effect = None
        client.users_info.return_value = {"ok": True, "user": {"id": USER, "is_bot": False}}
        assert await adapter._resolve_user_is_bot(USER, CHANNEL, TEAM) is False
    else:
        calls = client.users_info.await_count
        assert await adapter._resolve_user_is_bot(USER, CHANNEL, TEAM) is expected_status
        assert client.users_info.await_count == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("first_is_human", [False, True])
async def test_name_and_identity_caches_remain_workspace_scoped(first_is_human):
    other_team = "T87654321"
    first_user = {"id": USER, "profile": {"display_name": "First workspace"}}
    other_user = {"id": USER, "profile": {"display_name": "Other workspace"}}
    (first_user if first_is_human else other_user)["is_bot"] = False
    adapter, first_client = make_adapter({"ok": True, "user": first_user})
    other_client = SimpleNamespace(users_info=AsyncMock(
        return_value={"ok": True, "user": other_user}))
    adapter._team_clients[other_team] = other_client
    adapter._team_bot_user_ids[other_team] = BOT
    assert await adapter._humanize_user_mentions(
        f"<@{USER}>", CHANNEL, TEAM) == "@First workspace"
    assert await adapter._resolve_user_is_bot(USER, CHANNEL, TEAM) is (
        False if first_is_human else None)
    assert await adapter._humanize_user_mentions(
        f"<@{USER}>", CHANNEL, other_team) == "@Other workspace"
    resolved, response = await answer_clarify(adapter, other_team)
    assert resolved is (not first_is_human)
    assert response == (None if first_is_human else "custom answer")
    assert adapter._user_is_bot_cache.get((TEAM, USER)) is (
        False if first_is_human else None)
    assert adapter._user_is_bot_cache.get((other_team, USER)) is (
        None if first_is_human else False)
    assert first_client.users_info.await_count == 2
    # Unknown identity can be retried by more than one routing guard.
    assert other_client.users_info.await_count >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "name-warmed"])
@pytest.mark.parametrize("envelope, user, expected_status", [
    pytest.param({}, {"id": USER, "is_bot": False}, None, id="missing-ok"),
    pytest.param({"ok": False}, {"id": USER, "is_bot": False}, None, id="false-ok"),
    pytest.param({"ok": None}, {"id": USER, "is_bot": False}, None, id="null-ok"),
    pytest.param({"ok": 0}, {"id": USER, "is_bot": False}, None, id="zero-ok"),
    pytest.param({"ok": 1}, {"id": USER, "is_bot": False}, None, id="numeric-ok"),
    pytest.param({"ok": "true"}, {"id": USER, "is_bot": False}, None, id="string-true-ok"),
    pytest.param({"ok": "false"}, {"id": USER, "is_bot": False}, None, id="string-false-ok"),
    pytest.param({"ok": [True]}, {"id": USER, "is_bot": False}, None, id="list-ok"),
    pytest.param({"ok": {"accepted": True}}, {"id": USER, "is_bot": False}, None, id="dict-ok"),
    pytest.param({"ok": True}, None, None, id="null-user"),
    pytest.param({"ok": True}, [], None, id="list-user"),
    pytest.param({"ok": True}, "user", None, id="string-user"),
    pytest.param({"ok": True}, {"is_bot": False}, None, id="missing-id"),
    pytest.param({"ok": True}, {"id": None, "is_bot": False}, None, id="null-id"),
    pytest.param({"ok": True}, {"id": BOT, "is_bot": False}, None, id="wrong-id"),
    pytest.param({"ok": True}, {"id": [USER], "is_bot": False}, None, id="list-id"),
    pytest.param({"ok": True}, {"id": USER}, None, id="missing-flags"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": None}, None, id="null-flag"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": 0}, None, id="numeric-flag"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": "false"}, None, id="string-flag"),
    pytest.param({"ok": True}, {"id": USER, "is_workflow_bot": False}, None, id="workflow-only-false"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": False, "is_workflow_bot": 0}, None, id="numeric-workflow"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": False}, False, id="verified-human"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": True}, True, id="verified-bot"),
    pytest.param({"ok": True}, {"id": USER, "is_workflow_bot": True}, None, id="workflow-only-true"),
    pytest.param({"ok": True}, {"id": USER, "is_bot": True, "is_workflow_bot": True}, True, id="verified-workflow-bot"),
])
async def test_sdk_validated_identity_envelope_controls_exact_clarify(
    warm, envelope, user, expected_status,
):
    # The installed SDK validates truthiness, not literal True. Exercise its
    # real response validator before the adapter's stricter identity boundary.
    payload = {**envelope, "user": user}
    adapter, client = make_adapter(payload)

    async def sdk_users_info(*, user):
        assert user == USER
        return AsyncSlackResponse(
            client=client, http_verb="GET",
            api_url="https://slack.invalid/api/users.info", req_args={},
            data=payload, headers={}, status_code=200,
        ).validate()

    client.users_info.side_effect = sdk_users_info
    if warm:
        await adapter._humanize_user_mentions(f"Discuss with <@{USER}>", CHANNEL, TEAM)
        assert (TEAM, USER) not in adapter._user_is_bot_cache
    resolved, response = await answer_clarify(adapter)
    assert resolved is (expected_status is False)
    assert response == ("custom answer" if expected_status is False else None)
    assert adapter._user_is_bot_cache.get((TEAM, USER)) is expected_status
    if expected_status is None:
        assert (TEAM, USER) not in adapter._user_is_bot_cache
        adapter.handle_message.assert_not_awaited()
        # A previously cached name must not prevent later verified recovery.
        payload = {"ok": True, "user": {"id": USER, "is_bot": False}}
        assert await adapter._resolve_user_is_bot(USER, CHANNEL, TEAM) is False
    elif expected_status is False:
        adapter.handle_message.assert_awaited_once()
    else:
        adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("registered_outcome", [None, RuntimeError("synthetic auth failure")])
async def test_installed_unknown_authorization_is_terminal_for_clarify(registered_outcome):
    payload = {"ok": True, "user": {"id": USER, "is_bot": False}}
    adapter, client = make_adapter(payload)
    fallback = MagicMock(return_value=True)

    class Runner:
        _is_user_authorized = fallback

        async def message_handler(self, event):
            return None

    runner = Runner()
    adapter._message_handler = runner.message_handler

    def registered(*_args, **_kwargs):
        if isinstance(registered_outcome, Exception):
            raise registered_outcome
        return registered_outcome

    adapter.set_authorization_check(registered)
    before_teams = dict(adapter._channel_team)

    resolved, response = await answer_clarify(adapter)

    assert resolved is False
    assert response is None
    assert client.users_info.await_count == 0
    assert fallback.call_count == 0
    assert adapter._channel_team == before_teams
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"ok": True, "user": {"id": USER, "is_bot": None}},
    {"ok": True, "user": {"id": BOT, "is_bot": False}},
])
async def test_failed_clarify_identity_stops_before_context_and_routing(payload):
    adapter, client = make_adapter(payload)
    adapter._agent_view_context_for_event = MagicMock(return_value="must-not-run")
    adapter._remember_channel_team = MagicMock()
    before_teams = dict(adapter._channel_team)

    resolved, response = await answer_clarify(adapter)

    assert resolved is False
    assert response is None
    assert client.users_info.await_count == 1
    adapter._agent_view_context_for_event.assert_not_called()
    adapter._remember_channel_team.assert_not_called()
    assert adapter._channel_team == before_teams
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_authorization_routes_do_not_lookup_or_dispatch_clarify():
    adapter, client = make_adapter({"ok": True, "user": {"id": USER, "is_bot": False}})
    adapter.set_authorization_check(None)
    adapter._pending_text_clarify_id_for_thread = MagicMock(side_effect=AssertionError("pending lookup must follow authorization"))
    await adapter._handle_slack_message({
        "type": "message", "channel": CHANNEL, "channel_type": "channel",
        "team": TEAM, "user": USER, "text": "custom answer", "ts": "101.000",
        "thread_ts": "100.000", "client_msg_id": "synthetic-message-id",
    })
    adapter._pending_text_clarify_id_for_thread.assert_not_called()
    client.users_info.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_denied_sender_stops_before_replay_and_pending_state():
    adapter, client = make_adapter({"ok": True, "user": {"id": USER, "is_bot": False}})
    order = []
    adapter.set_authorization_check(lambda *_args, **_kwargs: order.append("authorization") or False)
    class Claims(dict):
        def __contains__(self, key):
            order.append(("processed", key))
            return super().__contains__(key)
    adapter._processed_message_ts = Claims()
    adapter._dedup.is_duplicate = MagicMock(side_effect=lambda *_: order.append("dedup") or False)
    adapter._pending_text_clarify_id_for_thread = MagicMock(side_effect=lambda **_: order.append("pending") or None)
    await adapter._handle_slack_message({
        "type": "message", "channel": CHANNEL, "channel_type": "channel",
        "team": TEAM, "user": USER, "text": "answer", "ts": "101.000",
        "thread_ts": "100.000", "client_msg_id": "message-id",
    })
    assert order == ["authorization"]
    client.users_info.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_primary_route_precedes_thread_key_and_source_profile_stamping():
    adapter, _client = make_adapter({"ok": True, "user": {"id": USER, "is_bot": False}})
    adapter.gateway_runner = SimpleNamespace(_profile_name_for_source=lambda source, adapter_profile=None: "worker")
    adapter._session_store = SimpleNamespace(
        config=SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False),
        _resolve_profile_for_key=lambda source: source.profile or "default",
    )
    key = adapter._build_thread_session_key(CHANNEL, "100.000", USER, team_id=TEAM)
    message = await adapter._build_message_event(
        {}, text="hello", original_text="hello", command_probe_text="hello", is_command_text=False,
        channel_id=CHANNEL, team_id=TEAM, ts="101.000", user_id=USER, thread_ts="100.000",
        is_dm=False, media_urls=[], media_types=[], channel_context=None)
    assert key is not None and key.startswith("agent:worker:")
    assert message.source.profile == "worker"
