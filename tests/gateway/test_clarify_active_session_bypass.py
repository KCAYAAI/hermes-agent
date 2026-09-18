"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_session_bypass_uses_profile_namespaced_key_under_multiplex():
    """Regression for issue #82975: under a named-profile multiplex, the
    adapter's clarify bypass lookup must use the SAME profile-namespaced
    session key that the runner registers pending clarifies under
    (SessionStore._generate_session_key() includes
    profile=self._resolve_profile_for_key(source)), not the legacy
    unnamespaced key. Otherwise the lookup misses, and a user's answer to
    a pending clarify is routed to the busy-session queue instead of
    resolving it -- the turn then hangs until the clarify's 3600s timeout."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")

    # A session_store configured for profile multiplexing, matching what
    # the runner's SessionStore._generate_session_key() actually produces.
    session_store = MagicMock()
    session_store._resolve_profile_for_key.return_value = "ops"
    adapter._session_store = session_store

    profile_namespaced_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        profile="ops",
    )
    # Sanity: the profile-namespaced key really is different from the
    # legacy unnamespaced one -- otherwise this test wouldn't distinguish
    # the fixed behavior from the bug.
    legacy_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    assert profile_namespaced_key != legacy_key

    adapter._active_sessions[profile_namespaced_key] = asyncio.Event()
    # The runner registers the pending clarify under its own
    # profile-namespaced key, exactly as it would in a real multiplexed
    # deployment.
    cm.register("clarify-1", profile_namespaced_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_stale_scoped_clarify_marker_stays_on_inline_resolver_path():
    """A prompt may resolve after Slack stamps its exact clarify marker but before
    active-session dispatch. The immutable marker must still reach the runner,
    which safely drops stale identities, instead of becoming a queued user turn.
    """
    _clear_clarify_state()
    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("late answer")
    event.metadata["_hermes_clarify_response_only"] = "already-resolved"
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_default_profile_busy_handler_uses_routed_runtime_and_transport_authorization(
    tmp_path,
):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    routed_home = tmp_path / "profiles" / "worker"
    scoped_homes = []

    @asynccontextmanager
    async def runtime_scope(profile_home):
        scoped_homes.append(Path(profile_home))
        yield

    runner._stamp_routed_profile = lambda source: (
        setattr(source, "profile", "worker") or True)
    runner._resolve_profile_home_for_source = lambda source: routed_home
    runner._session_key_for_source = lambda source: f"agent:{source.profile}:busy"
    runner._handle_active_session_busy_message = AsyncMock(return_value=True)
    event = _event()

    with patch("gateway.run.get_hermes_home", return_value=tmp_path), patch(
        "gateway.run._async_profile_runtime_scope", runtime_scope
    ):
        handled = await runner._make_default_profile_busy_session_handler()(
            event, "transport-key")

    assert handled is True
    assert event.source.profile == "worker"
    assert event.source._authorization_profile_home == tmp_path
    assert scoped_homes == [routed_home]
    runner._handle_active_session_busy_message.assert_awaited_once_with(
        event, "agent:worker:busy")


def test_wire_adapter_handlers_uses_default_profile_busy_wrapper_under_multiplex():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.session_store = object()
    runner._busy_text_mode = "queue"
    runner._make_default_profile_message_handler = MagicMock(
        return_value="message-handler")
    runner._make_default_profile_busy_session_handler = MagicMock(
        return_value="busy-handler")
    runner._handle_adapter_fatal_error = MagicMock()
    runner._handle_reaction_event = MagicMock()
    runner._recover_telegram_topic_thread_id = MagicMock()
    runner._make_adapter_auth_check = MagicMock(return_value="auth-check")
    runner._make_default_profile_platform_event_handler = MagicMock(
        return_value="platform-handler")
    adapter = MagicMock()
    adapter.platform = Platform.SLACK

    runner._wire_adapter_handlers(adapter)

    adapter.set_message_handler.assert_called_once_with("message-handler")
    adapter.set_busy_session_handler.assert_called_once_with("busy-handler")
    adapter.set_authorization_check.assert_called_once_with("auth-check")
