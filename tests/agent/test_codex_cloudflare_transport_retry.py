"""Codex WAF retries preserve OAuth and replace transport, within the existing budget."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error

CODEX_URL = "https://chatgpt.com/backend-api/codex"
HTML = (
    "<!doctype html><html><title>Attention Required! | Cloudflare</title></html>",
    "<!doctype html><html><title>Unable to load site</title>"
    "If you are using a VPN, try turning it off. Ray ID: test-ray</html>",
    "<!doctype html><html>Enable JavaScript and cookies to continue</html>",
    "<!doctype html><script src='/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page'></script>",
)


def waf_error(body=HTML[0], *, status=403, headers=None):
    response = httpx.Response(
        status, text=body, headers=headers or {"content-type": "text/html"},
        request=httpx.Request("POST", CODEX_URL + "/responses"),
    )
    return openai.PermissionDeniedError("Forbidden", response=response, body=body)


def classify(error, provider="openai-codex"):
    return classify_api_error(error, provider=provider, base_url=CODEX_URL)


@pytest.mark.parametrize("body", HTML)
@pytest.mark.parametrize("as_bytes", [False, True])
def test_codex_html_block_retries_without_credential_rotation(body, as_bytes):
    error = waf_error(body)
    if as_bytes:
        error.body = body.encode()
    verdict = classify(error)
    assert verdict.reason == FailoverReason.upstream_blocked
    assert verdict.retryable and not verdict.should_fallback
    assert not verdict.should_rotate_credential and not verdict.is_auth
    assert verdict.error_context["codex_cloudflare_block"] is True


def test_challenge_header_without_html_message_is_retryable():
    verdict = classify(waf_error("Forbidden", headers={"cf-mitigated": "challenge"}))
    assert verdict.retryable and verdict.reason == FailoverReason.upstream_blocked
    assert verdict.error_context["codex_cloudflare_block"] is True


@pytest.mark.parametrize("body", ["Invalid OAuth token", "Your request was blocked."])
def test_generic_codex_403_is_not_a_transport_retry(body):
    verdict = classify(waf_error(body))
    assert not verdict.retryable
    assert not verdict.error_context.get("codex_cloudflare_block")


@pytest.mark.parametrize("body", HTML)
def test_other_provider_waf_does_not_gain_codex_retries(body):
    verdict = classify(waf_error(body), provider="openai-api")
    assert not verdict.retryable
    assert not verdict.error_context.get("codex_cloudflare_block")


def test_401_with_waf_html_stays_auth():
    assert classify(waf_error(status=401)).is_auth


def make_agent():
    old, fresh = object(), object()
    agent = SimpleNamespace(
        provider="openai-codex", api_mode="codex_responses", base_url=CODEX_URL,
        model="test-model", api_key="synthetic-current-token", _fallback_activated=False,
        client=old, _client_kwargs={"api_key": "synthetic-current-token", "base_url": CODEX_URL},
        _primary_runtime={"api_key": "synthetic-stale-token"},
        _retire_shared_openai_client=Mock(), _create_openai_client=Mock(return_value=fresh),
    )
    return agent, old, fresh


def test_refresh_retires_shared_transport_without_restoring_stale_credentials():
    from agent.agent_runtime_helpers import refresh_codex_cloudflare_transport
    agent, old, fresh = make_agent()
    verdict = classify(waf_error())
    original_kwargs = dict(agent._client_kwargs)
    assert refresh_codex_cloudflare_transport(agent, verdict)
    agent._retire_shared_openai_client.assert_called_once_with(old, reason="codex_cloudflare_403")
    agent._create_openai_client.assert_called_once_with(
        original_kwargs, reason="codex_cloudflare_403", shared=True,
    )
    assert agent.client is fresh
    assert agent.api_key == "synthetic-current-token"
    assert agent._client_kwargs == original_kwargs


@pytest.mark.parametrize("change", ["foreign-provider", "fallback-active", "unmarked", "nonretryable", "missing-kwargs"])
def test_refresh_guards_do_not_retire_other_transports(change):
    from agent.agent_runtime_helpers import refresh_codex_cloudflare_transport
    agent, old, _ = make_agent()
    verdict = classify(waf_error())
    if change == "foreign-provider":
        agent.provider = "openai-api"
    elif change == "fallback-active":
        agent._fallback_activated = True
    elif change == "unmarked":
        verdict.error_context.clear()
    elif change == "nonretryable":
        verdict.retryable = False
    else:
        agent._client_kwargs = {}
    assert not refresh_codex_cloudflare_transport(agent, verdict)
    agent._retire_shared_openai_client.assert_not_called()
    agent._create_openai_client.assert_not_called()
    assert agent.client is old


def test_rebuild_failure_is_contained():
    from agent.agent_runtime_helpers import refresh_codex_cloudflare_transport
    agent, _, _ = make_agent()
    agent._create_openai_client.side_effect = RuntimeError("synthetic rebuild failure")
    assert not refresh_codex_cloudflare_transport(agent, classify(waf_error()))
    assert agent.api_key == "synthetic-current-token"


def test_waf_does_not_refresh_or_exhaust_oauth_pool():
    from agent.agent_runtime_helpers import recover_with_credential_pool
    pool = Mock(provider="openai-codex")
    pool.current.return_value = SimpleNamespace(id="test-entry", runtime_api_key="synthetic-current-token")
    pool.entries.return_value = [pool.current.return_value]
    agent, _, _ = make_agent()
    agent._credential_pool = pool
    agent._is_entitlement_failure = lambda *args, **kwargs: False
    recover_with_credential_pool(
        agent, status_code=403, has_retried_429=False,
        classified_reason=classify(waf_error()).reason, error_context={"message": HTML[0]},
    )
    pool.try_refresh_matching.assert_not_called()
    pool.mark_exhausted_and_rotate.assert_not_called()


@pytest.mark.parametrize("exhausted", [False, True])
def test_retry_budget_and_eventual_fallback_are_preserved(monkeypatch, exhausted):
    import agent.turn_api_error as handler
    from agent.turn_retry_state import TurnRetryState
    agent = Mock()
    agent.provider = "openai-codex"
    agent.api_mode = "codex_responses"
    agent.base_url = CODEX_URL
    agent.model = "test-model"
    agent._try_activate_fallback.return_value = exhausted
    agent._try_recover_primary_transport.return_value = False
    agent._has_pending_fallback.return_value = True
    monkeypatch.setattr(handler, "compute_error_backoff", lambda *args, **kwargs: 0)
    monkeypatch.setattr(handler, "interruptible_backoff_sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(handler, "_arm_fallback_restart", lambda *args: "test-prompt", raising=False)
    # The function imports this helper locally.
    import agent.conversation_loop as loop
    monkeypatch.setattr(loop, "_arm_fallback_restart", lambda *args: "test-prompt")
    result = handler.settle_unrecovered_error(
        agent, api_error=waf_error(), classified=classify(waf_error()), _retry=TurnRetryState(),
        status_code=403, error_msg=HTML[0], is_context_length_error=False, is_rate_limited=False,
        _is_zai_coding_overload=False, _provider=agent.provider, _base=CODEX_URL, _model=agent.model,
        messages=[], api_messages=[], api_kwargs={}, active_system_prompt="", conversation_history=[],
        approx_tokens=1, retry_count=3 if exhausted else 1, max_retries=3,
        compression_attempts=0, api_call_count=1,
    )
    if exhausted:
        assert result.action == "break"
        agent._try_activate_fallback.assert_called_once()
    else:
        assert result.action == "fallthrough"
        agent._try_activate_fallback.assert_not_called()


@pytest.mark.parametrize("retry_count,should_refresh", [(0, True), (2, False)])
def test_error_handler_refreshes_transport_before_error_hook(monkeypatch, retry_count, should_refresh):
    import agent.turn_api_error as handler
    from agent.turn_retry_state import TurnRetryState
    agent, old, fresh = make_agent()
    agent.thinking_callback = None
    agent.context_compressor = None
    agent._extract_api_error_context = lambda error: {}
    monkeypatch.setattr(handler, "recover_before_classification", lambda *args, **kwargs: (False, ""))

    class StopAfterHook(Exception):
        pass

    def hook(**kwargs):
        assert agent.client is (fresh if should_refresh else old)
        assert kwargs["retryable"] is True
        assert kwargs["reason"] == "upstream_blocked"
        raise StopAfterHook

    agent._invoke_api_request_error_hook = hook
    with pytest.raises(StopAfterHook):
        handler.handle_api_error(
            agent, api_error=waf_error(), _retry=TurnRetryState(), thinking_spinner=None, messages=[],
            api_messages=[], api_kwargs={}, system_message={}, active_system_prompt="",
            conversation_history=[], approx_tokens=1, retry_count=retry_count, max_retries=3,
            compression_attempts=0, max_compression_attempts=2, api_call_count=1,
            api_request_id="test-request", api_start_time=0, effective_task_id="test-task", turn_id="test-turn",
        )
    if should_refresh:
        agent._retire_shared_openai_client.assert_called_once_with(old, reason="codex_cloudflare_403")
    else:
        agent._retire_shared_openai_client.assert_not_called()
        agent._create_openai_client.assert_not_called()
