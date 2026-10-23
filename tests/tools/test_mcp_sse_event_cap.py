"""The MCP SDK's own per-SSE-event cap (mcp >= 2.0, 1 MiB default) is aligned with Hermes' wire cap.

Large-but-legitimate remote catalogs (Higgsfield's ``tools/list`` is > 1 MiB in one event) failed
with ``Server-sent event exceeded the 1048576 byte limit`` even though Hermes' own transport cap
(10 MiB per event) would accept them. Hermes now passes its cap to the SDK, with a per-server
``max_sse_event_bytes`` override that raises both caps together.
"""
from __future__ import annotations

import asyncio
import inspect
import urllib.request
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

URL = "https://mcp.example.com/mcp"
_PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy")


@pytest.fixture
def sdk(monkeypatch):
    for key in _PROXY_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(urllib.request, "getproxies", urllib.request.getproxies_environment)
    monkeypatch.setattr(urllib.request, "proxy_bypass", urllib.request.proxy_bypass_environment)
    from tools.mcp_tool import _ensure_mcp_sdk, sdk_httpx

    if not _ensure_mcp_sdk() or sdk_httpx() is None:
        pytest.skip("mcp SDK not installed")


class _RecordingClient:
    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _fake_sdk_client(record: dict, *, supports_cap: bool):
    if supports_cap:
        @asynccontextmanager
        async def streamable_http_client(url, *, http_client=None, terminate_on_close=True,
                                         max_sse_event_size=1024 * 1024):
            record.update(url=url, max_sse_event_size=max_sse_event_size, passed=True)
            yield (None, None)
    else:
        @asynccontextmanager
        async def streamable_http_client(url, *, http_client=None, terminate_on_close=True):
            record.update(url=url, passed=False)
            yield (None, None)
    return streamable_http_client


def _drive(server, config: dict, *, supports_cap: bool = True):
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_transport import _resolve_sse_event_cap

    record: dict = {}
    server._http_rejection = {}
    server._sse_event_cap = _resolve_sse_event_cap(server.name, config)
    with patch("tools.mcp_tool.streamable_http_client", _fake_sdk_client(record, supports_cap=supports_cap)), \
         patch.object(sdk_httpx(), "AsyncClient", _RecordingClient), \
         patch("tools.mcp_tool_transport._make_mcp_body_cap_transport",
               side_effect=lambda httpx_mod, inner, limit=None: ("capped", limit)):
        streams = server._streamable_http_transport(URL, {}, 5.0, True, None, None, False, set())

        async def run():
            async with streams:
                pass
        asyncio.run(run())
    return record, _RecordingClient.captured


def test_default_sdk_cap_matches_hermes_wire_cap(sdk):
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES

    record, client = _drive(MCPServerTask("remote"), {})
    assert record["max_sse_event_size"] == _MCP_HTTP_MAX_BODY_BYTES
    assert _MCP_HTTP_MAX_BODY_BYTES > 1024 * 1024  # strictly above the SDK's old 1 MiB default
    assert client["transport"] == ("capped", _MCP_HTTP_MAX_BODY_BYTES)


def test_per_server_override_raises_both_caps(sdk):
    from tools.mcp_tool import MCPServerTask

    cap = 32 * 1024 * 1024
    record, client = _drive(MCPServerTask("remote"), {"max_sse_event_bytes": cap})
    assert record["max_sse_event_size"] == cap
    assert client["transport"] == ("capped", cap)


def test_lower_override_lowers_sdk_cap_but_keeps_wire_cap_floor(sdk):
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES

    record, client = _drive(MCPServerTask("remote"), {"max_sse_event_bytes": 512 * 1024})
    assert record["max_sse_event_size"] == 512 * 1024
    assert client["transport"] == ("capped", _MCP_HTTP_MAX_BODY_BYTES)


@pytest.mark.parametrize("bad", [0, -5, "lots", True, [1]])
def test_invalid_override_falls_back_to_default(sdk, bad):
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES

    record, _ = _drive(MCPServerTask("remote"), {"max_sse_event_bytes": bad})
    assert record["max_sse_event_size"] == _MCP_HTTP_MAX_BODY_BYTES


def test_older_sdk_without_the_kwarg_is_called_unchanged(sdk):
    from tools.mcp_tool import MCPServerTask

    record, _ = _drive(MCPServerTask("remote"), {}, supports_cap=False)
    assert record["passed"] is False and "max_sse_event_size" not in record


def test_installed_sdk_accepts_the_kwarg_when_it_enforces_a_cap(sdk):
    """Guard against an SDK rename silently dropping the alignment on mcp >= 2.0."""
    from tools import mcp_tool as core
    from tools.mcp_tool_transport import _sdk_sse_event_cap_kwargs

    params = inspect.signature(core.streamable_http_client).parameters
    if "max_sse_event_size" not in params:
        pytest.skip("installed SDK has no per-event SSE cap")
    assert _sdk_sse_event_cap_kwargs(123) == {"max_sse_event_size": 123}


def test_real_sdk_decoder_accepts_a_2mib_event_under_the_aligned_cap(sdk):
    """End to end through the SDK's real SSE parser: a 2 MiB event fails at 1 MiB, passes at 10 MiB."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_errors import _MCP_HTTP_MAX_BODY_BYTES

    httpx = sdk_httpx()
    try:
        from httpx2._sse import EventSource  # mcp >= 2.0 ships SSE parsing in httpx2
    except ImportError:
        pytest.skip("SDK httpx has no built-in SSE parser")
    payload = b"data: " + b"x" * (2 * 1024 * 1024) + b"\n\n"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

    async def read(cap):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async with client.stream("GET", URL) as resp:
                return [len(e.data) async for e in EventSource(resp, max_event_size=cap)]

    with pytest.raises(Exception, match="byte limit"):
        asyncio.run(read(1024 * 1024))
    assert asyncio.run(read(_MCP_HTTP_MAX_BODY_BYTES)) == [2 * 1024 * 1024]
