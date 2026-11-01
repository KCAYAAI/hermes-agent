"""Tests for cross-loop client cache isolation fix (#2681).

Verifies that _get_cached_client() returns different AsyncOpenAI clients
when called from different event loops, preventing the httpx deadlock
that occurs when a cached async client bound to loop A is reused on loop B.

This test file is self-contained and does not import the full tool chain,
so it can run without optional dependencies like firecrawl.
"""

import asyncio
import threading
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import _get_cached_client without the full tree
# ---------------------------------------------------------------------------

def _stub_resolve_provider_client(provider, model, async_mode, **kw):
    """Return a unique mock client each time, simulating AsyncOpenAI creation."""
    client = MagicMock(name=f"client-{provider}-async={async_mode}")
    client.api_key = "test"
    client.base_url = kw.get("explicit_base_url", "http://localhost:8081/v1")
    return client, model or "test-model"

@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Clear the client cache before each test."""
    # We need to patch before importing
    with patch.dict("sys.modules", {}):
        pass
    # Import and clear
    import agent.auxiliary_client as ac
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()

class TestCrossLoopCacheIsolation:
    """Verify async clients are cached per-event-loop, not globally."""

    def test_different_loops_get_different_clients(self):
        """Different event loops must get separate client instances."""
        from agent.auxiliary_client import _get_cached_client

        results = {}

        def _get_client_on_new_loop(name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with patch("agent.auxiliary_client.resolve_provider_client",
                        side_effect=_stub_resolve_provider_client):
                client, _ = _get_cached_client("custom", "m1", async_mode=True,
                                                 base_url="http://localhost:8081/v1")
            results[name] = (id(client), id(loop))
            # Don't close loop — simulates real usage where loops persist

        t1 = threading.Thread(target=_get_client_on_new_loop, args=("a",))
        t2 = threading.Thread(target=_get_client_on_new_loop, args=("b",))
        t1.start(); t1.join()
        t2.start(); t2.join()

        client_id_a, loop_id_a = results["a"]
        client_id_b, loop_id_b = results["b"]

        assert loop_id_a != loop_id_b, "Test setup error: same loop on both threads"
        assert client_id_a != client_id_b, (
            "Different event loops got the SAME cached client — this causes "
            "httpx cross-loop deadlocks in gateway mode (#2681)"
        )


class _InFlightClient:
    """A process-transport client (Claude DirectSDK shape): a sync ``close()`` cancels its in-flight requests."""

    def __init__(self):
        self.api_key, self.base_url, self.closed = "test", "process://test", False

    def close(self):
        self.closed = True


class TestCrossLoopEvictionKeepsInFlightClients:
    """Parallel tool workers each own a persistent loop (model_tools._run_async), so two concurrent
    ``vision_analyze`` calls look up the same async key from different live loops. Evicting the other
    worker's entry must not close its client: that cancels the request it is still awaiting."""

    def _get_on_new_loop(self, loop, out):
        def run():
            asyncio.set_event_loop(loop)
            with patch("agent.auxiliary_client.resolve_provider_client",
                       side_effect=lambda *a, **k: (_InFlightClient(), "m1")):
                out.append(_get_cached_client_for_test())
        t = threading.Thread(target=run)
        t.start(); t.join()

    def test_live_foreign_loop_client_is_not_closed(self):
        loop_a, loop_b, got = asyncio.new_event_loop(), asyncio.new_event_loop(), []
        try:
            self._get_on_new_loop(loop_a, got)  # worker A: client cached, request in flight
            self._get_on_new_loop(loop_b, got)  # worker B: same key, different live loop
            assert got[0] is not got[1]
            assert not got[0].closed, "evicting a live loop's entry closed a client mid-request"
        finally:
            loop_a.close(); loop_b.close()

    def test_closed_owner_loop_client_is_still_closed(self):
        loop_a, loop_b, got = asyncio.new_event_loop(), asyncio.new_event_loop(), []
        try:
            self._get_on_new_loop(loop_a, got)
            loop_a.close()  # owner gone: nothing can still be awaiting this client
            self._get_on_new_loop(loop_b, got)
            assert got[0].closed
        finally:
            loop_b.close()


def _get_cached_client_for_test():
    from agent.auxiliary_client import _get_cached_client
    client, _ = _get_cached_client("custom", "m1", async_mode=True, base_url="http://localhost:8081/v1")
    return client
