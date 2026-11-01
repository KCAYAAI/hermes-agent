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


class TestInsertionRaceKeepsEachLoopsOwnClient:
    """Two loops miss the same async key and both build (the build runs outside the cache lock). The one that
    inserts second must not hand its caller the winner's client: that client is bound to the other loop, and
    using it from this loop deadlocks or fails the request (#2681). It keeps its own client, uncached, unless
    the winner's loop is already closed, in which case its entry is stale and is replaced."""

    def _race(self, loop_a, loop_b, close_a_first=False, sync=False):
        built, got = {}, {}

        def winner():
            if loop_a is None:
                asyncio.set_event_loop(None)  # a worker thread with no loop: its entry's owner is unknown
            else:
                asyncio.set_event_loop(loop_a)
            with patch("agent.auxiliary_client.resolve_provider_client",
                       side_effect=lambda *a, **k: (built.setdefault("a", _InFlightClient()), "m1")):
                got["a"] = _get_cached_client_for_test(sync=sync)

        def loser_build(*a, **k):
            # While B builds outside the lock, A misses, builds and inserts first.
            t = threading.Thread(target=winner)
            t.start(); t.join()
            if close_a_first:
                loop_a.close()
            return built.setdefault("b", _InFlightClient()), "m1"

        def loser():
            asyncio.set_event_loop(loop_b)
            with patch("agent.auxiliary_client.resolve_provider_client", side_effect=loser_build):
                got["b"] = _get_cached_client_for_test(sync=sync)

        t = threading.Thread(target=loser)
        t.start(); t.join()
        return built, got

    def test_loser_on_another_live_loop_keeps_its_own_client(self):
        import agent.auxiliary_client as ac
        loop_a, loop_b = asyncio.new_event_loop(), asyncio.new_event_loop()
        try:
            built, got = self._race(loop_a, loop_b)
            assert got["a"] is built["a"] and got["b"] is built["b"], "loop B was handed loop A's client"
            assert not built["a"].closed and not built["b"].closed
            (entry,) = ac._client_cache.values()
            assert entry[0] is built["a"] and entry[2] is loop_a, "the live winner's entry must stay cached"
        finally:
            for loop in (loop_a, loop_b):
                if not loop.is_closed():
                    loop.close()

    def test_loser_replaces_a_winner_whose_loop_already_closed(self):
        import agent.auxiliary_client as ac
        loop_a, loop_b = asyncio.new_event_loop(), asyncio.new_event_loop()
        try:
            built, got = self._race(loop_a, loop_b, close_a_first=True)
            assert got["b"] is built["b"] and not built["b"].closed
            assert built["a"].closed, "a closed loop's client is stale and is closed, as on a cache hit"
            (entry,) = ac._client_cache.values()
            assert entry[0] is built["b"] and entry[2] is loop_b
        finally:
            loop_b.close()

    def test_loser_replaces_an_unknown_owner_winner_without_closing_it(self):
        # A worker with no event loop wins with cached_loop=None and has already handed its client to
        # its caller. An unknown owner is not a closed owner: replace the entry, never close the client.
        import agent.auxiliary_client as ac
        loop_b = asyncio.new_event_loop()
        try:
            built, got = self._race(None, loop_b)
            assert got["a"] is built["a"] and got["b"] is built["b"]
            assert not built["a"].closed, "an unknown-owner winner's client was closed after it was returned"
            assert not built["b"].closed
            (entry,) = ac._client_cache.values()
            assert entry[0] is built["b"] and entry[2] is loop_b
        finally:
            loop_b.close()

    def test_sync_race_still_returns_the_winner_and_closes_the_loser(self):
        import agent.auxiliary_client as ac
        loop_a, loop_b = asyncio.new_event_loop(), asyncio.new_event_loop()
        try:
            built, got = self._race(loop_a, loop_b, sync=True)
            assert got["b"] is built["a"] and built["b"].closed and not built["a"].closed
            (entry,) = ac._client_cache.values()
            assert entry[0] is built["a"]
        finally:
            loop_a.close(); loop_b.close()


def _get_cached_client_for_test(sync=False):
    from agent.auxiliary_client import _get_cached_client
    client, _ = _get_cached_client("custom", "m1", async_mode=not sync, base_url="http://localhost:8081/v1")
    return client
