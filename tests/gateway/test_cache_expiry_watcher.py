"""Gateway owner of cache-expiry compaction (``run_watchers``): each due idle chat is compacted in
its own background job under the chat's profile scope, holding the session authority's claim gate
so no turn starts on the history being replaced, and a detached Desktop conversation (cached agent,
no routing entry of its own) is resolved through its live conversation."""

import asyncio
import contextlib
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionEntry, SessionSource
from gateway.config import Platform
from gateway.session_authority import LiveSession
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.waiting import async_wait_until

HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


def _source(chat_id="c1", detached_route=None):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1", chat_type="dm",
                         detached_route=detached_route)


def _entry(key, session_id, source):
    now = datetime.now()
    return SessionEntry(session_key=key, session_id=session_id, created_at=now, updated_at=now,
                        origin=source, platform=source.platform, chat_type=source.chat_type)


class _Store:
    """The SessionStore surface the watcher and ``route_entry`` read."""

    def __init__(self, entries):
        self.entries = entries

    def lookup_by_session_key(self, key):
        return self.entries.get(key)

    def load_transcript(self, session_id):
        return list(HISTORY)

    def _named_profile_for_key(self, key):
        return None  # standalone gateway: every key is the launch profile's

    def _persistence_for_key(self, key):
        return None  # no store: a detached entry starts from its key alone

    def _lazy(self, name, factory):
        return self.__dict__.setdefault(name, factory())


class _Edge:
    """The warmer surface the owners read: ``claim_edge`` answers *action* and records each claim."""

    def __init__(self, action="compact"):
        self.action, self.claims = action, []

    def claim_edge(self, *, releasing=False, now=None):
        self.claims.append(releasing)
        return self.action


class _Authority:
    """The authority surface the watcher reads, over a real admission store: the FIFO's claim is a
    durable ``started`` row, as in production."""

    def __init__(self, runner, persistence):
        self.runner, self.sessions, self.profile_id = runner, {}, "default"
        self.persistence = persistence
        self.epoch = persistence.runtime.begin_epoch(instance_id="t")

    def logical_owner(self, session_id):
        return session_id


@pytest.fixture
def make_runner(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner

    runners = []

    def make(entries, keys, *, authority=False):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.session_store = _Store(entries)
        if authority:
            (home := tmp_path / f"home{len(runners)}").mkdir()
            db = storage_handle(home)
            for session_id in {entry.session_id for entry in entries.values()} | {k.rsplit(":", 1)[-1] for k in keys}:
                seed_session(db, session_id, "telegram")
            runner.session_authority = _Authority(runner, db)
        else:
            runner.session_authority = None
        runner._agent_cache = {key: (SimpleNamespace(session_id=entries[key].session_id if key in entries
                                                     else key.rsplit(":", 1)[-1], _prompt_cache_warmer=_Edge()),
                                     "sig") for key in keys}
        runner._agent_cache_lock = threading.Lock()
        runner.compacted = []
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        for task in list(getattr(runner, "_background_tasks", ())):
            task.cancel()


async def _compact_and_settle(runner):
    """One scan, then every compaction job it started (they are background tasks)."""
    await runner._compact_idle_sessions_before_cache_expiry()
    await asyncio.gather(*getattr(runner, "_background_tasks", ()), return_exceptions=True)


@pytest.mark.asyncio
async def test_standalone_gateway_compacts_inside_the_launch_profile_scope(make_runner):
    """``multiplex_profiles`` off still binds the launch scope: once a hosted room flips the
    credential guard, an unscoped summary would fail closed on every credential read."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"])
    scope = []

    @contextlib.contextmanager
    def launch_scope():
        scope.append("entered")
        yield
        scope.append("exited")

    runner._standalone_launch_scope = launch_scope

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append((entry.session_id, trigger, list(scope)))
    runner._run_manual_compression = compress

    await _compact_and_settle(runner)

    assert runner.compacted == [("s1", "cache_expiry", ["entered"])]
    assert scope == ["entered", "exited"]


@pytest.mark.asyncio
async def test_a_turn_claim_waits_for_the_compaction_holding_the_claim_gate(make_runner):
    """A message arriving mid-summary must not start a reply on the old history while the summary
    replaces it: the authority's FIFO claim takes ``claim_gate``, which the compaction holds."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    live = runner.session_authority.sessions["s1"] = LiveSession(_source(), "k1")
    release, order = asyncio.Event(), []

    async def compress(source, entry, history, request, *, trigger=None):
        order.append("summary started")
        await release.wait()
        order.append("summary committed")
    runner._run_manual_compression = compress

    scan = asyncio.create_task(runner._compact_idle_sessions_before_cache_expiry())
    await async_wait_until(lambda: order == ["summary started"], timeout=5)

    authority = runner.session_authority
    db = authority.persistence
    # A person's input is queued (not claimed): it does not stop the summary, it waits behind it.
    db.admissions.admit(epoch=authority.epoch, principal_id="u1", session_id="s1", request_id="r1",
                        payload={"text": "hi"})

    async def claim():
        async with live.claim_gate:
            await asyncio.to_thread(db.admissions.claim, epoch=authority.epoch, session_id="s1")
            order.append("turn claimed")
    claimer = asyncio.create_task(claim())
    await asyncio.sleep(0)
    assert order == ["summary started"]  # the claim is parked behind the compaction
    release.set()
    await asyncio.wait_for(claimer, timeout=5)
    assert order == ["summary started", "summary committed", "turn claimed"]
    await asyncio.wait_for(scan, timeout=5)


def _record_cache_edge_work(runner, monkeypatch):
    """Every summary or cold offload the watcher would commit, by action."""
    import agent.turn_proactive_prune as prune
    done = []

    async def compress(source, entry, history, request, *, trigger=None):
        done.append(("compact", entry.session_id))

    def offload(live_agent, history, session_id):
        done.append(("offload", session_id))
        return history
    runner._run_manual_compression = compress
    monkeypatch.setattr(prune, "offload_cold_history", offload)
    return done


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["started", "unknown"])
@pytest.mark.parametrize("action", ["compact", "offload"])
async def test_a_turn_claimed_before_the_watcher_takes_the_gate_is_not_compacted_under(make_runner, monkeypatch,
                                                                                         action, status):
    """Claim first: the FIFO wins ``claim_gate``, commits the claim (its transcript mark) and releases
    the gate; the execution then waits in the multiplex ``_execution_scope`` before the runner installs
    its pending-agent sentinel. The watcher's gate is free and the runner reads idle, but a summary or
    offload written now lands after the mark and restart recovery reads it as the turn's own output
    (an ``unknown`` admission is the same turn after a crash). The owner's admission says the turn is
    live: the watcher writes nothing."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    authority = runner.session_authority
    db = authority.persistence
    live = authority.sessions["s1"] = LiveSession(_source(), "k1")
    runner._agent_cache["k1"][0]._prompt_cache_warmer.action = action
    done = _record_cache_edge_work(runner, monkeypatch)
    row, _ = db.admissions.admit(epoch=authority.epoch, principal_id="u1", session_id="s1", request_id="r1",
                                 payload={"text": "hi"})
    async with live.claim_gate:  # the FIFO's claim, as ``_claim_next`` commits it
        claimed = await asyncio.to_thread(db.admissions.claim, epoch=authority.epoch, session_id="s1")
    assert claimed and claimed.admission_id == row.admission_id
    if status == "unknown":  # the process died with the claim started; the next boot fences it
        db.admissions.recover(epoch=db.runtime.begin_epoch(instance_id="t2"))
    assert [r.status for r in db.admissions.list(session_id="s1")] == [status]
    assert not runner._is_session_running("k1")  # no runner sentinel yet: the scan claims the edge

    await _compact_and_settle(runner)

    assert done == []
    assert not live.claim_gate.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["compact", "offload"])
async def test_an_authority_chat_whose_live_gate_is_absent_is_not_worked_ungated(make_runner, monkeypatch, action):
    """Under an authority the FIFO claim is serialized only by the live session's ``claim_gate``: with
    no live entry to hold, a claim could land mid-summary, so the job does nothing rather than run
    ungated."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    runner._agent_cache["k1"][0]._prompt_cache_warmer.action = action
    done = _record_cache_edge_work(runner, monkeypatch)

    await _compact_and_settle(runner)

    assert done == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["compact", "offload"])
async def test_an_idle_authority_chat_with_only_settled_or_queued_input_is_still_worked(make_runner, monkeypatch,
                                                                                         action):
    """The guard reads the owner's started/unknown rows only: a settled earlier turn, or input still
    queued behind the gate (it is claimed after the summary commits), leaves the edge work running."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    authority = runner.session_authority
    db = authority.persistence
    authority.sessions["s1"] = LiveSession(_source(), "k1")
    runner._agent_cache["k1"][0]._prompt_cache_warmer.action = action
    done = _record_cache_edge_work(runner, monkeypatch)
    db.admissions.admit(epoch=authority.epoch, principal_id="u1", session_id="s1", request_id="r0",
                        payload={"text": "earlier"})
    earlier = db.admissions.claim(epoch=authority.epoch, session_id="s1")
    db.admissions.settle(epoch=authority.epoch, admission_id=earlier.admission_id, generation=earlier.generation,
                         outcome="completed", result={"final_response": "ok", "messages": []})
    db.admissions.admit(epoch=authority.epoch, principal_id="u1", session_id="s1", request_id="r1",
                        payload={"text": "next"})

    await _compact_and_settle(runner)

    assert done == [(action, "s1")]


@pytest.mark.asyncio
async def test_a_slow_summary_neither_delays_another_chat_nor_holds_up_housekeeping(make_runner):
    """A summary may run for minutes: the next due chat starts at once (it would otherwise go cold
    before its turn) and ``_session_housekeeping`` returns to its other sweeps."""
    entries = {"slow": _entry("slow", "s-slow", _source("c1")), "fast": _entry("fast", "s-fast", _source("c2"))}
    runner = make_runner(entries, ["slow", "fast"])
    runner._sweep_idle_cached_agents = lambda: 0
    runner._sweep_agent_cache_under_pressure = lambda: None
    runner._last_session_store_prune_ts = float("inf")
    never = asyncio.Event()

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append(entry.session_id)
        if entry.session_id == "s-slow":
            await never.wait()
    runner._run_manual_compression = compress

    await asyncio.wait_for(runner._session_housekeeping(), timeout=5)
    await async_wait_until(lambda: sorted(runner.compacted) == ["s-fast", "s-slow"], timeout=5)
    # The slow chat's job is still in flight: the next pass never compacts it a second time.
    await asyncio.wait_for(runner._compact_idle_sessions_before_cache_expiry(), timeout=5)
    await async_wait_until(lambda: runner.compacted.count("s-fast") == 2, timeout=5)
    assert runner.compacted.count("s-slow") == 1
    never.set()


@pytest.mark.asyncio
async def test_a_detached_desktop_conversation_without_a_routing_entry_is_compacted(make_runner):
    """A detached conversation has a cached agent but no store entry: its live conversation names
    the entry, so the window the scan already claimed is not wasted."""
    key = "agent:main:detached:s-det"
    runner = make_runner({}, [key], authority=True)
    runner.session_authority.sessions["s-det"] = LiveSession(_source(detached_route=key), key)

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append((entry.session_key, entry.session_id))
    runner._run_manual_compression = compress

    await _compact_and_settle(runner)

    assert runner.compacted == [(key, "s-det")]


@pytest.mark.asyncio
async def test_a_job_queued_behind_a_summary_skips_a_key_that_new_rerouted(make_runner):
    """The claim and the warm window belong to the chat the job was queued for: when ``/new``
    re-routes the key while the job waits for a slot, the new chat is not compacted."""
    runner = make_runner({"k1": _entry("k1", "s-old", _source())}, ["k1"])

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append(entry.session_id)
    runner._run_manual_compression = compress
    runner.session_store.entries["k1"] = _entry("k1", "s-new", _source())  # /new before the slot

    await _compact_and_settle(runner)

    assert runner.compacted == []


@pytest.mark.asyncio
async def test_running_chat_is_not_claimed_by_the_scan(make_runner):
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"])
    claims = runner._agent_cache["k1"][0]._prompt_cache_warmer.claims
    runner._session_state("k1").turn.agent = object()
    assert runner._cache_expiry_due_keys() == []
    assert claims == []
    runner._session_state("k1").turn.agent = None
    assert len(runner._cache_expiry_due_keys()) == 1
    assert len(claims) == 1


@pytest.mark.asyncio
async def test_summary_slots_bound_parallel_work_without_dropping_queued_chats(make_runner):
    entries = {f"k{i}": _entry(f"k{i}", f"s{i}", _source(f"c{i}")) for i in range(4)}
    runner = make_runner(entries, list(entries))
    release = asyncio.Event()

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append(entry.session_id)
        await release.wait()
    runner._run_manual_compression = compress
    try:
        await runner._compact_idle_sessions_before_cache_expiry()
        await async_wait_until(lambda: len(runner.compacted) >= 3, timeout=5)
        # Give the fourth job an event-loop turn to reach the same semaphore.
        await asyncio.sleep(0)
        assert len(runner.compacted) == 3
        assert len(runner._cache_expiry_jobs) == 4
    finally:
        release.set()
        await asyncio.gather(*getattr(runner, "_background_tasks", ()), return_exceptions=True)
    assert sorted(runner.compacted) == ["s0", "s1", "s2", "s3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["already_running", "started_during_read", "short_history"])
async def test_claimed_job_rechecks_running_state_and_minimum_history(make_runner, state):
    entry = _entry("k1", "s1", _source())
    runner = make_runner({"k1": entry}, ["k1"])
    loaded = []

    async def load(session_id):
        loaded.append(session_id)
        if state == "started_during_read":
            runner._session_state("k1").turn.agent = object()
        return HISTORY[:2] if state == "short_history" else list(HISTORY)

    async def compress(*args, **kwargs):
        runner.compacted.append("unexpected")
    runner._async_session_store = SimpleNamespace(_store=runner.session_store, load_transcript=load)
    runner._run_manual_compression = compress
    if state == "already_running":
        runner._session_state("k1").turn.agent = object()
    agent = runner._agent_cache["k1"][0]
    await runner._compact_claimed_session("k1", entry, agent)
    assert runner.compacted == []
    assert loaded == ([] if state == "already_running" else ["s1"])


@pytest.mark.asyncio
async def test_cold_claim_offloads_instead_of_requesting_a_summary(make_runner, monkeypatch):
    import agent.turn_proactive_prune as prune

    entry = _entry("k1", "s1", _source())
    runner = make_runner({"k1": entry}, ["k1"])
    agent = runner._agent_cache["k1"][0]
    agent._prompt_cache_warmer.action = "offload"
    offloads = []

    def offload(live_agent, history, session_id):
        offloads.append((live_agent is agent, history == HISTORY, session_id))
        return history

    async def compress(*args, **kwargs):
        runner.compacted.append("unexpected")
    monkeypatch.setattr(prune, "offload_cold_history", offload)
    runner._run_manual_compression = compress
    await _compact_and_settle(runner)
    assert offloads == [(True, True, "s1")]
    assert runner.compacted == []


@pytest.mark.asyncio
async def test_profile_retirement_joins_a_cache_edge_job_holding_the_old_gate(make_runner):
    """A summary runs under the retiring authority's ``claim_gate``; a successor authority's claim
    takes a different gate. Retirement joins the authority's tracked work (``join_authority_work``),
    so the job is that work: the profile is released only after its write, or kept (missed join)."""
    from gateway.session_runtime_workers import mutation_tasks
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    authority = runner.session_authority
    authority.sessions["s1"] = LiveSession(_source(), "k1")
    release, order = asyncio.Event(), []

    async def compress(source, entry, history, request, *, trigger=None):
        order.append("summary started")
        await release.wait()
        order.append("summary committed")
    runner._run_manual_compression = compress

    await runner._compact_idle_sessions_before_cache_expiry()
    await async_wait_until(lambda: order == ["summary started"], timeout=5)
    assert len(mutation_tasks(authority)) == 1  # retirement's join waits for it
    release.set()
    await asyncio.gather(*getattr(runner, "_background_tasks", ()), return_exceptions=True)
    assert order == ["summary started", "summary committed"]
    assert mutation_tasks(authority) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before_the_job", "while_the_route_resolves"])
@pytest.mark.parametrize("action", ["compact", "offload"])
async def test_a_retiring_authority_starts_no_cache_edge_work(make_runner, monkeypatch, action, when):
    """Retirement sets ``retiring`` before it joins the authority's work: a job that has not
    registered as that work by then (it may still be resolving its route in a thread) writes nothing,
    since the join would not wait for it and a successor's claim could land under it."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    authority = runner.session_authority
    authority.sessions["s1"] = LiveSession(_source(), "k1")
    runner._agent_cache["k1"][0]._prompt_cache_warmer.action = action
    done = _record_cache_edge_work(runner, monkeypatch)
    if when == "before_the_job":
        authority.retiring = True
    else:
        resolve = runner._cache_expiry_route

        def route_then_retire(key):
            routed = resolve(key)
            authority.retiring = True
            return routed
        runner._cache_expiry_route = route_then_retire

    await _compact_and_settle(runner)

    assert done == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["compact", "offload"])
async def test_a_multiplex_key_with_no_serving_authority_is_not_worked_ungated(make_runner, monkeypatch, action):
    """Under a multiplex registry a key whose profile no authority serves (unserved, or retiring and
    filtered out of ``serving()``) has no claim gate to hold: the job does nothing rather than take
    the standalone, ungated path."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"])
    runner.session_authorities = SimpleNamespace(serving=list, launch_key="launch")
    runner._agent_cache["k1"][0]._prompt_cache_warmer.action = action
    done = _record_cache_edge_work(runner, monkeypatch)

    await _compact_and_settle(runner)

    assert done == []


@pytest.mark.asyncio
async def test_a_refused_re_serve_is_not_recorded_as_served(tmp_path, monkeypatch):
    """A hot re-serve refused because retirement still holds ownership (a cache-edge summary is
    still running) must not record the profile as served: otherwise no adapter starts, the serve
    verb reports success, and later scans and serves see it as "already served" (Astra, r32 p13)."""
    from pathlib import Path
    from gateway import run
    from gateway.run import GatewayRunner
    from gateway.runtime_ownership import process_ownership

    monkeypatch.setattr(run, "_load_gateway_config", lambda *a: {
        "platform_toolsets": {"cli": [], "tui": []},
        "model": {"provider": "custom", "base_url": "http://127.0.0.1:9/v1", "api_key": "test-key"}})
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    beta = root / "profiles" / "beta"
    beta.mkdir(parents=True, mode=0o700)
    (beta / "config.yaml").write_text("{}")
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = (("default", root),)
    runner = GatewayRunner(config)
    process_ownership.reserve([root])
    try:
        started = []

        async def refuse(name, home):
            return f"profile '{name}' ownership retained: retirement is still joining its work"

        async def start_adapters(name, home, claimed):
            started.append(name)
            return 1

        async def nothing(*_a, **_k):
            return None

        monkeypatch.setattr(runner, "_serve_profile_runtime", refuse)
        monkeypatch.setattr(runner, "_start_one_profile_adapters", start_adapters)
        monkeypatch.setattr(runner, "_after_profiles_added", nothing)
        known = dict(runner._served_profile_homes or {})
        known["beta"] = beta.resolve()  # what the serve control verb passes
        result = await runner._apply_profile_changes(known, ["beta"], [], [], reason="control-socket")

        assert started == []
        assert result["parked"] == ["beta"]
        assert "beta" not in result["served_profiles"]
        assert "beta" not in (runner._served_profile_homes or {})
    finally:
        process_ownership.release(root)
