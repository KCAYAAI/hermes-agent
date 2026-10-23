"""ReactionLedger: request reactions always converge to the truth.

Invariants checked everywhere (after in-flight calls settle and the ledger has retried):
  * a message with a live turn or held background work shows 👀 only
  * a finished success shows ✅ only, a failure ❌ only, a cancelled turn nothing
  * nothing the ledger touched is left behind in a wrong state with no pending retry
"""

import asyncio
import json
import random

import pytest

from plugins.platforms.slack import reaction_ledger as rl
from plugins.platforms.slack.reaction_ledger import CHECK, CROSS, EYES, FAIL, GONE, OK, ReactionLedger


class Clock:
    def __init__(self, now=2_000_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeSlack:
    """Slack's reaction state plus scripted failures. ``fail`` holds per-call outcomes."""

    def __init__(self, rng=None, fail_rate=0.0, land_on_fail=0.5):
        self.state = {}               # (team, ch, ts) -> set
        self.holds = {}               # (team, ch, thread) -> "hold" | "failed"
        self.live = set()
        self.rng = rng or random.Random(0)
        self.fail_rate = fail_rate
        self.land_on_fail = land_on_fail
        self.calls = 0
        self.reads = 0
        self.script = []              # forced outcomes for next calls: OK | FAIL | "raise"
        self.gate = None              # asyncio.Event to park the next call on
        self.on_call = None           # callback run mid-call (between send and response)

    def _next(self):
        if self.script:
            return self.script.pop(0)
        return FAIL if self.rng.random() < self.fail_rate else OK

    async def react(self, team, ch, ts, name, remove):
        self.calls += 1
        outcome = self._next()
        if self.gate is not None:
            gate, self.gate = self.gate, None
            await gate.wait()
        if self.on_call is not None:
            cb, self.on_call = self.on_call, None
            cb()
        have = self.state.setdefault((team, ch, ts), set())
        landed = outcome == OK or (outcome in (FAIL, "raise") and self.rng.random() < self.land_on_fail)
        if landed:
            (have.discard if remove else have.add)(name)
        if outcome == "raise":
            raise asyncio.CancelledError()
        return OK if outcome == OK else FAIL

    async def read(self, team, ch, ts):
        self.reads += 1
        if self.fail_rate and self.rng.random() < self.fail_rate:
            return None
        return set(self.state.get((team, ch, ts), set()))

    def hold_state(self, entry):
        return self.holds.get((entry.team, entry.channel, entry.thread))

    def is_live(self, team, ch, ts):
        return (team, ch, ts) in self.live

    async def recent_own(self):
        return [(t, c, s, set(v)) for (t, c, s), v in self.state.items()]


def expected(outcome, held, active):
    if active:
        return {EYES}
    if outcome == "failure":
        return {CROSS}
    if outcome == "cancelled":
        return set()
    return {EYES} if held == "hold" else ({CROSS} if held == "failed" else {CHECK})


@pytest.fixture
def world(tmp_path):
    clock = Clock()
    slack = FakeSlack()
    ledger = ReactionLedger(slack, tmp_path / "ledger.json", clock=clock)
    return ledger, slack, clock, tmp_path / "ledger.json"


async def settle_all(ledger, clock, rounds=12):
    for _ in range(rounds):
        clock.now += 1000
        await ledger.run_due(force=True)


K = ("T1", "C1", "100.0")


@pytest.mark.asyncio
async def test_normal_turn_costs_two_calls_and_no_reads(world):
    ledger, slack, clock, _ = world
    fresh = ("T1", "C1", str(clock.now - 2))
    await ledger.start(*fresh, fresh[2])
    await ledger.complete(*fresh, fresh[2], "success")
    assert slack.state[fresh] == {CHECK}
    assert slack.calls == 3 and slack.reads == 0  # 👀 add, ✅ add, 👀 remove
    assert ledger.entries() == {}


@pytest.mark.asyncio
async def test_final_is_added_before_eyes_is_removed(world):
    ledger, slack, clock, _ = world
    order = []
    real = slack.react

    async def spy(team, ch, ts, name, remove):
        order.append(("-" if remove else "+") + name)
        return await real(team, ch, ts, name, remove)

    slack.react = spy
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    assert order == ["+eyes", "+white_check_mark", "-eyes"]


@pytest.mark.asyncio
async def test_held_background_work_keeps_eyes_then_settles(world):
    ledger, slack, clock, _ = world
    slack.holds[("T1", "C1", "100.0")] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    assert slack.state[K] == {EYES} and K in ledger.entries()
    slack.holds.pop(("T1", "C1", "100.0"))
    assert await ledger.settle("C1", "100.0", "T1") == 1
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_missed_settle_is_caught_by_the_loop_without_api_calls(world):
    """If the owner's settle never arrives, the loop sees the hold cleared (local check)."""
    ledger, slack, clock, _ = world
    slack.holds[("T1", "C1", "100.0")] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    calls = slack.calls
    await ledger.run_due()
    assert slack.calls == calls  # still held: no calls
    slack.holds.pop(("T1", "C1", "100.0"))
    await ledger.run_due()
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_refused_start_and_refused_final_still_converge(world):
    """13th review: both 👀 and ✅ refused. The finished message retries until ✅."""
    ledger, slack, clock, _ = world
    slack.land_on_fail = 0.0
    slack.script = [FAIL, FAIL]
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    assert K in ledger.entries()  # scheduled, not lost
    await settle_all(ledger, clock)
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_cancelled_mid_call_is_retried(world):
    """13th review: finalization interrupted mid-call. The entry stays and retries."""
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    slack.script = ["raise"]
    with pytest.raises(asyncio.CancelledError):
        await ledger.complete(*K, "100.0", "success")
    assert ledger.entries()[K].applied is None and ledger.entries()[K].next_at
    await settle_all(ledger, clock)
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_overlapping_turns_on_one_message_never_end_with_both(world):
    """13th review should-fix: two turns on the same message; one's 👀 refused."""
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    slack.land_on_fail = 0.0
    slack.script = [FAIL]
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    assert slack.state[K] == {EYES}  # the other turn is still working
    await ledger.complete(*K, "100.0", "success")
    await settle_all(ledger, clock)
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_work_started_during_a_call_is_reflected(world):
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    slack.on_call = lambda: slack.holds.__setitem__(("T1", "C1", "100.0"), "hold")
    await ledger.complete(*K, "100.0", "success")
    assert slack.state[K] == {EYES} and K in ledger.entries()


@pytest.mark.asyncio
async def test_cancelled_turn_leaves_no_reaction(world):
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "cancelled")
    assert slack.state[K] == set() and ledger.entries() == {}


@pytest.mark.asyncio
async def test_restart_resumes_from_disk_and_reads_slack_first(world, tmp_path):
    ledger, slack, clock, path = world
    slack.holds[("T1", "C1", "100.0")] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    ledger.closing = True  # process dies without drain (crash)
    saved = json.loads(path.read_text())
    assert len(saved["entries"]) == 1
    slack.holds.clear()  # background work died with the process
    fresh = ReactionLedger(slack, clock=clock)
    fresh.attach(path)
    reads = slack.reads
    await fresh.run_due()
    assert slack.reads == reads + 1
    assert slack.state[K] == {CROSS}  # held work died: ❌, never a false ✅
    assert fresh.entries() == {}


@pytest.mark.asyncio
async def test_turn_alive_at_crash_ends_x_after_restart(world):
    ledger, slack, clock, path = world
    await ledger.start(*K, "100.0")
    fresh = ReactionLedger(slack, clock=clock)
    fresh.attach(path)
    await fresh.run_due()
    assert slack.state[K] == {CROSS} and fresh.entries() == {}


@pytest.mark.asyncio
async def test_shutdown_closes_finished_and_marks_held_x(world):
    ledger, slack, clock, path = world
    k2 = ("T1", "C1", "200.0")
    slack.holds[("T1", "C1", "200.0")] = "hold"
    for k in (K, k2):
        await ledger.start(*k, k[2])
        await ledger.complete(*k, k[2], "success")
    await ledger.shutdown()
    assert slack.state[K] == {CHECK} and slack.state[k2] == {CROSS}
    assert json.loads(path.read_text())["entries"] == []


@pytest.mark.asyncio
async def test_hung_call_cannot_block_shutdown_and_is_kept_on_disk(world):
    ledger, slack, clock, path = world
    await ledger.start(*K, "100.0")
    slack.gate = asyncio.Event()  # never set
    task = asyncio.create_task(ledger.complete(*K, "100.0", "success"))
    await asyncio.sleep(0)
    await asyncio.wait_for(ledger.shutdown(timeout=0.2), 3.0)
    task.cancel()
    with pytest.raises(BaseException):
        await task
    assert json.loads(path.read_text())["entries"]


@pytest.mark.asyncio
async def test_deleted_message_is_dropped(world):
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")

    async def gone(*a):
        return GONE

    slack.react = gone
    await ledger.complete(*K, "100.0", "success")
    assert ledger.entries() == {}


@pytest.mark.asyncio
async def test_failure_time_only_marks_earlier_messages(world):
    ledger, slack, clock, _ = world
    thread = ("T1", "C1", "100.0")
    slack.holds[thread] = "hold"
    k2 = ("T1", "C1", "300.0")
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    await ledger.start(*k2, "100.0")
    await ledger.complete(*k2, "100.0", "success")
    slack.holds.clear()
    assert await ledger.settle("C1", "100.0", "T1", failed_at=200.0) == 2
    assert slack.state[K] == {CROSS} and slack.state[k2] == {CHECK}


@pytest.mark.asyncio
async def test_settle_is_workspace_scoped(world):
    ledger, slack, clock, _ = world
    k2 = ("T2", "C1", "100.0")
    slack.holds[("T1", "C1", "100.0")] = "hold"
    slack.holds[("T2", "C1", "100.0")] = "hold"
    for k in (K, k2):
        await ledger.start(*k, "100.0")
        await ledger.complete(*k, "100.0", "success")
    slack.holds.clear()
    assert await ledger.settle("C1", "100.0", "T2") == 1
    assert slack.state[k2] == {CHECK} and slack.state[K] == {EYES}


@pytest.mark.asyncio
async def test_sweep_clears_only_old_untracked_eyes(world):
    ledger, slack, clock, _ = world
    old = ("T1", "C9", str(clock.now - 10 * 3600))
    young = ("T1", "C9", str(clock.now - 600))
    tracked = ("T1", "C9", str(clock.now - 10 * 3600 + 1))
    live = ("T1", "C9", str(clock.now - 10 * 3600 + 2))
    for k in (old, young, tracked, live):
        slack.state[k] = {EYES}
    slack.holds[("T1", "C9", tracked[2])] = "hold"
    await ledger.start(*tracked, tracked[2])
    await ledger.complete(*tracked, tracked[2], "success")
    slack.live.add(live)  # marker only (no ledger start yet): swept; the start re-adds 👀
    assert await ledger.sweep() == 2
    assert slack.state[old] == slack.state[live] == set()
    assert slack.state[young] == slack.state[tracked] == {EYES}
    await ledger.start(*live, live[2])
    assert slack.state[live] == {EYES}


@pytest.mark.asyncio
async def test_overlapping_failure_then_success_ends_x(world):
    """14th review: the worst outcome of overlapping turns wins."""
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "failure")
    await ledger.complete(*K, "100.0", "success")
    assert slack.state[K] == {CROSS} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_sweep_racing_a_new_turn_leaves_eyes(world):
    """14th review: a turn starting during the sweep's removal still ends up showing 👀."""
    ledger, slack, clock, _ = world
    k = ("T1", "C9", str(clock.now - 10 * 3600))
    slack.state[k] = {EYES}
    started = []
    slack.on_call = lambda: started.append(asyncio.ensure_future(ledger.start(*k, k[2])))
    await ledger.sweep()
    await asyncio.gather(*started)
    assert slack.state[k] == {EYES} and ledger.entries()[k].active == 1


@pytest.mark.asyncio
async def test_long_outage_never_abandons_a_message(world):
    """14th review: after a multi-week outage, the message still converges."""
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    slack.fail_rate, slack.land_on_fail = 1.0, 0.0
    await ledger.complete(*K, "100.0", "success")
    for _ in range(24 * 21):
        clock.now += 3600
        await ledger.run_due()
    assert K in ledger.entries()
    slack.fail_rate = 0.0
    clock.now += 3600
    await ledger.run_due()
    assert slack.state[K] == {CHECK} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_lost_turn_is_closed_after_the_stale_window(world):
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")  # its complete never arrives
    clock.now += rl.STALE_ACTIVE_SECONDS + 1
    await ledger.run_due()
    assert slack.state[K] == {CROSS} and ledger.entries() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(300))
async def test_randomized_convergence(tmp_path, seed):
    """Random interleavings of turns, background work, Slack failures (landed or not),
    mid-call cancellations, and crash-restarts. After Slack recovers, every message shows
    exactly what its facts require and nothing is left pending."""
    rng = random.Random(seed)
    clock = Clock()
    slack = FakeSlack(rng=rng, fail_rate=rng.choice([0.0, 0.2, 0.5]), land_on_fail=rng.random())
    path = tmp_path / "l.json"
    ledger = ReactionLedger(slack, path, clock=clock)
    msgs = [("T%d" % rng.randint(1, 2), "C1", "%d.0" % (100 + i)) for i in range(rng.randint(1, 4))]
    thread = {m: (m[0], "C1", "100.0" if rng.random() < 0.5 else m[2]) for m in msgs}
    truth = {m: {"active": 0, "outcome": None} for m in msgs}
    pending = []
    # production: a turn is live while the adapter still holds its marker
    slack.is_live = lambda team, ch, ts: truth.get((team, ch, ts), {}).get("active", 0) > 0

    async def step():
        nonlocal ledger
        m = rng.choice(msgs)
        t = thread[m]
        op = rng.random()
        try:
            if op < 0.25:
                if truth[m]["active"] == 0:
                    truth[m]["outcome"] = None
                truth[m]["active"] += 1
                await ledger.start(*m, t[2])
            elif op < 0.5 and truth[m]["active"]:
                truth[m]["active"] -= 1
                out = rng.choice(["success", "success", "failure", "cancelled"])
                rank = {None: -1, "cancelled": 0, "success": 1, "failure": 2, "crashed": 3}
                if rank[out] > rank[truth[m]["outcome"]]:
                    truth[m]["outcome"] = out
                if rng.random() < 0.1:
                    slack.script.insert(0, "raise")
                try:
                    await ledger.complete(*m, t[2], out)
                finally:
                    if "raise" in slack.script:
                        slack.script.remove("raise")
            elif op < 0.65:
                slack.holds[t] = "hold"
            elif op < 0.8 and t in slack.holds:
                failed = rng.random() < 0.3
                slack.holds.pop(t, None)
                if failed:
                    slack.holds[t] = "failed"
                await ledger.settle(t[1], t[2], t[0])
            elif op < 0.85:
                clock.now += rng.choice([1, 60, 400, 30000])
                await ledger.run_due()
            elif op < 0.9:  # daily sweep, possibly racing a new turn on a swept message
                clock.now += rl.STALE_EYES_SECONDS + 1
                racer = rng.choice(msgs)
                if rng.random() < 0.5:
                    def _race(racer=racer):
                        truth[racer]["outcome"] = None if truth[racer]["active"] == 0 else truth[racer]["outcome"]
                        truth[racer]["active"] += 1
                        pending.append(asyncio.ensure_future(ledger.start(*racer, thread[racer][2])))
                    slack.on_call = _race
                await ledger.sweep()
                slack.on_call = None
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()
            else:  # crash + restart: live turns and held work die with the process
                ledger = ReactionLedger(slack, clock=clock)
                ledger.attach(path)
                for mm in msgs:
                    if truth[mm]["active"]:
                        truth[mm]["active"] = 0
                        truth[mm]["outcome"] = "crashed"
                for tt, v in list(slack.holds.items()):
                    if v == "hold":
                        slack.holds[tt] = "failed-by-crash"
        except asyncio.CancelledError:
            pass

    for _ in range(rng.randint(5, 40)):
        await step()
    # finish every live turn, end all background work, and let Slack recover
    for m in msgs:
        while truth[m]["active"]:
            truth[m]["active"] -= 1
            if truth[m]["outcome"] in (None, "cancelled"):
                truth[m]["outcome"] = "success"
            await ledger.complete(*m, thread[m][2], "success")
    for t in list(slack.holds):
        slack.holds[t] = "failed" if slack.holds[t] in ("failed", "failed-by-crash") else None
        if slack.holds[t] is None:
            slack.holds.pop(t)
        await ledger.settle(t[1], t[2], t[0])
    slack.fail_rate, slack.script = 0.0, []
    await settle_all(ledger, clock)

    assert ledger.entries() == {}, (seed, ledger.entries())
    for m in msgs:
        out = truth[m]["outcome"]
        got = slack.state.get(m, set())
        if out is None:
            continue  # never started
        if out == "crashed":
            assert got == {CROSS}, (seed, m, got)
        elif out == "cancelled":  # nothing, or X if work launched from its thread failed
            assert got in (set(), {CROSS}), (seed, m, got)
        elif out == "failure":
            assert got == {CROSS}, (seed, m, got)
        else:  # success: ✅ unless its thread's work failed (❌)
            assert got in ({CHECK}, {CROSS}), (seed, m, got)
            assert EYES not in got


@pytest.mark.asyncio
async def test_reprocessed_message_replaces_its_old_final(world):
    ledger, slack, clock, path = world
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    fresh = ReactionLedger(slack, clock=clock)  # survives a restart too
    fresh.attach(path)
    await fresh.start(*K, "100.0")
    await fresh.complete(*K, "100.0", "failure")
    assert slack.state[K] == {CROSS}


@pytest.mark.asyncio
async def test_old_message_reprocessed_after_memory_expired_reads_slack_once(world):
    ledger, slack, clock, _ = world
    slack.state[K] = {CHECK}  # left from a turn the ledger no longer remembers
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "failure")
    assert slack.state[K] == {CROSS} and slack.reads == 1


@pytest.mark.asyncio
async def test_refused_sweep_removal_is_adopted_and_retried(world):
    """15th review: an orphan 👀 whose removal is refused must not wait for the next sweep."""
    ledger, slack, clock, path = world
    k = ("T1", "C9", str(clock.now - 10 * 3600))
    slack.state[k] = {EYES}
    slack.land_on_fail = 0.0
    slack.script = [FAIL, FAIL]  # the sweep's removal and its immediate retry are both refused
    assert await ledger.sweep() == 0
    assert k in ledger.entries() and json.loads(path.read_text())["entries"]
    await settle_all(ledger, clock)
    assert slack.state[k] == set() and ledger.entries() == {}


@pytest.mark.asyncio
async def test_leaked_marker_cannot_block_the_sweep(world):
    """15th review should-fix: a marker with no processing lifecycle never pins 👀."""
    ledger, slack, clock, _ = world
    k = ("T1", "C9", str(clock.now - 10 * 3600))
    slack.state[k] = {EYES}
    slack.live.add(k)
    assert await ledger.sweep() == 1 and slack.state[k] == set()


@pytest.mark.asyncio
async def test_shutdown_is_bounded_even_if_the_loop_ignores_cancel(world):
    ledger, slack, clock, _ = world

    release = asyncio.Event()

    async def stubborn():
        while not release.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue

    task = ledger._task = asyncio.create_task(stubborn())
    await asyncio.sleep(0)
    await asyncio.wait_for(ledger.shutdown(timeout=0.3), 3.0)
    release.set()
    task.cancel()
    await asyncio.wait([task], timeout=1)


@pytest.mark.asyncio
async def test_revise_to_failure_after_close_replaces_check(world):
    """16th review: a completion cancelled mid-✅ and re-reported as FAILURE ends ❌."""
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    slack.script = [OK, "raise"]  # ✅ lands, then the call is cancelled
    slack.land_on_fail = 1.0
    with pytest.raises(asyncio.CancelledError):
        await ledger.complete(*K, "100.0", "success")
    await ledger.revise(*K, "100.0", "failure")
    await settle_all(ledger, clock)
    assert slack.state[K] == {CROSS} and ledger.entries() == {}


@pytest.mark.asyncio
async def test_revise_after_fully_closed_success(world):
    ledger, slack, clock, _ = world
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    await ledger.revise(*K, "100.0", "cancelled")
    assert slack.state[K] == {CHECK}
    await ledger.revise(*K, "100.0", "failure")
    assert slack.state[K] == {CROSS} and ledger.entries() == {}



@pytest.mark.asyncio
@pytest.mark.parametrize("child,final", [(None, set()), ("failed", {CROSS})])
async def test_cancelled_turn_keeps_eyes_for_running_child(world, child, final):
    """17th review: a cancelled parent with live background work keeps eyes, then ends by the child."""
    ledger, slack, clock, _ = world
    thread = ("T1", "C1", "100.0")
    slack.holds[thread] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "cancelled")
    assert slack.state[K] == {EYES} and K in ledger.entries()
    slack.holds.pop(thread)
    if child:
        slack.holds[thread] = child
    await ledger.settle("C1", "100.0", "T1")
    assert slack.state[K] == final and ledger.entries() == {}



@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancelled"])
async def test_crash_during_completion_read_keeps_held_child(world, outcome):
    """18th review: refused start 👀, crash while completion awaits a Slack read: the saved entry
    must already say it was waiting on background work, so restart ends it ❌."""
    ledger, slack, clock, path = world
    thread = ("T1", "C1", "100.0")
    slack.holds[thread] = "hold"
    slack.land_on_fail, slack.script = 0.0, [FAIL]
    await ledger.start(*K, "100.0")  # 👀 refused -> applied unknown

    async def crash_read(*a):
        raise asyncio.CancelledError()

    slack.read = crash_read
    with pytest.raises(asyncio.CancelledError):
        await ledger.complete(*K, "100.0", outcome)
    saved = json.loads(path.read_text())["entries"][0]
    assert saved["held"] is True
    del slack.read  # restore FakeSlack.read
    slack.holds.clear()  # the child died with the process
    fresh = ReactionLedger(slack, clock=clock)
    fresh.attach(path)
    await settle_all(fresh, clock)
    assert slack.state.get(K, set()) == {CROSS} and fresh.entries() == {}



@pytest.mark.asyncio
async def test_crash_during_settle_after_child_succeeded_ends_check(world):
    """19th review: the child finished OK; a crash during the settle's ✅ call must not end ❌."""
    ledger, slack, clock, path = world
    thread = ("T1", "C1", "100.0")
    slack.holds[thread] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    slack.holds.clear()  # child succeeded
    slack.script = ["raise"]
    slack.land_on_fail = 1.0
    with pytest.raises(asyncio.CancelledError):
        await ledger.settle("C1", "100.0", "T1")
    assert json.loads(path.read_text())["entries"][0]["held"] is False
    fresh = ReactionLedger(slack, clock=clock)
    fresh.attach(path)
    await settle_all(fresh, clock)
    assert slack.state[K] == {CHECK} and fresh.entries() == {}


@pytest.mark.asyncio
async def test_failed_child_is_saved_before_the_settle_call(world):
    ledger, slack, clock, path = world
    thread = ("T1", "C1", "100.0")
    slack.holds[thread] = "hold"
    await ledger.start(*K, "100.0")
    await ledger.complete(*K, "100.0", "success")
    slack.holds[thread] = "failed"
    slack.script = ["raise"]
    with pytest.raises(asyncio.CancelledError):
        await ledger.settle("C1", "100.0", "T1")
    assert json.loads(path.read_text())["entries"][0]["bad"] is True
    fresh = ReactionLedger(slack, clock=clock)
    fresh.attach(path)
    slack.holds.clear()
    await settle_all(fresh, clock)
    assert slack.state[K] == {CROSS} and fresh.entries() == {}



@pytest.mark.asyncio
async def test_idle_loop_waits_for_the_sweep_horizon_not_the_tick(world):
    ledger, _slack, clock, _ = world
    ledger.begin()
    try:
        assert ledger._loop_delay() == pytest.approx(rl.SWEEP_FIRST_DELAY_SECONDS)
        ledger._last_sweep = clock.now
        assert ledger._loop_delay() == pytest.approx(rl.IDLE_CAP_SECONDS)  # wall-clock deadlines: capped
        ledger._finals[K] = ([CHECK], clock.now - rl.FINALS_MEMORY_SECONDS + 500)
        assert ledger._loop_delay() == pytest.approx(500)
        ledger._entries[K] = rl.Entry(*K, thread="100.0", active=1)
        assert ledger._loop_delay() == rl.LOOP_TICK_SECONDS  # entries: the tick is the cap
        ledger._entries[K].next_at = clock.now + 10
        assert ledger._loop_delay() == pytest.approx(10)
        ledger._entries[K].next_at = clock.now - 10
        assert ledger._loop_delay() == 1.0
    finally:
        ledger._entries.clear()
        await ledger.shutdown()


@pytest.mark.asyncio
async def test_new_work_wakes_the_idle_loop(world):
    ledger, _slack, _clock, _ = world
    ledger._wake = asyncio.Event()
    await ledger.start(*K, "100.0")
    assert ledger._wake.is_set()
    ledger._wake.clear()
    await ledger.complete(*K, "100.0", "success")
    assert ledger._wake.is_set()
    ledger._wake.clear()
    await ledger.revise(*K, "100.0", "failure")
    assert ledger._wake.is_set()


