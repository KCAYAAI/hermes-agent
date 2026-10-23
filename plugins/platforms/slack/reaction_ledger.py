"""Durable ledger that drives the reactions on a Slack request to its true final state.

One component owns every status reaction the bot puts on a request message (👀 while working,
✅ done, ❌ failed). It replaces ad-hoc bookkeeping with three rules:

1. **Desired state is computed, never remembered.** Each pass derives what the message should
   show from live facts: turns still processing it, the turn outcome, and whether background
   work launched from its thread is still running (``hold_state``). Work that starts or ends
   mid-call is picked up on the next pass.
2. **Slack is the source of truth after any doubt.** The normal path trusts confirmed calls and
   costs the same two calls as before. After a failed, interrupted or ambiguous call the entry's
   reactions are marked unknown, and the next pass reads the message's real reactions first.
3. **Nothing is lost.** Every message that still needs a change is written to a small JSON file
   and removed only once Slack matches. A retry loop (backoff 1/2/5/15 min) and a restart both
   resume it. A daily sweep of the bot's own recent reactions clears any 👀 the ledger does not
   know about.

Adds always go before removals, so 👀 is only removed once the final reaction is on the message.
All changes to one message run under one lock. No model tokens are used; steady-state API cost is
unchanged, and recovery costs one read per affected message.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

EYES, CHECK, CROSS = "eyes", "white_check_mark", "x"
MANAGED = frozenset({EYES, CHECK, CROSS})
OK, FAIL, GONE = "ok", "fail", "gone"

BACKOFF_SECONDS = (60.0, 120.0, 300.0, 900.0, 3600.0)  # last step repeats until Slack accepts
PASS_BUDGET = 8                      # Slack calls per reconcile pass before backing off
STALE_ACTIVE_SECONDS = 12 * 3600     # a turn "processing" this long was lost: end it ❌
SWEEP_EVERY_SECONDS = 24 * 3600
SWEEP_FIRST_DELAY_SECONDS = 300.0
SWEEP_WINDOW_SECONDS = 24 * 3600
STALE_EYES_SECONDS = 6 * 3600        # sweep only clears 👀 older than this
FRESH_MESSAGE_SECONDS = 600          # newer requests are assumed to have no reactions yet
FINALS_MEMORY_SECONDS = 24 * 3600    # remember closed messages' reactions for re-processing
LOOP_TICK_SECONDS = 60.0             # loop cap while messages are tracked
IDLE_CAP_SECONDS = 900.0             # idle cap: deadlines are wall-clock, so re-check at least every 15 min

Key = Tuple[str, str, str]  # (team, channel, ts)
_RANK = {"": -1, "cancelled": 0, "success": 1, "failure": 2}  # overlapping turns: worst wins


@dataclass
class Entry:
    team: str
    channel: str
    ts: str
    thread: str
    active: int = 0                  # turns currently processing this message
    outcome: str = ""                # "" | success | failure | cancelled
    bad: bool = False                # work launched from the thread failed
    held: bool = False               # last settled on 👀 waiting for background work
    applied: Optional[List[str]] = None  # our reactions as last confirmed; None = read Slack
    attempts: int = 0
    next_at: float = 0.0             # 0 = no retry scheduled
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)

    @property
    def key(self) -> Key:
        return (self.team, self.channel, self.ts)


class ReactionLedger:
    """Owns request reactions for one Slack bot. ``transport`` provides:

    * ``async react(team, channel, ts, name, remove) -> OK | FAIL | GONE`` (benign
      ``already_reacted``/``no_reaction`` count as OK)
    * ``async read(team, channel, ts) -> set | None | GONE`` (the bot's own reactions)
    * ``hold_state(entry) -> "hold" | "failed" | None``
    * ``async recent_own() -> list[(team, channel, ts, names)] | None`` (newest first)
    * ``is_live(team, channel, ts) -> bool``
    """

    def __init__(self, transport: Any, path: Optional[Path] = None, clock=time.time):
        self._t = transport
        self._path = path
        self._clock = clock
        self._entries: Dict[Key, Entry] = {}
        # Closed messages -> (our reactions, closed at). A message processed again starts from
        # what it really shows, so an old ✅ can't sit next to a new ❌.
        self._finals: Dict[Key, Tuple[List[str], float]] = {}
        self._locks: Dict[Key, asyncio.Lock] = {}
        self._task: Optional[asyncio.Task] = None
        self._wake: Optional[asyncio.Event] = None
        self._last_sweep = 0.0
        self.closing = False

    # ----- lifecycle -----

    def attach(self, path: Optional[Path]) -> None:
        """Bind the durable file and load what a previous process left behind."""
        self._path = path
        self._load()

    def begin(self) -> None:
        """Start the retry/sweep loop on the running event loop."""
        self.closing = False
        if self._task is None or self._task.done():
            self._wake = asyncio.Event()
            self._last_sweep = self._clock() - SWEEP_EVERY_SECONDS + SWEEP_FIRST_DELAY_SECONDS
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Close what can be closed (held work dies with us: ❌), bounded; the rest stays on
        disk for the next process."""
        self.closing = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=min(2.0, timeout))
            self._task = None

        async def _drain() -> None:
            for key, entry in list(self._entries.items()):
                if entry.active == 0:
                    with contextlib.suppress(Exception):
                        await self.reconcile(key)

        with contextlib.suppress(BaseException):
            await asyncio.wait_for(_drain(), timeout=timeout)
        self._persist()

    # ----- events -----

    async def start(self, team: str, channel: str, ts: str, thread: str) -> None:
        """A turn started processing this message: it should show 👀."""
        key = (str(team or ""), str(channel), str(ts))
        entry = self._entries.get(key)
        if entry is None:
            now = self._clock()
            prior = self._finals.pop(key, None)
            if prior is not None:
                applied: Optional[List[str]] = list(prior[0])
            elif now - _ts(ts) > FRESH_MESSAGE_SECONDS:
                applied = None  # an older message we no longer remember: read Slack once
            else:
                applied = []    # a new request: nothing on it yet (no read needed)
            entry = self._entries[key] = Entry(*key, thread=str(thread or ts),
                                               applied=applied, created=now, touched=now)
        if entry.active == 0:
            entry.outcome = ""  # a new round of processing; overlapping turns keep the worst result
        entry.active += 1
        entry.touched = self._clock()
        self._persist()
        self._kick()
        await self.reconcile(key)

    async def complete(self, team: str, channel: str, ts: str, thread: str, outcome: str) -> None:
        """A turn on this message ended with ``outcome`` (success/failure/cancelled)."""
        key = (str(team or ""), str(channel), str(ts))
        entry = self._entries.get(key)
        if entry is None:  # start was never seen: Slack's real state is unknown
            now = self._clock()
            self._finals.pop(key, None)
            entry = self._entries[key] = Entry(*key, thread=str(thread or ts), applied=None, created=now, touched=now)
        entry.active = max(0, entry.active - 1)
        if _RANK.get(outcome, 0) > _RANK.get(entry.outcome, -1):
            entry.outcome = outcome
        entry.touched = self._clock()
        self._note_hold(entry)
        self._persist()
        self._kick()
        await self.reconcile(key)

    async def revise(self, team: str, channel: str, ts: str, thread: str, outcome: str) -> None:
        """A turn that already completed reports a worse outcome (e.g. its completion was
        cancelled mid-call and the gateway re-reports FAILURE). Never changes the turn count."""
        key = (str(team or ""), str(channel), str(ts))
        entry = self._entries.get(key)
        if entry is None:
            if outcome != "failure":
                return  # nothing worse than what already closed
            now = self._clock()
            prior = self._finals.pop(key, None)
            entry = self._entries[key] = Entry(*key, thread=str(thread or ts),
                                               applied=list(prior[0]) if prior else None,
                                               created=now, touched=now)
        if _RANK.get(outcome, 0) > _RANK.get(entry.outcome, -1):
            entry.outcome = outcome
        entry.touched = self._clock()
        self._note_hold(entry)
        self._persist()
        self._kick()
        await self.reconcile(key)

    async def settle(self, channel: str, thread: str, team: Optional[str] = None, *,
                     failed: bool = False, failed_at: Optional[float] = None) -> int:
        """Background work in a thread settled. Returns how many messages reached a final state.
        ``team`` empty/None matches any workspace."""
        keys = [k for k, e in list(self._entries.items())
                if e.channel == str(channel) and e.thread == str(thread)
                and (not team or not e.team or e.team == str(team))]
        closed = 0
        for key in keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if failed or (failed_at is not None and failed_at >= _ts(entry.ts)):
                entry.bad = True
                self._persist()
            if await self.reconcile(key):
                closed += 1
        return closed

    # ----- core -----

    def desired(self, entry: Entry) -> Set[str]:
        if entry.active > 0:
            return {EYES}
        if entry.outcome == "failure":
            return {CROSS}
        if entry.outcome not in ("success", "cancelled"):
            return set()
        # Background work launched from the thread outlives its turn, even a cancelled one.
        state = self._t.hold_state(entry)
        if state == "hold":
            return {CROSS} if self.closing else {EYES}
        if state == "failed" or entry.bad:
            return {CROSS}
        return {CHECK} if entry.outcome == "success" else set()

    async def reconcile(self, key: Key) -> bool:
        """Drive one message to its desired reactions. True when it reached a final state and
        left the ledger; False while it shows 👀 for live/held work or is scheduled to retry."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                return await self._reconcile_locked(key)
            except BaseException:
                entry = self._entries.get(key)
                if entry is not None:  # interrupted mid-call: Slack state unknown, retry later
                    entry.applied = None
                    self._schedule(entry)
                raise

    async def _reconcile_locked(self, key: Key) -> bool:
        for _ in range(PASS_BUDGET):
            entry = self._entries.get(key)
            if entry is None:
                return True
            before = (entry.held, entry.bad)
            self._note_hold(entry)
            if (entry.held, entry.bad) != before:
                self._persist()
            if entry.applied is None:
                truth = await self._t.read(entry.team, entry.channel, entry.ts)
                if truth == GONE:
                    return self._drop(key, "message no longer exists")
                if truth is None:
                    return self._schedule(entry)
                entry.applied = sorted(set(truth) & MANAGED)
            want = self.desired(entry)
            have = set(entry.applied)
            add, rem = sorted(want - have), sorted(have - want)
            if not add and not rem:
                if want == {EYES}:
                    entry.held = entry.active == 0
                    entry.attempts, entry.next_at = 0, 0.0
                    self._persist()
                    return False
                return self._drop(key, None, final=sorted(have))
            name, remove = (add[0], False) if add else (rem[0], True)
            result = await self._t.react(entry.team, entry.channel, entry.ts, name, remove)
            if result == GONE:
                return self._drop(key, "message no longer exists")
            if result != OK:
                entry.applied = None  # a failed or timed-out call may still have landed
                return self._schedule(entry)
            (have.discard if remove else have.add)(name)
            entry.applied = sorted(have)
        entry = self._entries.get(key)
        return self._schedule(entry) if entry is not None else True

    def _kick(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def _note_hold(self, entry: Entry) -> None:
        """Record, before any Slack call, whether finished turns are waiting on background work,
        so a crash at any await restores it as held (and ends it ❌ if that work died)."""
        if entry.active == 0 and entry.outcome in ("success", "cancelled"):
            with contextlib.suppress(Exception):
                state = self._t.hold_state(entry)
                # Mirror the live fact, not a sticky flag: once the work has finished, a crash
                # must not turn its result into ❌; a failure is recorded as bad before any call.
                if state == "failed":
                    entry.bad = True
                entry.held = state == "hold"

    def _schedule(self, entry: Entry) -> bool:
        entry.attempts += 1
        if entry.attempts in (5, 50):
            logger.warning("[Slack] reaction ledger: %s/%s still refused after %d attempts; retrying hourly",
                           entry.channel, entry.ts, entry.attempts)
        entry.next_at = self._clock() + BACKOFF_SECONDS[min(entry.attempts, len(BACKOFF_SECONDS)) - 1]
        self._persist()
        if self._wake is not None:
            self._wake.set()
        return False

    def _drop(self, key: Key, why: Optional[str], final: Optional[List[str]] = None) -> bool:
        entry = self._entries.pop(key, None)
        if final is not None:
            self._finals[key] = (final, self._clock())
        if why and entry is not None:
            logger.info("[Slack] reaction ledger: dropped %s/%s (%s)", entry.channel, entry.ts, why)
        self._persist()
        return True

    # ----- background loop -----

    async def run_due(self, *, force: bool = False) -> None:
        """One loop iteration: retries that are due, held messages whose work settled (a local
        check, no API call), lost turns, expiry. ``force`` retries everything now (tests)."""
        now = self._clock()
        for key in [k for k, (_, at) in self._finals.items() if now - at > FINALS_MEMORY_SECONDS]:
            self._finals.pop(key, None)
        for key, entry in list(self._entries.items()):
            if entry.active > 0 and now - entry.touched > STALE_ACTIVE_SECONDS \
                    and not self._t.is_live(*key):
                entry.active, entry.outcome = 0, "failure"
                entry.next_at = now
            due = entry.next_at and (force or entry.next_at <= now)
            settled = entry.held and entry.active == 0 and self.desired(entry) != {EYES}
            if due or settled or (force and entry.applied is None):
                with contextlib.suppress(Exception):
                    await self.reconcile(key)
        for key in [k for k, lock in self._locks.items()
                    if k not in self._entries and not lock.locked() and not getattr(lock, "_waiters", None)]:
            self._locks.pop(key, None)

    async def sweep(self) -> int:
        """Clear the bot's own 👀 left on messages nothing is tracking (older processes, crashes
        before the ledger existed). Returns how many were cleared."""
        self._last_sweep = self._clock()
        items = await self._t.recent_own()
        if not items:
            return 0
        now, cleared = self._clock(), 0
        for team, channel, ts, names in items:
            if EYES not in names or now - _ts(ts) < STALE_EYES_SECONDS:
                continue
            key = (team, channel, ts)
            async with self._locks.setdefault(key, asyncio.Lock()):
                if self._tracked(channel, ts):
                    continue
                result = await self._t.react(team, channel, ts, EYES, True)
                if result == OK:
                    cleared += 1
                elif result != GONE and not self._tracked(channel, ts):
                    # Adopt the orphan: desired state is "nothing", retried with backoff and
                    # persisted, so a refused removal is never left to next day's sweep window.
                    now = self._clock()
                    orphan = self._entries[key] = Entry(team, channel, ts, thread=ts, outcome="cancelled",
                                                        applied=None, created=now, touched=now)
                    self._schedule(orphan)
            # a turn may have started during the call: re-read Slack for it
            for k in ((team, channel, ts), ("", channel, ts)):
                entry = self._entries.get(k)
                if entry is not None:
                    entry.applied = None
                    await self.reconcile(k)
        if cleared:
            logger.info("[Slack] reaction ledger: sweep cleared %d stale 👀", cleared)
        return cleared

    def _tracked(self, channel: str, ts: str) -> bool:
        return any(e.channel == channel and e.ts == ts for e in self._entries.values())

    def active_turns(self, team: str, channel: str, ts: str) -> int:
        entry = self._entries.get((str(team or ""), str(channel), str(ts)))
        return entry.active if entry is not None else 0

    async def _run(self) -> None:
        while not self.closing:
            with contextlib.suppress(Exception):
                await self.run_due()
            if self._clock() - self._last_sweep >= SWEEP_EVERY_SECONDS:
                with contextlib.suppress(Exception):
                    await self.sweep()
            assert self._wake is not None
            self._wake.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._loop_delay())

    def _loop_delay(self) -> float:
        """How long the loop may sleep: until the next retry (capped at LOOP_TICK_SECONDS while
        messages are tracked, so held/lost turns are re-checked), else until the next sweep or
        the earliest finals expiry, capped at IDLE_CAP_SECONDS because those deadlines are wall-clock
        while the wait is monotonic. New work wakes it through ``_wake``."""
        now = self._clock()
        horizon = self._last_sweep + SWEEP_EVERY_SECONDS - now
        if self._finals:
            horizon = min(horizon, min(at for _, at in self._finals.values()) + FINALS_MEMORY_SECONDS - now)
        if not self._entries:
            return max(1.0, min(IDLE_CAP_SECONDS, horizon))
        delay = min(LOOP_TICK_SECONDS, horizon)
        upcoming = [e.next_at for e in self._entries.values() if e.next_at]
        if upcoming:
            delay = min(delay, min(upcoming) - now)
        return max(1.0, delay)

    # ----- persistence -----

    def _persist(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "version": 1,
                "entries": [asdict(e) for e in self._entries.values()],
                "finals": [[*k, names, at] for k, (names, at) in self._finals.items()],
            }))
            os.replace(tmp, self._path)
        except Exception:
            logger.warning("[Slack] reaction ledger: could not save %s", self._path, exc_info=True)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except Exception:
            logger.warning("[Slack] reaction ledger: unreadable %s; starting empty", self._path)
            return
        now = self._clock()
        for raw in data.get("finals") or []:
            with contextlib.suppress(Exception):
                team, channel, ts, names, at = raw
                if now - float(at) <= FINALS_MEMORY_SECONDS:
                    self._finals[(str(team), str(channel), str(ts))] = (list(names), float(at))
        for raw in data.get("entries") or []:
            with contextlib.suppress(Exception):
                entry = Entry(**{k: raw[k] for k in Entry.__dataclass_fields__ if k in raw})
                if entry.active > 0:           # its turn died with the previous process
                    entry.active, entry.outcome = 0, "failure"
                if entry.held:                 # its background work died too
                    entry.bad, entry.held = True, False
                entry.applied, entry.next_at = None, now  # truth unknown: read Slack first
                self._entries[entry.key] = entry
        if self._entries:
            logger.info("[Slack] reaction ledger: resuming %d message(s) from %s", len(self._entries), self._path)

    # ----- introspection (tests/diagnostics) -----

    def entries(self) -> Dict[Key, Entry]:
        return dict(self._entries)


def _ts(ts: Any) -> float:
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def outcome_name(outcome: Any) -> str:
    """Map a ProcessingOutcome (or its name) to the ledger's outcome string."""
    name = str(getattr(outcome, "name", outcome) or "").lower()
    return name if name in {"success", "failure", "cancelled"} else "cancelled"


def slack_error_code(exc: BaseException) -> str:
    code = ""
    with contextlib.suppress(Exception):
        code = str(getattr(exc, "response", {}).get("error", "") or "")
    return code or str(exc)


GONE_CODES = ("message_not_found", "channel_not_found", "is_archived", "account_inactive",
              "not_in_channel", "thread_not_found")


def classify(exc: BaseException, remove: bool) -> str:
    code = slack_error_code(exc)
    if ("no_reaction" in code) if remove else ("already_reacted" in code):
        return OK
    if any(g in code for g in GONE_CODES):
        return GONE
    return FAIL


def own_names(message: Any, uid: str) -> Set[str]:
    names: Set[str] = set()
    for r in (message or {}).get("reactions") or []:
        if uid and uid in (r.get("users") or []):
            names.add(str(r.get("name")))
    return names


def items_within(pages: Iterable[Any], window: float, now: float) -> List[Tuple[str, str, Set[str]]]:
    """Flatten reactions.list pages to (channel, ts, raw message) within ``window``."""
    out = []
    for page in pages:
        for item in (page or {}).get("items") or []:
            msg = item.get("message") or {}
            ts = str(msg.get("ts") or "")
            if ts and now - _ts(ts) <= window:
                out.append((str(item.get("channel") or ""), ts, msg))
    return out
