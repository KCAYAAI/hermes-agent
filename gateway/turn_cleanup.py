"""Bounded turn cleanup: a stuck background step must never hold a finished turn's reply.

After the agent returns, the gateway cancels and joins the turn's background tasks (progress
card, tool log, interrupt monitor, heartbeat, tracking, stream consumer) and clears the typing
status before it shapes and sends the reply. Each of those joins used to be unbounded, so one
task that ignored or swallowed its cancellation (for example a transport that turns
``CancelledError`` into ``TimeoutError``) held the reply and the session admission forever.

``join_cancelled_task`` and ``run_bounded`` give each step a fixed budget. A task that outlives
it is re-cancelled, logged with its await chain, and kept in a supervised set until it really
ends, so its exception is retrieved and its references are released. ``stall_watchdog`` logs
the await chain of a turn whose post-agent processing is unexpectedly slow, so the exact
blocking await is visible in the log without a debugger.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from contextlib import contextmanager
from typing import Any, Awaitable, Iterator, Optional

logger = logging.getLogger("gateway.run")

TURN_TASK_CLEANUP_TIMEOUT_S = 5.0
TURN_STALL_WATCHDOG_S = 30.0

# Tasks that outlived their cleanup budget. Held here (not on the turn) so the turn's context can
# be released while the task finishes; each removes itself when done.
_DETACHED: set[asyncio.Task] = set()


def detached_turn_tasks() -> frozenset:
    """Snapshot of cleanup tasks still running past their budget (for diagnostics and tests)."""
    return frozenset(_DETACHED)


def describe_await_chain(task: asyncio.Future) -> str:
    """Best-effort await chain of ``task``: full call graph on 3.14+, else the task's stack."""
    try:
        fmt = getattr(asyncio, "format_call_graph", None)
        if fmt is not None:
            return fmt(task)
    except Exception:  # pragma: no cover - diagnostics only
        pass
    try:
        frames = task.get_stack() if isinstance(task, asyncio.Task) else []
        return "".join(traceback.format_list(traceback.extract_stack(frames[-1]) if frames else []))
    except Exception:  # pragma: no cover - diagnostics only
        return "<await chain unavailable>"


def _retire(task: asyncio.Future) -> None:
    _DETACHED.discard(task)  # type: ignore[arg-type]
    if task.cancelled():
        return
    exc = task.exception()  # retrieve so asyncio never logs "exception was never retrieved"
    if exc is not None:
        logger.debug("Detached turn task ended with %r", exc)


def _detach(task: asyncio.Future, label: str, session_key: Optional[str], timeout: float) -> None:
    logger.warning(
        "Turn cleanup: %s did not finish %.1fs after cancel (session=%s); detaching it so the "
        "reply is not held. Await chain:\n%s", label, timeout, session_key, describe_await_chain(task),
    )
    task.cancel()
    _DETACHED.add(task)  # type: ignore[arg-type]
    task.add_done_callback(_retire)


async def join_cancelled_task(
    task: Optional[asyncio.Future], *, label: str, session_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> bool:
    """Wait at most ``timeout`` for an already-cancelled ``task`` to finish.

    Returns True when the task finished (or was absent), False when it was detached. Never raises
    for the task's own outcome. Cancellation of the CALLER is forwarded to the task (as the old
    direct ``await`` did) and cleanup continues, so the caller still releases its session slot.
    """
    if task is None:
        return True
    timeout = TURN_TASK_CLEANUP_TIMEOUT_S if timeout is None else timeout
    if not task.done():
        try:
            await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            # The cleanup itself was cancelled: pass it on to the child exactly like a direct
            # await would, then keep the bounded wait so cleanup still completes.
            task.cancel()
            with _suppress_cancel():
                await asyncio.wait({task}, timeout=timeout)
    if not task.done():
        _detach(task, label, session_key, timeout)
        return False
    if not task.cancelled():
        exc = task.exception()
        if exc is not None:
            logger.debug("background turn task %s failed during cleanup", label, exc_info=exc)
    return True


async def run_bounded(
    coro: Awaitable[Any], *, label: str, session_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> bool:
    """Run a best-effort cleanup step with a hard budget. True when it finished in time."""
    timeout = TURN_TASK_CLEANUP_TIMEOUT_S if timeout is None else timeout
    task = asyncio.ensure_future(coro)
    if not task.done():
        try:
            await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
    if not task.done():
        _detach(task, label, session_key, timeout)
        return False
    if not task.cancelled() and task.exception() is not None:
        logger.debug("cleanup step %s failed", label, exc_info=task.exception())
    return True


@contextmanager
def _suppress_cancel() -> Iterator[None]:
    try:
        yield
    except asyncio.CancelledError:
        pass


@contextmanager
def stall_watchdog(
    label: str, session_key: Optional[str] = None, *, seconds: Optional[float] = None,
) -> Iterator[None]:
    """Log the current task's await chain once if the wrapped block runs longer than ``seconds``."""
    seconds = TURN_STALL_WATCHDOG_S if seconds is None else seconds
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    task = asyncio.current_task(loop) if loop is not None else None
    if loop is None or task is None:
        yield
        return

    def _fire() -> None:
        if not task.done():
            logger.warning(
                "Turn stalled: %s still running after %.0fs (session=%s). Await chain:\n%s",
                label, seconds, session_key, describe_await_chain(task),
            )

    handle = loop.call_later(seconds, _fire)
    try:
        yield
    finally:
        handle.cancel()
