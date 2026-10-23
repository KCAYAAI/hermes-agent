"""Let background work decide a turn's final reaction.

A plugin that tracks work launched from a thread (e.g. subagents) registers a check. At the end of
a successful turn the adapter asks every check for the thread's state:

* ``"hold"``   - work is still running: keep 👀; the owner later calls the adapter's
  ``finish_deferred_reactions`` when it settles (it is routed to the ReactionLedger).
* ``"failed"`` - launched work already failed: end with ❌ instead of ✅.
* ``None``     - nothing to say: normal ✅.

Checks receive the adapter, so in a multiplex gateway one profile's work never decides another
profile's reaction.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

_checks: List[Callable[..., Optional[str]]] = []
_lock = threading.Lock()


def register_completion_deferral(fn: Callable[..., Optional[str]]) -> None:
    with _lock:
        if fn not in _checks:
            _checks.append(fn)


def completion_state(*, chat_id: str, thread_id: str, message_id: str, adapter: Any, team_id: str = "") -> Optional[str]:
    """``"hold"`` beats ``"failed"`` beats ``None`` across all registered checks."""
    with _lock:
        checks = list(_checks)
    result: Optional[str] = None
    for fn in checks:
        try:
            state = fn(chat_id=chat_id, thread_id=thread_id, message_id=message_id, adapter=adapter, team_id=team_id)
        except Exception:
            logger.debug("completion deferral check failed", exc_info=True)
            continue
        if state == "hold":
            return "hold"
        if state == "failed":
            result = "failed"
    return result
