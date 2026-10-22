"""Gateway housekeeping: end idle messaging sessions and forget their routes.

``SessionDB.end_idle_sessions`` closes rows of the sources the gateway owns (Slack by default) with
the reset reason ``idle``. Closing the row is only half the boundary: the routing index this process
holds in memory would hand the same id to the next inbound message, and the route check would heal a
route parked on a compression PARENT to its (now ended) tip and then skip the stale check. So for every
ended id this sweep also drops every ``SessionStore`` entry pointing at the id or at one of its
compression ancestors, persists the drop, and evicts the cached agent for each dropped key — the
boundary evict, so the old agent's transcript is committed before its clients go. A late reply then
has no routing entry for the thread, starts a fresh session, and Slack reseeds the thread history.

Runs on the housekeeping thread, once per served profile under that profile's runtime scope
(``_for_each_served_profile``): ``load_config()`` and ``acquire()`` resolve to the scoped profile's
config and state.db. Cadence is owned here, in the profile's ``state_meta`` (shared across processes
like the auto-prune marker), not by the tick count.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# state_meta key for the last run (epoch seconds), distinct from the auto-archive/auto-prune markers.
LAST_RUN_META_KEY = "last_messaging_idle_sweep"
# Max rows ended per profile per run; well above any plausible number of concurrently guarded rows
# (see ``end_idle_sessions``: guarded rows at the head of the order consume scan budget).
_END_LIMIT = 500


def _seconds_since(now: float, raw: Any) -> float | None:
    """Age of a state_meta timestamp; None when unset or corrupt (= no prior run)."""
    try:
        return now - float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _busy_session_ids(runner: Any, store: Any) -> set[str]:
    """Session ids the sweep must not end: entries holding an ``active_turn_token`` (persisted turn
    ownership) plus the sessions of agents mid-turn (``_running_agent_items`` yields the session
    KEY; the id is read off the entry). Both are belt-and-braces over the DB turn lease, which
    ``end_idle_sessions`` already honours."""
    busy: set[str] = set()
    if store is None:
        return busy
    running_keys: set[str] = set()
    items = getattr(runner, "_running_agent_items", None)
    if callable(items):
        try:
            pairs: Any = items()
            running_keys = {str(key) for key, _agent in list(pairs)}
        except Exception:
            logger.debug("Could not read running agents for the idle sweep", exc_info=True)
    with store._lock:
        store._ensure_loaded_locked()
        for key, entry in store._entries.items():
            if getattr(entry, "active_turn_token", None) or key in running_keys:
                sid = getattr(entry, "session_id", None)
                if sid:
                    busy.add(str(sid))
    return busy


def _lineage_ids(db: Any, session_id: str) -> set[str]:
    """``{session_id}`` plus its compression ancestors (``get_compression_lineage`` walks
    ``parent_session_id`` through parents ended by compression, skipping explicit forks)."""
    ids = {session_id}
    try:
        ids.update(str(x) for x in db.get_compression_lineage(session_id) if x)
    except Exception:
        logger.debug("Compression lineage lookup failed for %s", session_id, exc_info=True)
    return ids


def _drop_routes(store: Any, victims: set[str]) -> list[str]:
    """Drop every routing entry whose session_id is in *victims* under the store lock, persist the
    drop (``_save``, the way ``remove_by_session_id`` does) and return the dropped keys."""
    if store is None or not victims:
        return []
    with store._lock:
        store._ensure_loaded_locked()
        dropped = [key for key, entry in store._entries.items()
                   if getattr(entry, "session_id", None) in victims]
        for key in dropped:
            store._entries.pop(key, None)
        hints = getattr(store, "_session_owner_hints", None)
        if hints is not None:
            for sid in victims:
                hints.pop(sid, None)
        if dropped:
            store._save()
    return dropped


def sweep_idle_messaging_sessions(runner: Any, *, profile: str = "") -> dict[str, Any]:
    """One profile's idle sweep (the caller binds the profile scope). Returns a small report:
    ``skipped`` (``"disabled"`` / ``"interval"`` / ``"no_store"`` / None), ``ended`` ids,
    ``routes_dropped`` and ``agents_evicted`` counts."""
    from hermes_cli.config import load_config

    report: dict[str, Any] = {"skipped": None, "ended": [], "routes_dropped": 0, "agents_evicted": 0}
    cfg = load_config().get("sessions") or {}
    try:
        hours = float(cfg.get("messaging_idle_hours", 0) or 0)
    except (TypeError, ValueError):
        hours = 0.0
    if hours <= 0:
        report["skipped"] = "disabled"
        return report
    sources = tuple(str(s) for s in (cfg.get("messaging_idle_sources") or ["slack"]) if s)
    try:
        interval_s = float(cfg.get("messaging_idle_interval_minutes", 60) or 0) * 60
    except (TypeError, ValueError):
        interval_s = 3600.0
    store = getattr(runner, "session_store", None)
    if store is None:
        report["skipped"] = "no_store"
        return report

    from hermes_state_registry import acquire, release_or_close
    db = acquire()
    try:
        now = time.time()
        since = _seconds_since(now, db.get_meta(LAST_RUN_META_KEY))
        if since is not None and since < interval_s:
            report["skipped"] = "interval"
            return report
        exclude = _busy_session_ids(runner, store)
        ended = db.end_idle_sessions(
            sources=sources, max_idle_seconds=hours * 3600, end_reason="idle",
            exclude_ids=tuple(sorted(exclude)), limit=_END_LIMIT)
        # Record even an empty run so the next tick does not rescan within the interval.
        db.set_meta(LAST_RUN_META_KEY, str(now))
        report["ended"] = list(ended)
        if not ended:
            logger.debug("Messaging idle sweep (%s): nothing idle past %sh", profile or "default", hours)
            return report
        victims: set[str] = set()
        for sid in ended:
            victims |= _lineage_ids(db, str(sid))
        dropped = _drop_routes(store, victims)
        report["routes_dropped"] = len(dropped)
        evict = getattr(runner, "_evict_cached_agent_at_boundary", None)
        for key in dropped:
            if not callable(evict):
                break
            try:
                evict(key)
                report["agents_evicted"] += 1
            except Exception:
                logger.debug("Idle sweep: agent eviction failed for %s", key, exc_info=True)
        logger.info(
            "Messaging idle sweep (%s): ended=%d routes_dropped=%d agents_evicted=%d (idle > %sh, sources=%s)",
            profile or "default", len(ended), report["routes_dropped"], report["agents_evicted"], hours,
            ",".join(sources))
        return report
    finally:
        release_or_close(db)
