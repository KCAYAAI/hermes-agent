"""Gateway housekeeping: end idle messaging sessions and forget their routes.

``SessionDB.end_idle_sessions`` closes rows of the sources the gateway owns (Slack by default) with
the reset reason ``idle``. Closing the row is only half the boundary: the routing index this process
holds in memory would hand the same id to the next inbound message. So every run also RECONCILES
this profile's routing entries against the profile's state.db: an entry whose row (or whose
compression tip's row) ended with a non-recoverable reason is retired — dropped from the store,
persisted, and its cached agent evicted (the boundary evict, so the old agent's transcript is
committed before its clients go). A late reply then has no routing entry for the thread, starts a
fresh session, and Slack reseeds the thread history.

Reconcile is state-based, not event-based: it does not depend on what THIS run ended, so a route
whose cleanup was lost to a crash, a failed save or a failed lineage lookup after the DB commit is
picked up by the next run (idempotent, self-healing across restarts). It is bounded to
``_RECONCILE_LIMIT`` entries per run, read in a few batched ``IN (...)`` queries.

Runs on the housekeeping thread, once per served profile under that profile's runtime scope
(``_for_each_served_profile``): ``load_config()`` and ``acquire()`` resolve to the scoped profile's
config and state.db. ``runner.session_store`` is ONE store across served profiles, so every step
here looks only at entries whose key namespace belongs to the swept profile (ids are timestamps
plus eight hex chars, not globally unique; the same id can exist in two profiles' databases).
Cadence is owned here, in the profile's ``state_meta`` (shared across processes like the auto-prune
marker), not by the tick count; the marker is written only after reconcile succeeded, so a failed
run is retried on the next tick.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from hermes_state_common import _RECOVERABLE_END_REASONS, _id_chunks, _placeholders

logger = logging.getLogger(__name__)

# state_meta key for the last run (epoch seconds), distinct from the auto-archive/auto-prune markers.
LAST_RUN_META_KEY = "last_messaging_idle_sweep"
# Max rows ended per profile per run; well above any plausible number of concurrently guarded rows
# (see ``end_idle_sessions``: guarded rows at the head of the order consume scan budget).
_END_LIMIT = 500
# Max routing entries (of the swept profile) examined by one reconcile run. Entries beyond it are
# examined on a later run; the housekeeping thread runs this at most once per interval, so the bound
# caps one run's read cost, not what eventually gets cleaned.
_RECONCILE_LIMIT = 2000
# A persistently failing sweep warns at most this often per profile (debug otherwise).
_WARN_INTERVAL_S = 3600.0
_WARNED_AT: dict[str, float] = {}
# Rows ended with these reasons are never retired here: ``compression`` means "continued by a
# child" (the tip decides), the recoverable set is reopened by the next message (#54878 recovery).
_KEEP_END_REASONS = frozenset(_RECOVERABLE_END_REASONS) | {"compression"}


def _seconds_since(now: float, raw: Any) -> float | None:
    """Age of a state_meta timestamp; None when unset or corrupt (= no prior run)."""
    try:
        return now - float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _profile_of_key(session_key: Any) -> str:
    """Profile that owns a routing key (``agent:<ns>:...``; ``agent:main`` and legacy keys without
    a namespace are the default profile's). Same decoding as ``SessionStore._db_for_key``."""
    from hermes_state_keys import profile_from_session_key
    return profile_from_session_key(session_key) or "default"


def _owned_entries(store: Any, owner: str) -> list[tuple[str, Any]]:
    """``(key, entry)`` of the swept profile's routing entries. Lock held by the caller."""
    return [(key, entry) for key, entry in store._entries.items() if _profile_of_key(key) == owner]


def _running_keys(runner: Any) -> set[str]:
    items: Any = getattr(runner, "_running_agent_items", None)
    if not callable(items):
        return set()
    try:
        pairs: Any = items()
        return {str(key) for key, _agent in list(pairs)}
    except Exception:
        logger.debug("Could not read running agents for the idle sweep", exc_info=True)
        return set()


def _is_busy(key: str, entry: Any, running: set[str]) -> bool:
    """An entry holding an ``active_turn_token`` (persisted turn ownership) or whose agent is
    mid-turn (``_running_agent_items`` yields the session KEY)."""
    return bool(getattr(entry, "active_turn_token", None)) or key in running


def _busy_session_ids(runner: Any, store: Any, owner: str) -> set[str]:
    """Session ids of the swept profile the sweep must not end. Belt-and-braces over the DB turn
    lease, which ``end_idle_sessions`` already honours. Only this profile's entries count: another
    profile's busy entry sharing the id is a different conversation in a different database."""
    busy: set[str] = set()
    if store is None:
        return busy
    running = _running_keys(runner)
    with store._lock:
        store._ensure_loaded_locked()
        for key, entry in _owned_entries(store, owner):
            if _is_busy(key, entry, running):
                sid = getattr(entry, "session_id", None)
                if sid:
                    busy.add(str(sid))
    return busy


def _end_reasons(db: Any, session_ids: set[str]) -> dict[str, Optional[str]]:
    """``{id: end_reason}`` for the rows that exist, in chunked ``IN (...)`` reads."""
    out: dict[str, Optional[str]] = {}
    for chunk in _id_chunks(sorted(session_ids)):
        rows = db._read_all(
            f"SELECT id, end_reason FROM sessions WHERE id IN ({_placeholders(chunk)})", tuple(chunk))
        for row in rows:
            out[str(row[0])] = row[1]
    return out


def _retired_reason(reason: Any) -> bool:
    """A row is retired when ended with a deliberate, non-recoverable reason. A missing row is NOT
    retired here (the routing-time self-heal owns hard deletes, #42422)."""
    return reason is not None and reason not in _KEEP_END_REASONS


def _lineage_ids(db: Any, session_id: str) -> set[str]:
    """``{session_id}`` plus its compression ancestors/descendants (``get_compression_lineage``
    walks ``parent_session_id`` through parents ended by compression, skipping explicit forks)."""
    ids = {session_id}
    try:
        ids.update(str(x) for x in db.get_compression_lineage(session_id) if x)
    except Exception:
        logger.debug("Compression lineage lookup failed for %s", session_id, exc_info=True)
    return ids


def _find_retired_routes(db: Any, store: Any, owner: str, running: set[str]) -> dict[str, set[str]]:
    """``{key: retired session ids}`` for the swept profile's routing entries whose row ended with
    a non-recoverable reason, or whose row is a compression parent whose tip did. At most
    ``_RECONCILE_LIMIT`` entries examined: one batched read for the routed ids, one per compression
    parent for its tip (rare: routes normally heal to the tip), one batched read for those tips.
    Busy entries are skipped: the turn's own unwind must still find its entry (#99106 guards the
    next message)."""
    with store._lock:
        store._ensure_loaded_locked()
        candidates = [(key, str(entry.session_id)) for key, entry in _owned_entries(store, owner)
                      if getattr(entry, "session_id", None) and not _is_busy(key, entry, running)]
    candidates = candidates[:_RECONCILE_LIMIT]
    if not candidates:
        return {}
    reasons = _end_reasons(db, {sid for _key, sid in candidates})
    tips: dict[str, str] = {}
    for _key, sid in candidates:
        if reasons.get(sid) == "compression" and sid not in tips:
            try:
                tip = db.get_compression_tip(sid)
            except Exception:
                logger.debug("Compression tip lookup failed for %s", sid, exc_info=True)
                continue
            if tip and str(tip) != sid:
                tips[sid] = str(tip)
    if tips:
        reasons.update(_end_reasons(db, set(tips.values()) - set(reasons)))
    retired: dict[str, set[str]] = {}
    for key, sid in candidates:
        # A compression parent without a resolvable tip keeps ``target == sid`` and its
        # ``compression`` reason is in the keep set, so it is left alone.
        target = tips.get(sid, sid)
        if _retired_reason(reasons.get(target)):
            retired[key] = _lineage_ids(db, target) | {sid}
    return retired


def _drop_routes(store: Any, keys: list[str], session_ids: set[str]) -> tuple[list[str], Optional[Exception]]:
    """Drop *keys* under the store lock, forget owner hints for *session_ids*, persist once
    (``_save``, the way ``remove_by_session_id`` does). Returns ``(dropped keys, save error)``:
    the keys are gone from memory even when the save failed, so the caller still evicts their
    agents and then re-raises (the on-disk copy is repaired by the next successful save or the
    startup prune, ``_prune_stale_sessions_locked``)."""
    if store is None or not keys:
        return [], None
    with store._lock:
        store._ensure_loaded_locked()
        dropped = [key for key in keys if store._entries.pop(key, None) is not None]
        hints = getattr(store, "_session_owner_hints", None)
        if hints is not None:
            for sid in session_ids:
                hints.pop(sid, None)
        if not dropped:
            return dropped, None
        try:
            store._save()
        except Exception as exc:
            return dropped, exc
    return dropped, None


def _evict_retired_agent(runner: Any, key: str, retired_ids: set[str]) -> bool:
    """Boundary-evict the agent cached under *key* only while it still belongs to one of the
    retired ids; a fresh conversation's agent already installed under the same key is left alone
    (the identity compare happens under ``_agent_cache_lock`` inside ``_evict_cached_agent``)."""
    evict = getattr(runner, "_evict_cached_agent_at_boundary", None)
    if not callable(evict):
        return False
    return bool(evict(key, retired_session_ids=frozenset(retired_ids)))


def _reconcile_routes(runner: Any, db: Any, store: Any, owner: str, report: dict[str, Any]) -> None:
    retired = _find_retired_routes(db, store, owner, _running_keys(runner))
    if not retired:
        return
    all_ids: set[str] = set().union(*retired.values())
    dropped, save_error = _drop_routes(store, list(retired), all_ids)
    report["routes_dropped"] = len(dropped)
    for key in dropped:
        try:
            if _evict_retired_agent(runner, key, retired[key]):
                report["agents_evicted"] += 1
        except Exception:
            logger.debug("Idle sweep: agent eviction failed for %s", key, exc_info=True)
    if save_error is not None:
        raise save_error


def sweep_idle_messaging_sessions(runner: Any, *, profile: str = "") -> dict[str, Any]:
    """One profile's idle sweep (the caller binds the profile scope). Returns a small report:
    ``skipped`` (``"disabled"`` / ``"interval"`` / ``"no_store"`` / None), ``ended`` ids,
    ``routes_dropped`` and ``agents_evicted`` counts. Raises when the run could not complete
    (the interval marker is then left untouched so the next tick retries)."""
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
    owner = profile or "default"

    from hermes_state_registry import acquire, release_or_close
    db = acquire()
    try:
        now = time.time()
        since = _seconds_since(now, db.get_meta(LAST_RUN_META_KEY))
        if since is not None and since < interval_s:
            report["skipped"] = "interval"
            return report
        exclude = _busy_session_ids(runner, store, owner)
        ended = db.end_idle_sessions(
            sources=sources, max_idle_seconds=hours * 3600, end_reason="idle",
            exclude_ids=tuple(sorted(exclude)), limit=_END_LIMIT)
        report["ended"] = list(ended)
        # Always: routes retired by this run, by a run whose cleanup failed, or by any other
        # deliberate end (reset, prune) are reconciled from DB state, not from ``ended``.
        _reconcile_routes(runner, db, store, owner, report)
        # Record even an empty run so the next tick does not rescan within the interval. After
        # reconcile: a run that failed before this point is retried on the next tick.
        db.set_meta(LAST_RUN_META_KEY, str(now))
        if not ended and not report["routes_dropped"]:
            logger.debug("Messaging idle sweep (%s): nothing idle past %sh", owner, hours)
            return report
        logger.info(
            "Messaging idle sweep (%s): ended=%d routes_dropped=%d agents_evicted=%d (idle > %sh, sources=%s)",
            owner, len(ended), report["routes_dropped"], report["agents_evicted"], hours,
            ",".join(sources))
        return report
    finally:
        release_or_close(db)


def run_idle_sweep_guarded(runner: Any, *, profile: str = "") -> Optional[dict[str, Any]]:
    """``sweep_idle_messaging_sessions`` for the housekeeping loop: a failure is WARNED at most
    once per ``_WARN_INTERVAL_S`` per profile (debug otherwise), so a broken sweep stays visible
    without a warning on every 60-second tick. Never raises."""
    try:
        return sweep_idle_messaging_sessions(runner, profile=profile)
    except Exception as exc:
        label = profile or "default"
        now = time.monotonic()
        last = _WARNED_AT.get(label)
        if last is None or now - last >= _WARN_INTERVAL_S:
            _WARNED_AT[label] = now
            logger.warning("Messaging idle sweep (%s) failed: %s", label, exc, exc_info=True)
        else:
            logger.debug("Messaging idle sweep (%s) failed again: %s", label, exc)
        return None
