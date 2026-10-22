"""Tests for ``SessionDB.end_idle_sessions``: the gateway's deliberate idle boundary for
messaging sources whose lifecycle it owns.

Unlike ``sweep_orphaned_sessions`` (a recoverable ``startup_orphan_reap``), an idle end is a
RESET boundary: the row must never be reopened by stale-route recovery, and the peer's
conversation generation must advance so a late reply starts a fresh conversation.
"""

from __future__ import annotations

import time

import pytest

from hermes_state import SessionDB

IDLE_S = 6 * 3600


@pytest.fixture
def db(tmp_path):
    handle = SessionDB(tmp_path / "state.db")
    yield handle
    handle.close()


def _backdate_session(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (ts, session_id))
    db._conn.commit()


def _set_message_timestamps(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (ts, session_id))
    db._conn.commit()


def _set_last_activity(db: SessionDB, session_id: str, ts: float) -> None:
    db._conn.execute("UPDATE sessions SET last_activity_at = ? WHERE id = ?", (ts, session_id))
    db._conn.commit()


def _make_session(
    db: SessionDB, session_id: str, *, source: str, started_at: float,
    message_at: float | None = None, **kwargs,
) -> None:
    db.create_session(session_id, source=source, **kwargs)
    if message_at is not None:
        db.append_message(session_id, role="user", content="hello")
        _set_message_timestamps(db, session_id, message_at)
    _backdate_session(db, session_id, started_at)
    # create_session/append_message stamp last_activity_at = now; pin it to the message time
    # so the canonical last-active expression reflects the intended age.
    _set_last_activity(db, session_id, message_at if message_at is not None else started_at)


class TestEndIdleSessions:

    def test_ends_idle_slack_row(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "idle-slack", source="slack", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["idle-slack"]
        row = db.get_session("idle-slack")
        assert row["ended_at"] is not None
        assert row["end_reason"] == "idle"

    def test_leaves_fresh_active_foreign_and_ended_rows_alone(self, db):
        now = time.time()
        stale = now - 8 * 3600
        # Fresh row.
        _make_session(db, "fresh", source="slack", started_at=now - 60, message_at=now - 60)
        # Old start, fresh activity.
        _make_session(db, "old-but-active", source="slack", started_at=stale, message_at=now - 60)
        # Source not in scope.
        _make_session(db, "telegram-stale", source="telegram", started_at=stale, message_at=stale)
        # Already ended with a different reason.
        _make_session(db, "already-ended", source="slack", started_at=stale, message_at=stale)
        db.end_session("already-ended", "agent_close")

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("fresh")["ended_at"] is None
        assert db.get_session("old-but-active")["ended_at"] is None
        assert db.get_session("telegram-stale")["ended_at"] is None
        assert db.get_session("already-ended")["end_reason"] == "agent_close"

    def test_exclude_ids_spares_row(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "held", source="slack", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(
            sources=("slack",), max_idle_seconds=IDLE_S, exclude_ids=("held",)) == []
        assert db.get_session("held")["ended_at"] is None

    def test_pinned_row_spared_by_default_ended_when_opted_in(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "pinned", source="slack", started_at=stale, message_at=stale)
        db.set_session_pinned("pinned", True)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("pinned")["ended_at"] is None

        assert db.end_idle_sessions(
            sources=("slack",), max_idle_seconds=IDLE_S, exclude_pinned=False) == ["pinned"]
        assert db.get_session("pinned")["end_reason"] == "idle"

    def test_live_turn_lease_spares_row(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "busy", source="slack", started_at=stale, message_at=stale)
        assert db.try_acquire_session_turn_lease("busy", "live-turn", ttl_seconds=300)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("busy")["ended_at"] is None

    def test_limit_caps_oldest_first(self, db):
        now = time.time()
        _make_session(db, "oldest", source="slack", started_at=now - 30 * 3600, message_at=now - 30 * 3600)
        _make_session(db, "middle", source="slack", started_at=now - 20 * 3600, message_at=now - 20 * 3600)
        _make_session(db, "newest", source="slack", started_at=now - 10 * 3600, message_at=now - 10 * 3600)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=2) == ["oldest", "middle"]
        assert db.get_session("newest")["ended_at"] is None
        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=2) == ["newest"]

    @pytest.mark.parametrize("reason", ["startup_orphan_reap", "agent_close"])
    def test_non_reset_reason_rejected(self, db, reason):
        stale = time.time() - 8 * 3600
        _make_session(db, "victim", source="slack", started_at=stale, message_at=stale)

        with pytest.raises(ValueError):
            db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, end_reason=reason)
        assert db.get_session("victim")["ended_at"] is None

    def test_ended_row_is_not_recoverable_for_peer(self, db):
        """Late reply starts fresh: the gateway's peer recovery must not resurrect the idle row."""
        stale = time.time() - 8 * 3600
        _make_session(db, "keyed", source="slack", started_at=stale, message_at=stale,
                      session_key="slack:dm:U1", user_id="U1", chat_id="D1", chat_type="dm")
        before = db.find_latest_gateway_session_for_peer(
            source="slack", session_key="slack:dm:U1", user_id="U1", chat_id="D1", chat_type="dm")
        assert before is not None and before["id"] == "keyed"

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["keyed"]

        after = db.find_latest_gateway_session_for_peer(
            source="slack", session_key="slack:dm:U1", user_id="U1", chat_id="D1", chat_type="dm")
        assert after is None

    def test_second_call_is_noop(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "once", source="slack", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["once"]
        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("once")["end_reason"] == "idle"

    def test_degenerate_arguments_are_noops(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "victim", source="slack", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=0) == []
        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=0) == []
        assert db.end_idle_sessions(sources=("", ""), max_idle_seconds=IDLE_S) == []
        assert db.get_session("victim")["ended_at"] is None


def _generation(db: SessionDB, session_key: str, source: str = "slack") -> int:
    return db.latest_conversation_boundary(session_key, source) or 0


def _make_keyed(db: SessionDB, session_id: str, *, started_at: float, message_at: float | None,
                source: str = "slack") -> str:
    """A messaging row whose peer identity is ``<source>:dm:<session_id>``; returns the key."""
    key = f"{source}:dm:{session_id}"
    _make_session(db, session_id, source=source, started_at=started_at, message_at=message_at,
                  session_key=key, user_id=f"U-{session_id}", chat_id=f"D-{session_id}", chat_type="dm")
    return key


def _insert_compression_lock(db: SessionDB, session_id: str, holder: str, expires_at: float) -> None:
    db._conn.execute(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (session_id, holder, time.time(), expires_at))
    db._conn.commit()


def _compression_lock_rows(db: SessionDB, session_id: str) -> list:
    return db._conn.execute(
        "SELECT holder FROM compression_locks WHERE session_id = ?", (session_id,)).fetchall()


class TestConversationGeneration:
    """Finding 2: the peer's generation advances exactly once per row this call ended, never for a
    row it spared or that was already closed."""

    @pytest.mark.parametrize("reason", ["idle", "daily"])
    def test_ended_row_bumps_peer_generation_exactly_once(self, db, reason):
        stale = time.time() - 8 * 3600
        key = _make_keyed(db, "keyed", started_at=stale, message_at=stale)
        assert _generation(db, key) == 0

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, end_reason=reason) == ["keyed"]
        assert _generation(db, key) == 1
        assert db.get_session("keyed")["end_reason"] == reason

        # Already ended: a repeat call must not rotate the peer again.
        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, end_reason=reason) == []
        assert _generation(db, key) == 1

    def test_one_bump_per_ended_row_across_peers(self, db):
        stale = time.time() - 8 * 3600
        key_a = _make_keyed(db, "a", started_at=stale - 60, message_at=stale - 60)
        key_b = _make_keyed(db, "b", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["a", "b"]
        assert _generation(db, key_a) == 1
        assert _generation(db, key_b) == 1

    def test_spared_rows_do_not_bump_generation(self, db):
        now = time.time()
        stale = now - 8 * 3600
        key_excluded = _make_keyed(db, "excluded", started_at=stale, message_at=stale)
        key_guarded = _make_keyed(db, "guarded", started_at=stale, message_at=stale)
        assert db.try_acquire_session_turn_lease("guarded", "live-turn", ttl_seconds=300)
        key_pinned = _make_keyed(db, "pinned", started_at=stale, message_at=stale)
        db.set_session_pinned("pinned", True)
        key_already = _make_keyed(db, "already", started_at=stale, message_at=stale)
        db.end_session("already", "agent_close")  # not a reset reason: generation stays 0
        key_fresh = _make_keyed(db, "fresh", started_at=stale, message_at=now - 60)  # predicate-rejected
        key_victim = _make_keyed(db, "victim", started_at=stale, message_at=stale)

        assert db.end_idle_sessions(
            sources=("slack",), max_idle_seconds=IDLE_S, exclude_ids=("excluded",)) == ["victim"]

        assert _generation(db, key_victim) == 1
        for key in (key_excluded, key_guarded, key_pinned, key_already, key_fresh):
            assert _generation(db, key) == 0, key
        for sid in ("excluded", "guarded", "pinned", "fresh"):
            assert db.get_session(sid)["ended_at"] is None, sid
        assert db.get_session("already")["end_reason"] == "agent_close"


class TestCoupledClocks:
    """Finding 3: BOTH ``started_at`` and canonical last activity must be past the cutoff."""

    def test_old_start_but_fresh_message_is_spared(self, db):
        """Activity is read from the message timestamp, not ``last_activity_at``."""
        now = time.time()
        stale = now - 8 * 3600
        _make_session(db, "chatty", source="slack", started_at=stale, message_at=now - 60)
        _set_last_activity(db, "chatty", stale)  # stale heartbeat column; the message is the truth

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("chatty")["ended_at"] is None

    def test_fresh_start_with_old_activity_is_spared(self, db):
        """A fresh compression child carries its parent's old messages; its young ``started_at`` spares it."""
        now = time.time()
        stale = now - 8 * 3600
        _make_session(db, "child", source="slack", started_at=now - 60, message_at=stale)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("child")["ended_at"] is None

    def test_exact_cutoff_is_not_idle(self, db, monkeypatch):
        """Both predicates are strict ``<``: a row whose clocks sit exactly on the cutoff is spared;
        one second older is ended."""
        frozen = 1_800_000_000.0
        monkeypatch.setattr(time, "time", lambda: frozen)
        cutoff = frozen - IDLE_S
        _make_session(db, "on-cutoff", source="slack", started_at=cutoff, message_at=cutoff)
        _make_session(db, "past-cutoff", source="slack", started_at=cutoff - 1, message_at=cutoff - 1)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["past-cutoff"]
        assert db.get_session("on-cutoff")["ended_at"] is None


class TestCompressionGuards:
    """Finding 4: a live compression lock spares the row; an expired one is reclaimed and the row ends."""

    def test_live_compression_lock_spares_row(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "compressing", source="slack", started_at=stale, message_at=stale)
        assert db.try_acquire_compression_lock("compressing", "foreign-writer", ttl_seconds=300)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == []
        assert db.get_session("compressing")["ended_at"] is None
        assert [r["holder"] for r in _compression_lock_rows(db, "compressing")] == ["foreign-writer"]

    def test_expired_compression_lock_is_reclaimed_and_row_ends(self, db):
        stale = time.time() - 8 * 3600
        _make_session(db, "abandoned", source="slack", started_at=stale, message_at=stale)
        _insert_compression_lock(db, "abandoned", "dead-writer", expires_at=time.time() - 60)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S) == ["abandoned"]
        assert db.get_session("abandoned")["end_reason"] == "idle"
        assert _compression_lock_rows(db, "abandoned") == []


class TestStarvation:
    """Finding 1/5: ``limit`` is the max rows ENDED per call, not rows inspected; excluded or guarded
    rows at the head of the oldest-first order must not starve eligible rows behind them."""

    def test_limit_one_with_oldest_excluded_ends_next_eligible(self, db):
        now = time.time()
        _make_session(db, "oldest-held", source="slack", started_at=now - 30 * 3600, message_at=now - 30 * 3600)
        _make_session(db, "eligible", source="slack", started_at=now - 20 * 3600, message_at=now - 20 * 3600)

        assert db.end_idle_sessions(
            sources=("slack",), max_idle_seconds=IDLE_S, limit=1, exclude_ids=("oldest-held",)) == ["eligible"]
        assert db.get_session("oldest-held")["ended_at"] is None

    def test_limit_one_with_oldest_guarded_ends_next_eligible(self, db):
        now = time.time()
        _make_session(db, "oldest-busy", source="slack", started_at=now - 30 * 3600, message_at=now - 30 * 3600)
        assert db.try_acquire_session_turn_lease("oldest-busy", "live-turn", ttl_seconds=300)
        _make_session(db, "eligible", source="slack", started_at=now - 20 * 3600, message_at=now - 20 * 3600)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=1) == ["eligible"]
        assert db.get_session("oldest-busy")["ended_at"] is None

    def test_limit_counts_ended_rows_not_scanned(self, db):
        now = time.time()
        for i in range(3):
            sid = f"busy-{i}"
            _make_session(db, sid, source="slack", started_at=now - (40 - i) * 3600, message_at=now - (40 - i) * 3600)
            assert db.try_acquire_session_turn_lease(sid, f"turn-{i}", ttl_seconds=300)
        _make_session(db, "e1", source="slack", started_at=now - 20 * 3600, message_at=now - 20 * 3600)
        _make_session(db, "e2", source="slack", started_at=now - 10 * 3600, message_at=now - 10 * 3600)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=2) == ["e1", "e2"]

    def test_scan_bound_caps_work_per_call(self, db):
        """At most ``limit * 4`` candidates are inspected per call so a store full of guarded rows
        cannot hold the write lock indefinitely; the eligible row past that bound waits for a later call."""
        now = time.time()
        for i in range(4):  # exactly limit*4 guarded rows ahead of the eligible one
            sid = f"busy-{i}"
            _make_session(db, sid, source="slack", started_at=now - (40 - i) * 3600, message_at=now - (40 - i) * 3600)
            assert db.try_acquire_session_turn_lease(sid, f"turn-{i}", ttl_seconds=300)
        _make_session(db, "eligible", source="slack", started_at=now - 20 * 3600, message_at=now - 20 * 3600)

        assert db.end_idle_sessions(sources=("slack",), max_idle_seconds=IDLE_S, limit=1) == []
        assert db.get_session("eligible")["ended_at"] is None
        # Excluded ids are filtered in SQL, so they never consume scan budget.
        assert db.end_idle_sessions(
            sources=("slack",), max_idle_seconds=IDLE_S, limit=1,
            exclude_ids=tuple(f"busy-{i}" for i in range(4))) == ["eligible"]
