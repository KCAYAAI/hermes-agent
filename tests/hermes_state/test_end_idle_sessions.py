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
