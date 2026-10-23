"""Contract cases for ``CoordinationRepository.record_recovered_reply`` (session restart recovery ledgers the saved answer
of a person's lost turn), run against every backend.

The clock is pinned through ``time.time``; owners and the dead-owner probe arrive as data, so no case needs a dead process.
"""

from __future__ import annotations

import pytest

from persistence.authority.coordination import FailedObligationRow, OwnerStamp, RecordedOwner

NOW = 1_800_000_000.0
OLD = OwnerStamp(pid=11, started_at=111)
ME = OwnerStamp(pid=22, started_at=222)


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("time.time", lambda: now[0])
    return now


def recovered(coord, oid: str, *, session_key: str = "sk-1", content: str = "the saved answer", since: float = NOW - 1,
              owner: OwnerStamp = ME) -> bool:
    return coord.record_recovered_reply(obligation_id=oid, session_key=session_key, platform="slack", chat_id="C1",
                                        thread_id="171.1", content=content, since=since, adapter_profile=" bot ",
                                        owner=owner)


def test_inserts_once_owned_and_attempting(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    # CONTRACT: the first call inserts and answers True: the row is attempting, owned by the caller, one attempt spent.
    assert recovered(coord, "o-rec") is True
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(22, 222)})
    # CONTRACT: the same id is never inserted again, whatever the session key, content, since or owner.
    assert recovered(coord, "o-rec", session_key="sk-other", content="other", since=NOW + 10, owner=OLD) is False
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(22, 222)})
    # A sweeper that sees the owner alive leaves the row alone; once the owner is dead the row is claimed with the marker.
    assert coord.sweep_recoverable_obligations(owner=OLD, dead_owners=frozenset({RecordedOwner(11, 111)})) == ()
    (claimed,) = coord.sweep_recoverable_obligations(owner=OLD, dead_owners=frozenset({RecordedOwner(22, 222)}))
    assert (claimed.obligation_id, claimed.session_key, claimed.platform, claimed.chat_id, claimed.thread_id, claimed.content,
            claimed.profile, claimed.attempts, claimed.needs_marker) == (
        "o-rec", "sk-1", "slack", "C1", "171.1", "the saved answer", "bot", 2, True)


def test_same_reply_for_the_conversation_since_dedupes(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    coord.record_delivery_obligation(obligation_id="o-live", session_key="sk-1", platform="slack", chat_id="C1",
                                     thread_id="171.1", content="the saved answer", owner=OLD)
    # CONTRACT: a row with this session key and content created at or after ``since`` (any id, any state) dedupes.
    assert recovered(coord, "o-rec", since=NOW - 1) is False
    assert recovered(coord, "o-rec", since=NOW) is False
    coord.mark_obligation_delivered(obligation_id="o-live")
    assert recovered(coord, "o-rec", since=NOW) is False
    # Another conversation or another answer is not the same reply; a row older than ``since`` does not count.
    assert recovered(coord, "o-other-sk", session_key="sk-2") is True
    assert recovered(coord, "o-other-text", content="a different answer") is True
    clock[0] += 5
    assert recovered(coord, "o-later", since=NOW + 1) is True
    # The recovered row itself dedupes the same reply under a new id.
    assert recovered(coord, "o-again", since=NOW + 1) is False


def test_a_failed_row_is_never_re_owned(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    assert recovered(coord, "o-rec") is True
    coord.mark_obligation_failed(obligation_id="o-rec", error="boom")
    clock[0] += 1
    # CONTRACT: a ledgered row in any state is never re-owned: the call answers False and the row keeps its owner,
    # state, error and attempt count.
    assert recovered(coord, "o-rec", owner=OLD) is False
    assert recovered(coord, "o-rec-2", owner=OLD, since=NOW) is False
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(22, 222)})
    assert coord.list_owned_failed_obligations(owner=OLD) == ()
    assert coord.list_owned_failed_obligations(owner=ME) == (
        FailedObligationRow(platform="slack", adapter_profile="bot", updated_at=NOW, last_error="boom", attempts=1,
                            created_at=NOW),)


def test_a_turn_tag_limits_the_same_reply_to_this_turns_rows(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    def live(oid, content, turn, sk="sk-1"):
        coord.record_delivery_obligation(obligation_id=oid, session_key=sk, platform="slack", chat_id="C1",
                                         thread_id="171.1", content=content, owner=OLD, turn=turn)
    live("o-other-turn", "the saved answer", "admission:p:other")
    live("o-untagged-new", "the saved answer", None)          # a send with no turn (stored as '')
    live("o-this-turn", "another answer", "admission:p:head")  # this turn's send, whatever id keyed it
    live("o-raw", "Done\n[ SILENT ]", "admission:p:head")

    def turn(oid, content="the saved answer", contents=None, **kw):
        return coord.record_recovered_reply(obligation_id=oid, session_key="sk-1", platform="slack", chat_id="C1",
                                            thread_id="171.1", content=content, since=NOW - 1, adapter_profile=None,
                                            owner=ME, turn="admission:p:head", contents=contents, **kw)
    # CONTRACT: with a turn, another turn's identical reply and an untagged ('') row do not dedupe, even when
    # their ids are listed as legacy ids ('' is not a legacy row): inserted.
    assert turn("o-rec", legacy_ids=["o-untagged-new", "o-other-turn"]) is True
    # CONTRACT: a row tagged with this turn and the same content dedupes, whatever its id.
    assert turn("o-rec-2", content="another answer") is False
    # CONTRACT: ``contents`` widens the match to the other texts a send of this turn may have used.
    assert turn("o-rec-3", content="Done", contents=["Done", "Done\n[ SILENT ]"]) is False
    assert turn("o-rec-4", content="Done") is True
    # CONTRACT: the id itself still never re-inserts.
    assert turn("o-rec") is False
    # CONTRACT: without a turn, the old content-only rule holds (an owed record written before tags).
    assert coord.record_recovered_reply(obligation_id="o-rec-7", session_key="sk-1", platform="slack", chat_id="C1",
                                        thread_id="171.1", content="the saved answer", since=NOW - 1,
                                        adapter_profile=None, owner=ME) is False
