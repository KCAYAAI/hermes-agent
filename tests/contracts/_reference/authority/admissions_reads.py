"""The admission reads of the reference and the API payload compaction."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping, Sequence

from persistence.authority.admissions import (
    RECOVERY_ATTEMPTS_PREFIX, RECOVERY_OWED_PREFIX, AdmissionRecord, AdmissionResult, LostTurnRow, LostTurnTranscript,
)
from persistence.core.conventions import RowVisibility
from persistence.core.canonical import canonical_json
from persistence.core.errors import AuthorityRefusal, PermanentStoreError
from session_rules.api_payloads import compacted_api_payload

from . import admissions_rows as rows
from .admissions_results import load_result

_PENDING = ("queued", "started", "unknown")


def _api_turn(payload: Any) -> Any:
    return payload.get("api_turn_v1") if isinstance(payload, dict) else None


def _compaction_candidate(row: Mapping[str, Any]) -> bool:
    """Whether a terminal API admission's payload is a compaction candidate; the text match is ASCII case-insensitive."""
    if row["status"] != "terminal" or row["principal_id"] != "api":
        return False
    data = _api_turn(row["payload"])
    if not isinstance(data, dict):
        return False
    inline = "media" in data and "data:image/" in canonical_json(row["payload"]).lower()
    history = isinstance(data.get("history"), list) and not str(row["request_id"]).lower().startswith("responses:")
    return inline or history


def _webhook_source(payload: Any) -> Mapping[str, Any] | None:
    native = payload.get("native_text_v1") if isinstance(payload, dict) else None
    source = native.get("source") if isinstance(native, dict) else None
    if not isinstance(source, dict) or source.get("platform") != "webhook" or native.get("automation") is not None:
        return None
    return source


class AdmissionReadsMixin:
    """The reads; mixed into ``InMemoryAdmissionRepository``."""

    def _committed(self) -> dict[str, Any]:
        return self._state().snapshot

    def find(self, *, principal_id: str, session_id: str, request_id: str) -> AdmissionRecord | None:
        key = (rows._bound_id(principal_id, 1), rows._bound_id(session_id, 2), rows._bound_id(request_id, 3))
        row = next((r for r in rows.rows_by_seq(self._committed()) if (r["principal_id"], r["target_session_id"], r["request_id"]) == key), None)
        return rows.record(row) if row is not None else None

    def match(self, *, request_id: str | None = None, admission_id: str | None = None, principal_id: str | None = None, session_id: str | None = None,
              include_retired: bool = False) -> list[AdmissionRecord]:
        given = [(name, value) for name, value in (("request_id", request_id), ("admission_id", admission_id), ("principal_id", principal_id), ("target_session_id", session_id)) if value is not None]
        try:
            wanted = {name: rows._bound_id(value, index) for index, (name, value) in enumerate(given, start=1)}
        except sqlite3.ProgrammingError as exc:
            raise PermanentStoreError(str(exc)) from exc
        authority = self._committed()
        found = [r for r in rows.rows_by_seq(authority) if all(r[name] == value for name, value in wanted.items())]
        if include_retired and request_id is not None:
            found.extend(t for t in self._api_tombstones(authority, request_id) if all(t.get(name) == value for name, value in wanted.items()))
        return [rows.record(r) for r in found]

    def list(self, *, session_id: str, pending_only: bool = True) -> list[AdmissionRecord]:
        pending = bool(pending_only)
        selected = rows.session_rows(self._committed(), session_id)
        return [rows.record(r) for r in selected if not pending or r["status"] in _PENDING]

    def get(self, *, admission_id: str) -> AdmissionRecord | None:
        row = rows.lookup(self._committed(), admission_id)
        return rows.record(row) if row is not None else None

    def latest_admission(self, *, session_id: str, payload_marker: str) -> AdmissionRecord | None:
        selected = rows.session_rows(self._committed(), session_id)
        row = next((r for r in reversed(selected) if isinstance(r["payload"], dict) and payload_marker in r["payload"]), None)
        return rows.record(row) if row is not None else None

    def settled_result(self, *, admission_id: str, epoch: int | None = None, generation: int | None = None) -> AdmissionResult | None:
        if (epoch is None) != (generation is None):
            raise ValueError("settled_result takes epoch and generation together or not at all")
        authority = self._committed()
        if epoch is not None:
            self._state().require_epoch(authority, epoch)
        row = rows.admission(authority, admission_id)
        saved = authority["meta"].get(rows._result_key(row["admission_id"]))
        if epoch is None:
            if row["status"] != "terminal" or saved is None:
                return None
            return AdmissionResult(record=rows.record(row), result=load_result(json.dumps(saved)))
        if row["owner_epoch"] != epoch or row["generation"] != generation:
            raise rows.stale("admission", admission_id)
        if row["status"] != "terminal":
            return AdmissionResult(record=rows.record(row), result=None)
        if saved is None:
            raise AuthorityRefusal("storage_unavailable")
        return AdmissionResult(record=rows.record(row), result=load_result(json.dumps(saved)))

    def pending_targets(self, *, statuses: frozenset[str], principal_id: str | None = None) -> frozenset[str]:
        return frozenset(r["target_session_id"] for r in self._committed()["admissions"].values()
                         if r["status"] in statuses and (principal_id is None or r["principal_id"] == principal_id))

    def list_unknown(self) -> list[AdmissionRecord]:
        return [rows.record(r) for r in rows.rows_by_seq(self._committed()) if r["status"] == "unknown"]

    def lost_turn_transcript(self, *, session_id: str, event_id: str | None, since: float | None,
                             admission_id: str | None = None) -> LostTurnTranscript:
        authority, tables = self._committed(), self._tables()
        sessions = tables.get("sessions", {})
        chain: list[str] = []
        for start in (session_id, rows.physical_target(self._backend, authority, tables, session_id)):
            if start not in sessions:
                raise AuthorityRefusal("not_found")
            walk = [start]
            for _ in range(100):
                child = self._backend.lineage._chain_step(self._backend.scope, sessions, walk[-1])
                if child is None:
                    break
                if child in walk:
                    raise AuthorityRefusal("admission_conflict")
                walk.append(child)
            else:
                raise AuthorityRefusal("admission_conflict")
            chain += [sid for sid in walk if sid not in chain]
        transcripts = tables.get("messages", {}).get("transcripts", {})
        stored = sorted((row for sid in chain if sid in transcripts for row in transcripts[sid].rows), key=lambda row: row.id)
        anchor = None
        if event_id:
            anchor = next((row for row in reversed(stored) if row.message.role == "user" and row.message.platform_message_id == event_id), None)
        if anchor is None:
            claim = authority["admissions"].get(admission_id) if admission_id else None
            after = claim.get("transcript_mark") if claim is not None else None
            after = -1 if after is None else after
            ran = since is not None and any(row.message.role in ("assistant", "tool") and row.epoch >= since and row.id > after
                                            for row in stored)
            return LostTurnTranscript(anchored=False, anchor_timestamp=None, ran_since=ran, rows=())
        return LostTurnTranscript(anchored=True, anchor_timestamp=anchor.epoch, ran_since=False, rows=tuple(
            LostTurnRow(role=row.message.role, content=row.message.content,
                        tool_calls=json.dumps(list(row.message.tool_calls)) if row.message.tool_calls else None,
                        tool_call_id=row.message.tool_call_id, display_kind=row.message.display_kind, timestamp=row.epoch,
                        active=row.visibility == RowVisibility.ACTIVE, finish_reason=row.message.finish_reason)
            for row in stored if row.id > anchor.id))

    def recovery_attempts(self, *, admission_id: str) -> int:
        found = self._committed()["meta"].get(RECOVERY_ATTEMPTS_PREFIX + admission_id)
        try:
            return int(found) if found is not None else 0
        except (TypeError, ValueError):
            return 1_000

    def owed_recovery_answers(self) -> list[tuple[str, str]]:
        meta = self._committed()["meta"]
        return sorted((key[len(RECOVERY_OWED_PREFIX):], value) for key, value in meta.items()
                      if isinstance(key, str) and key.startswith(RECOVERY_OWED_PREFIX))

    def has_pending(self, *, session_ids: Sequence[str] = (), statuses: frozenset[str] | None = None) -> bool:
        targets = set(session_ids)
        return any((not targets or r["target_session_id"] in targets) and (statuses is None or r["status"] in statuses)
                   for r in self._committed()["admissions"].values())

    def ledger_busy(self, session_id: str, target: str) -> bool:
        authority = self._committed()
        if any(r["target_session_id"] == session_id and r["status"] in _PENDING for r in authority["admissions"].values()):
            return True
        return any(w["session_id"] in (session_id, target) and w["status"] in ("registered", "running", "unknown") for w in authority["workers"].values())

    def webhook_ingress_payloads(self, *, chat_id: str) -> list[Mapping[str, Any]]:
        out = []
        for r in rows.rows_by_seq(self._committed()):
            source = _webhook_source(r["payload"])
            if source is not None and source.get("chat_id") == chat_id:
                out.append(rows.record(r).payload)
        return out

    def unfinalized_webhook_admissions(self) -> list[str]:
        sessions = self._tables().get("sessions", {})
        out = []
        for r in rows.rows_by_seq(self._committed()):
            session = sessions.get(r["target_session_id"])
            if (session is not None and session.get("source") == "webhook" and session.get("ended_at") is None and r["status"] == "terminal"
                    and r["generation"] is not None and _webhook_source(r["payload"]) is not None):
                out.append(r["admission_id"])
        return out

    def terminal_ids(self, admission_ids: Sequence[str]) -> frozenset[str]:
        authority = self._committed()
        live = {i for i in admission_ids if (row := authority["admissions"].get(i)) is not None and row["status"] == "terminal"}
        return frozenset(live | {i for i in admission_ids if rows._terminal_key(i) in authority["meta"]})

    def api_compaction_candidates(self) -> list[str]:
        return [r["admission_id"] for r in rows.rows_by_seq(self._committed()) if _compaction_candidate(r)]

    def compact_api_payloads(self, *, admission_ids: Sequence[str]) -> int:
        state = self._state()
        with state.transaction() as staged:
            compacted = 0
            for admission_id in admission_ids:
                row = staged.authority["admissions"].get(admission_id)
                if row is None or not _compaction_candidate(row):
                    continue
                payload = compacted_api_payload(row["payload"], row["request_id"])
                if payload is not None:
                    row["payload"] = rows.encode(payload)
                    compacted += 1
            return compacted
