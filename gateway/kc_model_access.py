"""KC: what a non-admin may change with ``/model`` and ``/reasoning``.

Policy (Nathan, 2026-10-08): every allowed chat user may change the model and reasoning level for
their own conversation. Changes that reach every user of the agent (``--global`` and the reasoning
display toggle) stay admin-only, and when the platform has a curated ``model_picker`` a non-admin
may only switch to one of its rows. Listed admins keep the full behavior.

"Admin" is an explicitly listed admin of the profile whose bot received the message: its
slash-gating admins (``allow_admin_from`` / ``group_allow_admin_from``) or its ``model_admin_from``
list; see ``is_admin``.
"""

from __future__ import annotations

from typing import Any, List, Optional

GLOBAL_DENIED = (
    "⛔ `--global` changes this agent for everyone, so only an admin can use it. "
    "Run the same command without `--global` to change only this conversation."
)
DISPLAY_DENIED = (
    "⛔ Showing or hiding reasoning changes this agent for everyone, so only an admin can do it."
)
NOT_CURATED = (
    "⛔ That model is not on this agent's list. Choose one of: {labels}. "
    "Send `/model` with no name to open the picker."
)

DISPLAY_TOGGLES = frozenset({"show", "hide", "on", "off"})


def restricted(runner: Any, source: Any) -> bool:
    """True when ``source`` is a Slack chat user who is not a listed admin of the receiving profile.

    Scoped to Slack: the request was for Slack users, and other surfaces (Telegram, Desktop, CLI,
    API) keep their stock behavior."""
    platform = getattr(source, "platform", None)
    value = getattr(platform, "value", platform)
    if str(value or "").lower() != "slack":
        return False
    return not is_admin(runner, source)


def is_admin(runner: Any, source: Any) -> bool:
    """True only for an EXPLICITLY listed admin of the receiving profile.

    Listed means the slash-gating admin list for the chat's scope (``allow_admin_from`` /
    ``group_allow_admin_from``) or the KC ``model_admin_from`` list on the platform. Unlike
    ``SlashAccessPolicy.is_admin`` an ungated profile does NOT make everyone an admin here, so a
    profile with no admin list still keeps agent-wide changes away from ordinary chat users."""
    from gateway.slash_access import _coerce_id_list, platform_extra_for_runner_source, policy_for_runner_source

    user_id = str(getattr(source, "user_id", None) or "").strip()
    if not user_id:
        return False
    try:
        policy = policy_for_runner_source(runner, source)
        if policy.enabled and user_id in policy.admin_user_ids:
            return True
    except Exception:
        pass
    try:
        extra = platform_extra_for_runner_source(runner, source)
    except Exception:
        return False
    return user_id in _coerce_id_list((extra or {}).get("model_admin_from"))


def curated_entries(runner: Any, source: Any) -> List[dict]:
    """The receiving profile's ``model_picker`` rows for ``source``'s platform (``[]`` if unset).

    Same normalization as the Slack adapter: rows need label, model and provider; a repeated
    ``(provider, model)`` pair is dropped."""
    from gateway.slash_access import platform_extra_for_runner_source

    try:
        extra = platform_extra_for_runner_source(runner, source)
    except Exception:
        return []
    raw = extra.get("model_picker") if isinstance(extra, dict) else None
    if not isinstance(raw, list):
        return []
    entries: List[dict] = []
    seen: set = set()
    for row in raw[:100]:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        model = str(row.get("model") or "").strip()
        provider = str(row.get("provider") or "").strip()
        if not (label and model and provider) or (provider, model) in seen:
            continue
        seen.add((provider, model))
        entries.append({"label": label, "model": model, "provider": provider})
    return entries


def entry_for_label(entries: List[dict], target: str) -> Optional[dict]:
    """The curated row whose label matches ``target`` (case- and space-insensitive), if any.

    Lets anyone type ``/model Sonnet 5.5`` instead of the provider slug and model ID."""
    wanted = " ".join(str(target or "").split()).lower()
    if not wanted:
        return None
    for entry in entries:
        if " ".join(entry["label"].split()).lower() == wanted:
            return entry
    return None


def is_curated_route(entries: List[dict], provider: str, model: str) -> bool:
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    return any(e["provider"] == provider and e["model"] == model for e in entries)


def not_curated_message(entries: List[dict]) -> str:
    return NOT_CURATED.format(labels=", ".join(e["label"] for e in entries))
