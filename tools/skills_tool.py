#!/usr/bin/env python3
"""Skills Tool — list and view skill documents (progressive disclosure). A skill is a directory
holding SKILL.md (YAML frontmatter + instructions) plus optional references/, templates/, assets/,
scripts/. `skills_list` returns name/description only; `skill_view` returns full content and
linked files. Sibling modules (skills_tool_setup / _plugin / _dedup) re-export here."""

import contextvars
import errno
import json
import logging
import os
import stat
import string
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS, is_skill_support_path as _is_skill_support_path)
from tools.skills_tool_setup import (
    SkillReadinessStatus, _build_setup_note, _capture_required_environment_variables,
    _get_required_environment_variables, _is_env_var_persisted, _is_remote_env_backend)
from tools.skills_tool_plugin import (
    MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH, _INJECTION_PATTERNS, _fail, _json,
    _mark_background_review_read, _preprocess_skill, _read_skill_text, _safe_frontmatter,
    _serve_plugin_skill, _serve_skill_file, _truncate_description)
from tools.skills_tool_dedup import dedup_skill_view, reset_skill_view_dedup
from tools.skill_provenance import is_background_review

logger = logging.getLogger(__name__)

# Per-session discovery cache: {cache_key: (signature, timestamp, skills_list)}. Signature =
# every catalog directory's mtime and child-dir names, walked like the catalog itself (an install
# under an existing category such as ``mlops/training/`` moves only that dir's mtime) + the disabled
# set (config-only change, no mtime) + platform; the TTL bounds staleness from in-place SKILL.md
# edits, which no directory signature can see.
_SKILLS_CACHE: dict = {}
_SKILLS_CACHE_TTL_SECONDS = 30.0
# Deepest directory the signature descends to below a root (a skill sits at <cat>/<sub>/<skill>/).
_SIGNATURE_MAX_DEPTH = 6


def clear_skills_cache() -> None:
    """Drop the scanned catalog so the next listing re-scans (explicit reloads must not wait out the TTL)."""
    _SKILLS_CACHE.clear()
    clear_skill_lookup_index()


def _dir_tree_signature(root, sig, depth=_SIGNATURE_MAX_DEPTH) -> None:
    """Append ``(dir, mtime_ns, child dir names)`` for *root* and the dirs below it the catalog walks
    (``iter_skill_index_files``: no excluded dirs, no support dirs of a skill), depth-bounded. The child
    names, not only mtimes: an install and an uninstall inside one coarse mtime tick (1-2 s on some
    filesystems) leave every mtime equal, and the change must still show. ~1 ms for 170 dirs."""
    from agent.skill_utils import EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS
    try:
        mtime = os.stat(root).st_mtime_ns
        with os.scandir(root) as it:
            entries = list(it)
    except OSError:
        return
    names, is_skill = [], False
    for entry in entries:
        with suppress(OSError):
            if entry.is_dir():
                names.append(entry.name)
            elif entry.name == "SKILL.md":
                is_skill = True
    names.sort()
    sig.append((str(root), mtime, tuple(names)))
    if depth <= 0:
        return
    for name in names:
        if name not in EXCLUDED_SKILL_DIRS and not (is_skill and name in SKILL_SUPPORT_DIRS):
            _dir_tree_signature(os.path.join(root, name), sig, depth - 1)


def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """Stat-based change signature over every directory the catalog walks; platform is read via
    ``agent.skill_utils.sys`` so test patches are honored."""
    from agent import skill_utils as _skill_utils
    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig: list = []
    for d in dirs_to_scan:
        _dir_tree_signature(str(d), sig)
    return (tuple(sig), frozenset(disabled), platform)


HERMES_HOME = get_hermes_home()  # all skills live in ~/.hermes/skills/ (seeded from bundled)
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time: the patched ``SKILLS_DIR`` when a patcher changed
    it, else live profile-scoped HERMES_HOME (long-lived runtimes may import before profile set)."""
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


_secret_capture_callback = None
_LOOKUP_HINT = "Use a skill name or relative path within the skills directory."


def skill_lookup_path_error(name: str) -> Optional[str]:
    """Error if lookup *name* could escape the search roots it is joined onto. Windows drive
    paths are rejected too: their ``:`` would be misread as a plugin namespace separator."""
    from tools.path_security import has_traversal_component
    if not isinstance(name, str):
        return "Skill name must be a string."
    win = PureWindowsPath(candidate := name.strip())
    if PurePosixPath(candidate).is_absolute() or win.is_absolute() or win.drive:
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


def load_env() -> dict[str, str]:
    """Snapshot of HERMES_HOME/.env for the post-skill secret-capture diff (same tokenizer that
    installs the profile scope, so a captured value never differs from the served one). A frozen-policy
    worker never opens the profile ``.env``: ``hermes_cli.config.load_env`` owns that short-circuit."""
    from agent.safe_worker_policy import worker_config_snapshot

    if worker_config_snapshot() is not None:
        return {}
    from agent.secret_scope import load_env_file

    return load_env_file(get_hermes_home() / ".env")


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


# A turn-scoped capture callback (the session owner binds one per local-viewer turn): it wins over the
# process-global slot and over the gateway-surface "unsupported" hint, because that turn's viewer can
# answer. A ContextVar, so tool worker threads started with ``propagate_context_to_thread`` inherit it.
_turn_secret_capture = contextvars.ContextVar("turn_secret_capture", default=None)


@contextmanager
def bind_turn_secret_capture(callback):
    token = _turn_secret_capture.set(callback)
    try:
        yield
    finally:
        _turn_secret_capture.reset(token)


def _skill_utils_delegate(attr: str):
    """Lazy call-time delegate to ``agent.skill_utils.<attr>`` (re-export; patches honored)."""
    def _delegate(*args):
        from agent import skill_utils
        return getattr(skill_utils, attr)(*args)
    _delegate.__name__ = _delegate.__qualname__ = attr
    return _delegate


skill_matches_platform = _skill_utils_delegate("skill_matches_platform")
# Offer-time relevance gate (kanban/docker/s6), NOT hard compatibility; explicit loads bypass it.
skill_matches_environment = _skill_utils_delegate("skill_matches_environment")
skill_matches_apps = _skill_utils_delegate("skill_matches_apps")
_parse_frontmatter = _skill_utils_delegate("parse_frontmatter")
_get_disabled_skill_names = _skill_utils_delegate("get_disabled_skill_names")


def check_skills_requirements() -> bool:
    return True  # always available: the directory is created on first use


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """``~/.hermes/skills/mlops/axolotl/SKILL.md`` -> ``"mlops"``; active profile dir first
    (respects test monkeypatching), then skills.external_dirs."""
    dirs_to_check = [_skills_dir()]
    with suppress(Exception):
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    for skills_dir in dirs_to_check:
        with suppress(ValueError):
            if len(parts := skill_path.relative_to(skills_dir).parts) >= 3:
                return parts[0]
    return None


def _parse_tags(tags_value) -> list[str]:
    """Tags from frontmatter: a parsed list, "[a, b]", or "a, b"."""
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]
    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]


def _is_skill_disabled(*names: str, platform: str | None = None) -> bool:
    """Any of *names* disabled in config (one config load)? Platform precedence: explicit arg,
    ``HERMES_PLATFORM``, session ``HERMES_SESSION_PLATFORM``. A globally-disabled skill stays disabled on every platform
    (keep in sync with agent.skill_utils.get_disabled_skill_names)."""
    try:
        from hermes_cli.config import load_config
        skills_cfg = load_config().get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM")
        if not resolved_platform:
            with suppress(Exception):
                from gateway.session_context import get_session_env
                resolved_platform = get_session_env("HERMES_SESSION_PLATFORM") or ""
        platform_disabled = None
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
        disabled = skills_cfg.get("disabled", [])
        return any((platform_disabled is not None and n in platform_disabled) or n in disabled for n in names)
    except Exception:
        return False


def skill_search_dirs() -> tuple[list[tuple[int, Path]], Path]:
    """(``(tier, dir)`` roots in precedence order, active_skills_dir) — the shared
    ``agent.skill_utils.get_skill_search_roots`` order with the live profile dir (dropped if absent)."""
    from agent.skill_utils import TIER_LOCAL, get_skill_search_roots
    active_skills_dir = _skills_dir()
    roots = [(t, d) for t, d in get_skill_search_roots(active_skills_dir)
             if t != TIER_LOCAL or d.exists()]
    return roots, active_skills_dir


def _skill_catalog(*, skip_disabled: bool = False, include_hidden: bool = False) -> list[dict[str, Any]]:
    """Every scanned skill resolved by ``agent.skill_utils.resolve_skill_catalog`` (status /
    load_name / tier / path), visible ones only unless *include_hidden*; cached per session.
    Resolution runs over ALL files first — skill_view ignores platform/disabled gates when
    collecting candidates, so it asks for hidden rows too."""
    from agent.skill_utils import (
        TIER_PROJECT, is_disabled_entry, iter_project_skill_files, iter_skill_index_files, resolve_skill_catalog)
    cache_key = ("with_disabled" if skip_disabled else "filtered", include_hidden)
    disabled = set() if skip_disabled else _get_disabled_skill_names()
    roots, _ = skill_search_dirs()
    signature = _skills_scan_signature([d for _t, d in roots], disabled)
    now = time.monotonic()
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and (now - cached[1]) < _SKILLS_CACHE_TTL_SECONDS:
        # Shallow copies: callers mutate the returned dicts (web_server annotates
        # s["enabled"]/s["usage"]); handing out cached objects would poison the cache.
        return [dict(s) for s in cached[2]]
    scanned = []
    for tier, scan_dir in roots:  # project dirs go through the quarantine chokepoint
        _iter = iter_project_skill_files if tier == TIER_PROJECT else lambda d: iter_skill_index_files(d, "SKILL.md")
        for skill_md in _iter(scan_dir):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                frontmatter, body = _parse_frontmatter(_read_skill_text(skill_md)[:4000])
                description = frontmatter.get("description", "")
                if not description:  # first non-heading body line (a null value stays null)
                    description = next((ln for ln in map(str.strip, body.strip().split("\n"))
                                        if ln and not ln.startswith("#")), description)
                scanned.append({
                    "name": frontmatter.get("name", skill_md.parent.name)[:MAX_NAME_LENGTH],
                    "description": _truncate_description(description),
                    "category": _get_category_from_path(skill_md), "tier": tier, "root": scan_dir,
                    "path": skill_md, "visible": bool(skill_matches_platform(frontmatter)
                                                      and skill_matches_environment(frontmatter)
                                                      and skill_matches_apps(frontmatter))})
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
            except Exception as e:
                logger.debug("Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True)
    skills = [s for s in resolve_skill_catalog(scanned)
              if (s.pop("visible") or include_hidden) and not is_disabled_entry(s, disabled)]
    # Keyed by the signature computed BEFORE the scan: a write racing the scan changes the
    # signature, so the next call re-scans instead of serving a torn result.
    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(s) for s in skills]


def find_all_skills(*, skip_disabled: bool = False) -> list[dict[str, Any]]:
    """Loadable skills (name, description, category): ``name`` is what skill_view() accepts —
    the declared name, or the exact relative path for a same-tier duplicate. Shadowed and
    unloadable copies are left out. ``skip_disabled=True`` ignores disabled state (config UI)."""
    return [{"name": s["load_name"], "description": s["description"], "category": s["category"]}
            for s in _skill_catalog(skip_disabled=skip_disabled) if s["load_name"]]


def sort_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


def _host_skills(fn):
    """*fn* reads this host's skill files: when skills are off (a hosted scope) it answers with the
    refusal instead, for direct callers too, not only registry dispatch."""
    import functools

    @functools.wraps(fn)
    def gated(*args, **kwargs):
        from tools.hosted_tool_policy import host_feature_refusal
        if refusal := host_feature_refusal("skills", fn.__name__):
            return _fail(refusal)
        return fn(*args, **kwargs)
    return gated


@_host_skills
def skills_list(category: str | None = None, task_id: str | None = None) -> str:
    """Tier 1 listing: name + description (+ category) only; ``task_id`` is handler parity."""
    try:
        _skills_dir().mkdir(parents=True, exist_ok=True)
        all_skills = find_all_skills()
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager
            discover_plugins()
            for plugin_skill in get_plugin_manager().list_plugin_skill_metadata():
                frontmatter = plugin_skill.pop("frontmatter", {})
                if not skill_matches_platform(frontmatter) or _is_skill_disabled(plugin_skill["name"]):
                    continue
                all_skills.append(plugin_skill)
        except Exception:
            logger.debug("Plugin skill listing failed", exc_info=True)
        if not all_skills:
            return _json({"success": True, "skills": [], "categories": [],
                          "message": "No skills found in skills/ directory."})
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]
        all_skills = sort_skills(all_skills)
        categories = sorted({s.get("category") for s in all_skills if s.get("category")})
        return _json({
            "success": True, "skills": all_skills, "categories": categories,
            "count": len(all_skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files"})
    except Exception as e:
        return tool_error(str(e), success=False)


def _legacy_portable_skill_name(pm, namespace, bare):
    """Portable plugin skills were named ``agent-plugin-<slug>-<sha8>:<skill>`` before they took the
    plugin's own name; prompts and cron jobs saved since then may still carry that form. Resolved
    through what that exact plugin registered, so a skill it lost to a name clash is not found."""
    if not namespace.startswith("agent-plugin-"):
        return None
    keys = {key for key, loaded in pm._plugins.items()
            if loaded.manifest.portable and loaded.manifest.data_namespace == namespace}
    return next((qualified for qualified, entry in pm._plugin_skills.items()
                 if entry.get("plugin_key") in keys and entry.get("bare_name") == bare), None)


def _resolve_plugin_skill(name, file_path, task_id, preprocess):
    """``plugin:skill`` dispatch: ``(result_json, None)`` when answered, else ``(None,
    local_category_name)`` to fall through to the flat-tree scan — categorized local skills also use
    ``category:skill`` in config/gateway prompts, so the on-disk ``category/skill`` form returns."""
    from agent.skill_utils import is_valid_namespace, parse_qualified_name
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    namespace, bare = parse_qualified_name(name)
    if namespace == "plugin" and ":" in bare:
        # The skills index lists plugin skills under a "plugin" heading, and models write that
        # heading as a prefix ("plugin:agent-plugin-blender-5a80e4ba:blender"). A bare skill name
        # never holds a colon, so the remainder is the real qualified name.
        name = bare
        namespace, bare = parse_qualified_name(name)
    if not is_valid_namespace(namespace):
        return _fail(f"Invalid namespace '{namespace}' in '{name}'. Namespaces must match [a-zA-Z0-9_-]+."), None
    discover_plugins()  # idempotent
    pm = get_plugin_manager()
    if legacy := _legacy_portable_skill_name(pm, namespace, bare):
        name = legacy
        namespace = legacy.partition(":")[0]
    active_memory_provider = None
    try:
        from plugins.memory import _get_active_memory_provider, _prune_inactive_memory_provider_skills
        active_memory_provider = _get_active_memory_provider()
        _prune_inactive_memory_provider_skills(active_memory_provider)
    except Exception as exc:
        logger.debug("Failed pruning inactive memory-provider skills: %s", exc)
    plugin_skill_md = pm.find_plugin_skill(name)
    # Memory providers load through plugins.memory, not the general PluginManager: load the
    # namespaced provider once so its collector can forward its skills into the registry.
    if plugin_skill_md is None and namespace == active_memory_provider:
        try:
            from plugins.memory import load_memory_provider
            load_memory_provider(namespace)
            plugin_skill_md = pm.find_plugin_skill(name)
        except Exception as exc:
            logger.debug("Failed lazy memory-provider skill load for %s: %s", namespace, exc)
    if plugin_skill_md is not None and not plugin_skill_md.exists():
        pm.remove_plugin_skill(name)  # stale registry entry — file deleted out of band
        return _fail(
            f"Skill '{name}' file no longer exists at {plugin_skill_md}. The registry entry "
            f"has been cleaned up — try again after the plugin is reloaded."), None
    if plugin_skill_md is not None:
        return _serve_plugin_skill(
            plugin_skill_md, namespace, bare, file_path=file_path, preprocess=preprocess, session_id=task_id), None
    if available := pm.list_plugin_skills(namespace):  # plugin exists but this specific skill is missing
        return _fail(
            f"Skill '{bare}' not found in plugin '{namespace}'.",
            available_skills=[f"{namespace}:{s}" for s in available],
            hint=f"The '{namespace}' plugin provides {len(available)} skill(s)."), None
    return None, (f"{namespace}/{bare}" if bare else None)  # plugin not found → local scan


def _under_any(path: Path, dirs) -> bool:
    """True when ``path`` (resolved where possible) lives under one of ``dirs``."""
    resolved = path
    with suppress(Exception):
        resolved = path.resolve()
    return any(resolved.is_relative_to(d) for d in dirs)


def _is_package_owned_markdown(path: Path, search_root: Path) -> bool:
    """True when a legacy Markdown candidate belongs to an ancestor directory skill."""
    try:
        relative = path.relative_to(search_root)
    except ValueError:
        return False
    return any(
        (search_root.joinpath(*relative.parts[:depth]) / "SKILL.md").is_file()
        for depth in range(1, len(relative.parts))
    )


# Per-process skill lookup index, one per real directory: {realpath(search dir): _LookupIndex}. A
# lookup used to parse every SKILL.md frontmatter and rglob the whole tree (backups, .git) per
# skill_view call. Keyed by the real directory and holding relative paths, so the profiles whose
# skills dir is a link to one shared tree share one index. Valid while the lookup signature and
# every indexed SKILL.md's (mtime_ns, size) hold; the TTL bounds what neither sees (flat .md files
# inside pruned dirs, skills below signature depth).
_SKILL_LOOKUP_INDEX: dict = {}  # insertion order is build order: the first key is the oldest build
_LOOKUP_INDEX_MAX_DIRS = 64
_SKILL_LOOKUP_LOCK = threading.Lock()
_GLOB_CHARS = frozenset("*?[]")
_ASCII_SWAP = str.maketrans(string.ascii_lowercase + string.ascii_uppercase,
                            string.ascii_uppercase + string.ascii_lowercase)


class _LookupIndex(NamedTuple):
    signature: list
    stats: list
    built_at: float
    skill_mds: list  # relative SKILL.md paths in iter_skill_index_files order; the maps hold indices
    root_skill: Optional[int]  # the search dir's own SKILL.md: its folder name is the caller's dir name
    by_folder: dict
    by_frontmatter: dict
    by_filename: dict  # normcase(<file>.md) -> [(relative path, normcased relative parts)] in rglob order
    linked_dirs: frozenset  # normcased names of symlinks to directories anywhere in the tree
    plain: bool  # every directory of the tree could be listed and matches names by exact case


def clear_skill_lookup_index() -> None:
    """Drop every lookup index so the next skill_view rebuilds (after a skill_manage write)."""
    with _SKILL_LOOKUP_LOCK:
        _SKILL_LOOKUP_INDEX.clear()


def _skill_md_stats(root: str, relative_paths) -> list:
    out = []
    for rel in relative_paths:
        try:
            s = os.stat(os.path.join(root, rel))
            out.append((s.st_mtime_ns, s.st_size))
        except OSError:
            out.append(None)
    return out


def _folds_case(directory: str, names) -> bool:
    """True when *directory* matches names without regard to case (a casefold directory, a
    case-insensitive mount). The literal glob then finds Foo.md for foo; the index, keyed by the
    exact name, would not. Probed with an ASCII-only case change of one entry name: every folding
    filesystem folds ASCII, while a non-ASCII letter may have no partner it folds to. A directory
    whose names have cased letters but no ASCII one cannot be probed and counts as folding.
    Undecodable filenames are opaque on non-strict Linux casefold filesystems: their negative
    probes say nothing about how valid filenames in that directory match."""
    try:
        for name in names:
            name.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return True
    unsure = False
    for name in names:
        other = name.translate(_ASCII_SWAP)
        if other != name:
            # Both spellings as entries means names differ by case here: not folding.
            if other in names:
                return False
            try:
                os.lstat(os.path.join(directory, other))
            except FileNotFoundError:
                return False
            except OSError as exc:
                if exc.errno == errno.ENAMETOOLONG:
                    raise  # the caller's shorter spelling may work
                return True  # inconclusive: retain the original glob
            return True
        unsure = unsure or name.swapcase() != name
    return unsure


def _tree_links(root) -> tuple:
    """``(names of the symlinks to directories, plain)`` for the real directories under *root*, the
    ones rglob walks. *plain* is False when a directory could not be listed or folds case. The glob
    finds a file through a symlinked directory named in its pattern, inside a directory it may enter
    but not list, and under another case in a case-folding directory; the index walk sees none of
    these, so lookups that could differ keep the real glob."""
    linked, plain, stack = set(), True, [str(root)]
    while stack:
        directory = stack.pop()
        names = set()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    names.add(entry.name)
                    try:
                        if entry.is_symlink():
                            if entry.is_dir():
                                linked.add(os.path.normcase(entry.name))
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                    except OSError as exc:
                        if exc.errno == errno.ENAMETOOLONG:
                            raise
                        plain = False
        except OSError as exc:
            if exc.errno == errno.ENAMETOOLONG:
                raise  # canonical spelling can be longer than the caller's usable alias
            plain = False
        if plain and _folds_case(directory, names):
            plain = False
    return frozenset(linked), plain


def _lookup_signature(key: str) -> list:
    """The catalog signature with the search root's own mtime replaced by the root's markdown
    files and symlinks, by name. Every skill load rewrites <root>/.usage.json, which moves that
    mtime, and an index dropped on every load would cost more than the walk it replaces. A lookup
    depends on the root's directories (already in the signature), its ``.md`` files, and which of
    its entries are links (the glob does not walk into a linked directory), nothing else there."""
    signature: list = []
    _dir_tree_signature(key, signature)
    if signature:
        try:
            with os.scandir(key) as entries:
                names = tuple(sorted((e.name, e.is_symlink()) for e in entries
                                     if e.name.endswith(".md") or e.is_symlink()))
        except OSError:
            names = None
        signature[0] = (signature[0][0], names, signature[0][2])
    return signature


def _frontmatter_name(skill_md) -> Optional[str]:
    """The frontmatter ``name`` of a regular SKILL.md, else None. Opened without blocking and
    checked on the open file: a FIFO or a device in its place must not hang the build, which runs
    under the lookup lock. Any failure to read means no name, as in ``_safe_frontmatter``."""
    content, fd = None, None
    try:
        fd = os.open(skill_md, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.set_blocking(fd, True)
            # The outer finally owns fd, even if text-wrapper construction fails.
            with open(fd, encoding="utf-8-sig", errors="replace", closefd=False) as handle:
                content = handle.read()
    except Exception as exc:
        content = None  # a read or wrapper-close failure invalidates the entire read
        if isinstance(exc, OSError) and exc.errno == errno.ENAMETOOLONG:
            raise  # the index must fall back rather than cache an omitted alias
    finally:
        if fd is not None:
            try:
                os.close(fd)  # once only: an error can still mean this fd was released
            except OSError:
                content = None
    if content is None:
        return None
    frontmatter = _safe_frontmatter(content=content)
    name = frontmatter.get("name") if isinstance(frontmatter, dict) else None
    return name if isinstance(name, str) else None


def _build_lookup_index(key: str) -> _LookupIndex:
    """Signature and stats first, then read: a write racing the build shows as a change next call.
    The TTL runs from the end of the build, so a slow build is not stale the moment it is done."""
    from agent.skill_utils import iter_skill_index_files
    root = Path(key)
    signature = _lookup_signature(key)
    skill_mds = [os.path.relpath(smd, key) for smd in iter_skill_index_files(root, "SKILL.md")]
    stats = _skill_md_stats(key, skill_mds)
    root_skill = None
    by_folder: dict = {}
    by_frontmatter: dict = {}
    for i, rel in enumerate(skill_mds):
        folder = os.path.dirname(rel)
        if folder:
            by_folder.setdefault(os.path.basename(folder), []).append(i)
        else:
            root_skill = i
        if (fm_name := _frontmatter_name(os.path.join(key, rel))) is not None:
            by_frontmatter.setdefault(fm_name, []).append(i)
    # Same unpruned walk as today's rglob("<name>.md"), once per build; filters run per lookup.
    by_filename: dict = {}
    for md in root.rglob("*.md"):
        parts = md.relative_to(root).parts
        normed = tuple(os.path.normcase(part) for part in parts)
        by_filename.setdefault(normed[-1], []).append((os.path.join(*parts), normed))
    linked_dirs, plain = _tree_links(key)
    return _LookupIndex(signature, stats, time.monotonic(), skill_mds, root_skill, by_folder, by_frontmatter,
                        by_filename, linked_dirs, plain)


def _index_is_current(key: str, index: _LookupIndex) -> bool:
    if time.monotonic() - index.built_at >= _SKILLS_CACHE_TTL_SECONDS:
        return False
    return _lookup_signature(key) == index.signature and _skill_md_stats(key, index.skill_mds) == index.stats


def _lookup_index(search_dir: Path, checked: Optional[dict] = None) -> Optional[_LookupIndex]:
    """The current index of *search_dir*'s real directory. *checked* holds the indexes one lookup
    has already validated, so two search dirs that are the same directory are checked once."""
    key = os.path.realpath(search_dir)
    if checked is not None and key in checked:
        return checked[key]
    index = _SKILL_LOOKUP_INDEX.get(key)
    if index is None or not _index_is_current(key, index):
        with _SKILL_LOOKUP_LOCK:  # one build per change, not one per concurrent gateway thread
            current = _SKILL_LOOKUP_INDEX.get(key)
            # While this thread waited, a clear may have dropped the entry or another thread may
            # have rebuilt it: a rebuilt one is used only if it still holds.
            if current is None or current is index or not _index_is_current(key, current):
                try:
                    current = _build_lookup_index(key)
                except OSError as exc:
                    if exc.errno != errno.ENAMETOOLONG:
                        raise
                    # A shorter caller spelling can still work. Do not publish a partial index.
                    _SKILL_LOOKUP_INDEX.pop(key, None)
                    if checked is not None:
                        checked[key] = None
                    return None
                _SKILL_LOOKUP_INDEX.pop(key, None)
                while len(_SKILL_LOOKUP_INDEX) >= _LOOKUP_INDEX_MAX_DIRS:
                    del _SKILL_LOOKUP_INDEX[next(iter(_SKILL_LOOKUP_INDEX))]
                _SKILL_LOOKUP_INDEX[key] = current
            index = current
    if checked is not None:
        checked[key] = index
    return index


def _still_walked(skill_md: Path, search_dir: Path) -> bool:
    """False when *skill_md* now sits in a support dir (references, templates, ...) of a directory
    that holds a SKILL.md: the catalog walk skips those. A directory below the signature depth can
    gain a SKILL.md without the index noticing, so this rule is read from disk for every cached hit."""
    from agent.skill_utils import SKILL_SUPPORT_DIRS
    try:
        parts = skill_md.parent.relative_to(search_dir).parts
    except ValueError:
        return True
    at = search_dir
    for part in parts:
        if part in SKILL_SUPPORT_DIRS:
            owner = at / "SKILL.md"
            if os.path.lexists(owner) and not os.path.isdir(owner):
                return False
        at = at / part
    return True


def _indexed_skill_matches(search_dir: Path, name: str, checked: Optional[dict] = None):
    """``(skill_mds, flat_mds)`` exactly as the tree walk and ``rglob(f"{name}.md")`` find them, or
    None when *name* holds glob characters or odd segments (those keep today's walk)."""
    parts = name.split("/")
    if _GLOB_CHARS.intersection(name) or "\\" in name or any(p in ("", ".", "..") for p in parts):
        return None
    index = _lookup_index(search_dir, checked)
    if index is None or not index.signature or index.signature[0][1] is None:
        # The caller may reach an open-directory alias whose canonical parents are inaccessible.
        # An unreadable canonical root is not evidence that its caller-spelled tree is empty.
        return None
    if index.linked_dirs and len(os.fsencode(os.path.realpath(search_dir))) > len(os.fsencode(search_dir)):
        # The catalog follows directory links but the flat-file walk does not. A canonical
        # walk can silently skip a too-long path inside one: only the shorter caller walk
        # can decide which frontmatter aliases exist in that subtree.
        return None
    hits = {*index.by_folder.get(name, ()), *index.by_frontmatter.get(name, ())}
    if index.root_skill is not None and search_dir.name == name:
        hits.add(index.root_skill)
    hits = sorted(hits)
    if any(os.path.islink(search_dir / index.skill_mds[i])
           and (search_dir / index.skill_mds[i]).parent.name != name for i in hits):
        # A canonical-root read can follow a final-file link that exceeds the caller path's
        # symlink budget. Let the original walk decide frontmatter-only hits on such files.
        return None
    want = tuple(os.path.normcase(p) for p in parts[:-1]) + (os.path.normcase(parts[-1] + ".md"),)
    # lexists per hit, as the literal glob does: a file deleted where the signature cannot see it, or
    # one in a directory that can be listed but not entered, is not a candidate.
    flat = [md for rel, normed in index.by_filename.get(want[-1], ())
            if normed[-len(want):] == want and os.path.lexists(md := search_dir / rel)]
    if not index.plain or (len(parts) > 1 and (len(flat) > 1 or index.linked_dirs.intersection(want[:-1]))):
        # A qualified pattern is matched under each directory in walk order, the index lists files in
        # the order their own directory is walked: with several copies only the real glob has today's
        # order. The glob also follows a symlinked directory named in the pattern, finds a file in a
        # directory it cannot list, and matches under another case in a case-folding directory; the
        # index walk sees none of these.
        flat = list(search_dir.rglob(f"{name}.md"))
    return [smd for i in hits if os.path.lexists(smd := search_dir / index.skill_mds[i])
            and _still_walked(smd, search_dir)], flat


def _collect_skill_candidates(name, local_category_name, all_dirs):
    """ALL (skill_dir, skill_md) candidates across every dir and lookup strategy (direct path,
    recursive by dir / frontmatter name, legacy flat <name>.md), deduped by resolved path.
    Every copy is returned so the caller resolves them via ``agent.skill_utils.pick_skill_candidate``
    (cross-tier precedence; a same-tier tie of different skills is refused, never guessed)."""
    from agent.skill_utils import iter_skill_index_files
    candidates: list[tuple[Optional[Path], Path]] = []
    seen_md: set = set()
    checked: dict = {}  # lookup indexes already validated for this call, by real directory

    def _record(sd: Optional[Path], smd: Path) -> None:
        key = smd
        with suppress(Exception):
            key = smd.resolve()
        if key not in seen_md:
            seen_md.add(key)
            candidates.append((sd, smd))

    def _record_direct(direct_path: Path, search_root: Path) -> None:  # "mlops/axolotl" / "axolotl" or its flat .md sibling
        flat = direct_path.with_suffix(".md")
        if not _is_skill_support_path(direct_path) and direct_path.is_dir() and (direct_path / "SKILL.md").exists():
            _record(direct_path, direct_path / "SKILL.md")
        elif (flat.exists() and not _is_skill_support_path(flat)
              and not _is_package_owned_markdown(flat, search_root)):
            _record(None, flat)

    for search_dir in all_dirs:
        for direct in filter(None, (name, local_category_name)):  # "p:x" with no plugin p → "p/x"
            _record_direct(search_dir / direct, search_dir)
        # Recursive by directory name plus frontmatter `name:` — skills_list()
        # exposes the frontmatter name, so skill_view(name) must accept it too.
        if (indexed := _indexed_skill_matches(search_dir, name, checked)) is not None:
            skill_hits, flat_hits = indexed
        else:
            skill_hits = [
                smd for smd in iter_skill_index_files(search_dir, "SKILL.md")
                if smd.parent.name == name or _safe_frontmatter(smd).get("name") == name]
            flat_hits = search_dir.rglob(f"{name}.md")
        for found_skill_md in skill_hits:
            _record(found_skill_md.parent, found_skill_md)
        # Legacy flat <name>.md anywhere under the dir. Markdown owned by an ancestor
        # directory skill loads through file_path and must not shadow a real skill.
        for found_md in flat_hits:
            if (found_md.name != "SKILL.md" and not _is_skill_support_path(found_md)
                    and not _is_package_owned_markdown(found_md, search_dir)):
                _record(None, found_md)
    return candidates


# (support dir, globs, recursive, files only) — order is the linked_files key order.
_LINKED_FILE_SPECS = (
    ("references", ["*.md"], False, False),
    ("templates", ["*.md", "*.py", "*.yaml", "*.yml", "*.json", "*.tex", "*.sh"], True, False),
    ("assets", ["*"], True, True),
    ("scripts", ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"], False, False))


def _skill_linked_files(skill_dir: Optional[Path]) -> dict:
    """references/templates/assets/scripts of a directory skill (empty groups dropped)."""
    files: dict = {}
    for sub, globs, recursive, files_only in _LINKED_FILE_SPECS if skill_dir else ():
        base = skill_dir / sub
        found = [
            f.relative_to(skill_dir).as_posix() for g in globs if base.exists()
            for f in (base.rglob(g) if recursive else base.glob(g))
            if not files_only or f.is_file()]
        if found:
            files[sub] = found
    return files


def _skill_readiness(frontmatter: dict[str, Any], skill_name: str) -> tuple[dict, dict]:
    """Resolve required env vars / credential files (prompting for secrets where the surface
    allows) and register what's available for sandboxes. Returns ``(fields, extras)``: fields go
    before the readiness extras in the skill_view result — key order is tool output."""
    required_env_vars = _get_required_environment_variables(frontmatter)
    from tools.terminal_scope import terminal_env
    backend = str(terminal_env("TERMINAL_ENV", "local")).strip().lower() or "local"
    env_snapshot = load_env()
    missing_required_env_vars = [
        e for e in required_env_vars
        if not e.get("optional") and not _is_env_var_persisted(e["name"], env_snapshot)]
    capture_result = _capture_required_environment_variables(skill_name, missing_required_env_vars)
    if missing_required_env_vars:  # re-read: a successful capture persisted into .env
        env_snapshot = load_env()
    still_missing = set(capture_result["missing_names"])
    remaining = [
        e["name"] for e in required_env_vars if not e.get("optional")
        and (e["name"] in still_missing or not _is_env_var_persisted(e["name"], env_snapshot))]
    setup_needed = bool(remaining)
    # Only vars actually set pass through to sandboxed execution (execute_code, terminal).
    refused: list[str] = []
    if available_env_names := [e["name"] for e in required_env_vars if e["name"] not in remaining]:
        try:
            from tools.env_passthrough import register_env_passthrough
            refused = register_env_passthrough(available_env_names)
        except Exception:
            logger.debug("Could not register env passthrough for skill %s", skill_name, exc_info=True)
    # Credential files for remote sandboxes: existing host files are registered,
    # missing ones flag setup_needed.
    required_cred_files_raw = frontmatter.get("required_credential_files", [])
    missing_cred_files: list = []
    if isinstance(required_cred_files_raw, list) and required_cred_files_raw:
        try:
            from tools.credential_files import register_credential_files
            missing_cred_files = register_credential_files(required_cred_files_raw)
            setup_needed = setup_needed or bool(missing_cred_files)
        except Exception:
            logger.debug("Could not register credential files for skill %s", skill_name, exc_info=True)
    status = SkillReadinessStatus.SETUP_NEEDED if setup_needed else SkillReadinessStatus.AVAILABLE
    fields = {
        "required_environment_variables": required_env_vars, "required_commands": [],
        "missing_required_environment_variables": remaining,
        "missing_credential_files": missing_cred_files, "missing_required_commands": [],
        "setup_needed": setup_needed, "setup_skipped": capture_result["setup_skipped"],
        "readiness_status": status.value}
    extras: dict = {}
    if setup_help := next((e["help"] for e in required_env_vars if e.get("help")), None):
        extras["setup_help"] = setup_help
    if capture_result["gateway_setup_hint"]:
        extras["gateway_setup_hint"] = capture_result["gateway_setup_hint"]
    if refused:  # otherwise the skill's commands fail auth with no hint why
        extras["env_passthrough_refused"] = (
            f"{', '.join(refused)} is a Hermes-managed credential and is never passed to the "
            "terminal or execute_code; commands in this skill that need it will not see it.")
    missing_items = [f"env ${n}" for n in remaining] + [f"file {p}" for p in missing_cred_files]
    if setup_needed and (setup_note := _build_setup_note(status, missing_items, setup_help)):
        if _is_remote_env_backend(backend):
            setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
        extras["setup_note"] = setup_note
    return fields, extras


def _owning_search_dir(skill_md: Path, all_dirs) -> Optional[Path]:
    """Most specific search dir containing *skill_md*, compared lexically: a symlinked entry
    belongs to the root that exposes it, not to the root its target lives in."""
    owners = [Path(d) for d in all_dirs if skill_md.is_relative_to(d)]
    return max(owners, key=lambda d: len(d.parts), default=None)


def locate_skill(name: str, local_category_name: Optional[str], roots):
    """Unique on-disk skill for *name* over ``(tier, dir)`` *roots*: cross-tier precedence
    (project > local > create_dir > external, shadowed copies logged), same-tier collision refusal,
    same-root identical-copy ranking, quarantine gate, not-found listing. ``(error_json, skill_dir,
    skill_md)``; skill_md set iff no error."""
    from agent.skill_utils import AMBIGUOUS_SKILL_PREFIX, TIER_PROJECT, pick_skill_candidate, skill_candidate_rank
    all_dirs = [d for _t, d in roots]
    if not all_dirs:
        return _fail(
            "Skills directory does not exist yet. It will be created on first install."), None, None
    candidates = _collect_skill_candidates(name, local_category_name, all_dirs)
    tier_of = {d: t for t, d in roots}
    if len(candidates) > 1:
        owned = [(c, _owning_search_dir(c[1], all_dirs)) for c in candidates]
        won, contenders = pick_skill_candidate([
            (tier_of[root], str(root), skill_candidate_rank(c[1], root), c[1]) for c, root in owned])
        if won is not None:
            if dropped := [str(smd) for i, (_sd, smd) in enumerate(candidates) if i != won]:
                logger.info("Skill '%s' resolved to %s by precedence (project > local > create_dir > "
                            "external_dirs, then identical same-root copies); not loaded: %s",
                            name, candidates[won][1], "; ".join(dropped))
            candidates = [candidates[won]]
        else:
            candidates = [candidates[i] for i in contenders]
    if len(candidates) > 1:
        paths = [str(smd) for _, smd in candidates]
        # A root-level copy's path IS the bare name, so it can't disambiguate itself.
        load_names = sorted({_owned_relative(sd, smd, all_dirs) for sd, smd in candidates} - {name})
        logger.warning("Skill name collision for '%s': %d candidates — %s", name, len(candidates), "; ".join(paths))
        return _fail(
            f"{AMBIGUOUS_SKILL_PREFIX}'{name}': "
            + (f"use one of {', '.join(load_names)}. " if load_names else "")
            + f"{len(candidates)} different skills share this name in the same skills directory tier; "
            "refusing to guess.",
            matches=paths, load_names=load_names,
            hint="Pass the exact relative path instead of the bare name, or rename one of the "
            "colliding skills so each name is unique."), None, None
    skill_dir, skill_md = candidates[0] if candidates else (None, None)
    # Quarantine gate: a project-tier skill with a dangerous scan verdict must not
    # load even by explicit name (same chokepoint the index and skills_list use).
    if skill_md is not None and tier_of.get(_owning_search_dir(skill_md, all_dirs)) == TIER_PROJECT:
        from agent.skill_utils import is_quarantined_project_skill
        if is_quarantined_project_skill(skill_md):
            return _fail(
                f"Project skill '{name}' is quarantined: the security scan flagged its content as "
                "dangerous. It will not load until the repo's skill content changes and passes a re-scan.",
                hint="Inspect the skill in the repo checkout, or untrust the repo with "
                "`hermes skills untrust`."), None, None
    if not skill_md or not skill_md.exists():
        available = [s["name"] for s in sort_skills(find_all_skills())[:20]]
        return _fail(f"Skill '{name}' not found.", available_skills=available,
                     hint="Use skills_list to see all available skills"), None, None
    return None, skill_dir, skill_md


def _owned_relative(skill_dir: Optional[Path], skill_md: Path, all_dirs) -> str:
    """Exact load path of a candidate: its skill dir (or flat ``.md`` stem) relative to its root."""
    root = _owning_search_dir(skill_md, all_dirs)
    target = skill_dir if skill_dir is not None else skill_md.with_suffix("")
    return target.relative_to(root).as_posix() if root is not None else str(target)


def _log_security_warnings(name: str, skill_md: Path, content: str, all_dirs, active_skills_dir):
    """Warn (never block) when loaded from outside the trusted dirs (project + local + external)
    and/or when common prompt-injection patterns appear. The check is on the RESOLVED path:
    every candidate is built as ``<search_dir>/...`` so a lexical test can never fire, and a
    SKILL.md symlinked to a file outside every root is exactly what this guards against."""
    trusted_dirs = [active_skills_dir.resolve()]
    with suppress(Exception):
        trusted_dirs.extend(d.resolve() for d in all_dirs)
    warnings = []
    if not _under_any(skill_md, trusted_dirs):
        warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        warnings.append("skill content contains patterns that may indicate prompt injection")
    if warnings:
        logger.warning("Skill security warning for '%s': %s", name, "; ".join(warnings))


@_host_skills
def skill_view(
    name: str, file_path: str | None = None, task_id: str | None = None, preprocess: bool = True) -> str:
    """View a skill (SKILL.md) or a file within its directory, as JSON. ``name`` is a skill name
    or path ("axolotl", "03-fine-tuning/axolotl"); "<plugin>:<skill>" resolves plugin-provided
    skills. ``preprocess`` applies the configured SKILL.md template / inline shell rendering;
    slash/preload callers render the message themselves."""
    try:
        # Validate before the ':' dispatch so a Windows drive path (C:\skills\foo) can't be
        # reinterpreted as a plugin namespace.
        if lookup_error := skill_lookup_path_error(name):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        local_category_name: str | None = None
        if ":" in name:  # plugin registry; bare names use the flat-tree scan below
            served, local_category_name = _resolve_plugin_skill(name, file_path, task_id, preprocess)
            if served is not None:
                return served
        # The fall-through form (namespace/bare) joins onto each search dir too; re-validate it
        # since `bare` is not namespace-checked.
        if local_category_name and (lookup_error := skill_lookup_path_error(local_category_name)):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        roots, active_skills_dir = skill_search_dirs()
        all_dirs = [d for _t, d in roots]
        error, skill_dir, skill_md = locate_skill(name, local_category_name, roots)
        if error is not None:
            return error
        try:  # read once — reused for platform check and main content
            content = _read_skill_text(skill_md)
        except Exception as e:
            return _fail(f"Failed to read skill '{name}': {e}")
        _log_security_warnings(name, skill_md, content, all_dirs, active_skills_dir)
        frontmatter = _safe_frontmatter(content=content)
        if not skill_matches_platform(frontmatter):
            return _fail(f"Skill '{name}' is not supported on this platform.", readiness_status=SkillReadinessStatus.UNSUPPORTED.value)
        resolved_name = frontmatter.get("name", skill_md.parent.name)
        # Disabled by declared name, or by the exact path that IS this copy's load name (a same-tier
        # duplicate's row) — whatever alias reached it; an unrelated copy at that path elsewhere is not.
        rel = _owned_relative(skill_dir, skill_md, all_dirs)
        if _is_skill_disabled(resolved_name, rel) and (_is_skill_disabled(resolved_name) or any(
                e["path"] == skill_md and e["load_name"] == rel for e in _skill_catalog(skip_disabled=True, include_hidden=True))):
            return _fail(f"Skill '{resolved_name}' is disabled. Enable it with `hermes skills` or inspect the files directly on disk.")
        if file_path and skill_dir:
            return _serve_skill_file(
                skill_dir, file_path, name, list_available=True, mark_read=True,
                hint="Use a relative path within the skill directory")
        # tags/related_skills: metadata.hermes.* (agentskills.io) first, then top-level.
        metadata = frontmatter.get("metadata")
        hermes_meta = (metadata.get("hermes", {}) or {}) if isinstance(metadata, dict) else {}
        tags, related_skills = (
            _parse_tags(hermes_meta.get(k) or frontmatter.get(k, "")) for k in ("tags", "related_skills"))
        linked_files = _skill_linked_files(skill_dir)
        try:
            rel_path = str(skill_md.relative_to(active_skills_dir))
        except ValueError:  # external skill — relative to its own parent dir
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get("name", skill_md.stem if not skill_dir else skill_dir.name)
        readiness, readiness_extras = _skill_readiness(frontmatter, skill_name)
        rendered_content = content if not preprocess else _preprocess_skill(
            content, skill_dir, task_id, "Could not preprocess skill content for %s", skill_name)
        # ── pm tool deps (`deps: [ffmpeg]` frontmatter) ──────────────
        # Loading the skill IS the activation moment: ensure each declared
        # pm package now so the skill's commands work when the model runs
        # them. Failure never blocks the skill content — the note carries
        # the remedy.
        deps_note = None
        declared_deps = frontmatter.get("deps") or []
        if isinstance(declared_deps, str):
            declared_deps = [declared_deps]
        if isinstance(declared_deps, list) and declared_deps:
            failed_deps = []
            for dep in [str(d).strip() for d in declared_deps if str(d).strip()]:
                try:
                    import pm

                    pm.ensure(dep)
                except Exception as exc:
                    failed_deps.append(f"{dep}: {exc}")
            if failed_deps:
                deps_note = (
                    "Tool dependencies could not be installed — "
                    + "; ".join(failed_deps)
                    + ". Run `hermes pm install "
                    + " ".join(str(d) for d in declared_deps)
                    + "` and reload."
                )

        result = {
            "success": True, "name": skill_name, "description": frontmatter.get("description", ""),
            "tags": tags, "related_skills": related_skills, "content": rendered_content,
            "path": rel_path, "skill_dir": str(skill_dir) if skill_dir else None,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'" if linked_files else None,
            **readiness,
            **readiness_extras}
        if deps_note:
            result["deps_note"] = deps_note
        _mark_background_review_read(skill_md)
        if frontmatter.get("compatibility"):  # agentskills.io optional fields
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        return _json(result)
    except Exception as e:
        return tool_error(str(e), success=False)


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Load a skill's SKILL.md, or one of its linked references, templates or scripts. The first call returns SKILL.md plus a 'linked_files' dict; call again with file_path to read one.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form 'plugin:skill'.",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list", toolset="skills", schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(category=args.get("category"), task_id=kw.get("task_id")),
    check_fn=check_skills_requirements, emoji="📚")


def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count/use on success (best-effort). Repeat-view dedup
    mirrors read_file's unchanged-stub: a result byte-identical to one already served for the
    same skill file in this session returns a short stub (cache cleared on context compression)."""
    name = args.get("name", "")
    task_id = kw.get("task_id")
    # The background-review fork shares the parent's task_id (prefix-cache parity). A stub there
    # (a) skips the read-mark its read-before-write guard requires and (b) lets it patch from a
    # possibly-pruned transcript copy (#95976). No dedup in the fork; None also keeps its views
    # out of the parent's bucket.
    dedup_task_id = None if is_background_review() else task_id
    result = skill_view(name, file_path=args.get("file_path"), task_id=task_id)
    with suppress(Exception):
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            if (stub := dedup_skill_view(dedup_task_id, name, args.get("file_path"), parsed, result)) is not None:
                return stub
            if resolved := parsed.get("name") or name:  # qualified forms return the canonical name
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # Viewing is actively loading the skill to act on it — that counts as use
                # (the curator's stale timer keys off last_used_at).
                bump_use(str(resolved), task_id=kw.get("task_id"), session_id=kw.get("session_id"))
    return result


registry.register(
    name="skill_view", toolset="skills", schema=SKILL_VIEW_SCHEMA, handler=_skill_view_with_bump,
    check_fn=check_skills_requirements, emoji="📚")
