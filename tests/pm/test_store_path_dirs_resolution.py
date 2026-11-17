"""The provisioned PATH resolves the tool store once per call, not once per package.

Every terminal command builds its PATH through ``pm.install._store_path_dirs``. Resolving the
store root walks parent directories for an install stamp, so resolving it for each package (five
times each) made every command spawn pay for the whole store again.
"""

from __future__ import annotations

import json

from pm import install
from pm import paths
from pm import registry
from pm.install import _installed_location, _store_path_dirs
from pm.lock import Facts, Lockfile
from pm.store import current_target


def _seed_store(tmp_path, monkeypatch, names=("node", "npm")):
    """Installed store packages fabricated the way pm records them."""
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    lock_dir = tmp_path / "repo-lock"
    lock_dir.mkdir()
    lockfile_path = lock_dir / "lock.json"
    monkeypatch.setattr(paths, "lockfile_path", lambda: lockfile_path)
    lock = Lockfile(lockfile_path)
    for name in names:
        lock.set_pin(name, "1", {"any": {"url": "x", "sha256": "0" * 64}})
    lock.save()

    store = paths.store_root()
    facts_path = paths.facts_path()
    facts_path.parent.mkdir(parents=True, exist_ok=True)
    facts_path.write_text(json.dumps({"schema": 1, "packages": {}}), encoding="utf-8")
    facts = Facts(facts_path)
    for name in names:
        entry = f"{name}-1"
        entry_dir = store / entry
        entry_dir.mkdir(parents=True, exist_ok=True)
        binary = registry.get_package(name).binary(entry_dir, current_target())
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"installed tool")
        facts.record(name, "1", entry, {"PATH": ["{store}/" + entry]}, store_root=store,
                     target=current_target(), artifacts=["0" * 64])
    return Lockfile(lockfile_path)


def _count_store_root(monkeypatch):
    calls = []
    real = paths.store_root

    def counted():
        calls.append(1)
        return real()

    monkeypatch.setattr(paths, "store_root", counted)
    return calls


def test_store_path_dirs_resolve_the_store_once_per_call(tmp_path, monkeypatch):
    _seed_store(tmp_path, monkeypatch)
    expected = _store_path_dirs()
    assert any(d.endswith("node-1") for d in expected) and any(d.endswith("npm-1") for d in expected), expected

    calls = _count_store_root(monkeypatch)
    assert _store_path_dirs() == expected
    # One for the store, one inside writable_store_root: independent of how many packages the lock names.
    assert len(calls) <= 2, f"store root resolved {len(calls)} times for one PATH"


def test_installed_location_resolves_the_store_once_per_root(tmp_path, monkeypatch):
    lockfile = _seed_store(tmp_path, monkeypatch)
    package, target = registry.get_package("node"), current_target()
    expected = _installed_location(package, lockfile, target)
    assert expected is not None

    calls = _count_store_root(monkeypatch)
    facts, store = _installed_location(package, lockfile, target)
    assert (facts.path, store.root) == (expected[0].path, expected[1].root)
    assert len(calls) <= 2, f"store root resolved {len(calls)} times for one package"

    del calls[:]
    facts, store = _installed_location(package, lockfile, target, roots=(expected[1].root,))
    assert (facts.path, store.root) == (expected[0].path, expected[1].root)
    assert not calls, "an explicit root needs no store resolution"


def test_a_path_with_no_package_resolves_no_store(tmp_path, monkeypatch):
    """With nothing to locate the PATH build never resolved the store, so a broken install stamp
    could not fail it. It still must not."""
    _seed_store(tmp_path, monkeypatch, names=())
    calls = _count_store_root(monkeypatch)
    assert _store_path_dirs() == []
    assert not calls, f"store root resolved {len(calls)} times for an empty PATH"


def test_store_path_dirs_are_the_recorded_dirs_of_each_located_package(tmp_path, monkeypatch):
    """The PATH is what each package's own lookup (no shared roots) records, dependents first."""
    lockfile = _seed_store(tmp_path, monkeypatch)
    target = current_target()
    store = paths.store_root()
    expected = []
    for name in ("npm", "node"):  # npm depends on node, so it comes first
        facts, found = _installed_location(registry.get_package(name), lockfile, target)
        assert (facts.path, found.root) == (store / "facts.json", store)
        expected += [install.native(d) for d in facts.env_for(name, found.root)["PATH"]]
    assert len(expected) == 2 and expected[0].endswith("npm-1") and expected[1].endswith("node-1"), expected
    assert _store_path_dirs() == expected
