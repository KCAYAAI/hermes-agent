"""Two turn threads meeting one home's provider plugin at once both get its providers, and keep them.

One process serves many profile homes on threads (multiplex gateway, Desktop ``serve``). A new Desktop chat
resolves its provider from several threads at once, so a secondary profile's first lookups raced: the scan
that lost found the plugin module already in ``sys.modules`` (the winner was still importing it), took that
for its own completed import, and published the nothing it had registered over the winner's providers. The
tree was then recorded as live, so no later scan imported it: ``Unknown provider`` until the process
restarted, and every new chat of that profile fell to its fallback model.
"""

from __future__ import annotations

import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN = textwrap.dedent(
    """
    import time
    from providers import register_provider
    from providers.base import ProviderProfile

    time.sleep({delay})  # a real plugin imports its package here; long enough for a second scan to arrive
    register_provider(ProviderProfile(name="{name}", aliases=("{name}-alias",), auth_type="external_process",
                                      base_url="process://{name}", api_mode="chat_completions"))
    """
)


def _install(home: Path, name: str, *, delay: float = 0.0, stamp: int | None = None) -> Path:
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN.format(name=name, delay=delay), encoding="utf-8")
    if stamp is not None:  # a release's files carry that release's times, as an update writes them
        for path in plugin.iterdir():
            os.utime(path, ns=(stamp, stamp))
    return plugin


@pytest.fixture
def home(tmp_path, monkeypatch):
    import providers
    from providers import plugin_reload

    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    monkeypatch.setattr(providers, "_discovering", 0)
    monkeypatch.setattr(plugin_reload, "_tree_imports", {}, raising=False)
    yield tmp_path
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _bound(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def _race(jobs: list) -> list:
    """Run *jobs* on threads released together; each job's result (or the exception it raised)."""
    gate = threading.Barrier(len(jobs))
    results: list = [None] * len(jobs)

    def run(index: int, job) -> None:
        gate.wait()
        try:
            results[index] = job()
        except BaseException as exc:  # reported through the assertion
            results[index] = exc

    threads = [threading.Thread(target=run, args=(i, job)) for i, job in enumerate(jobs)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return results


def test_concurrent_first_lookups_all_see_the_plugin_and_it_stays_registered(home):
    from hermes_cli.auth import resolve_provider
    from providers import get_provider_profile

    secondary = home / "profiles" / "scaleup"
    _install(secondary, "racer", delay=0.3)

    seen = _race([lambda: _bound(secondary, lambda: get_provider_profile("racer"))] * 6)

    # No scan answers before the tree is imported, and none erases what the importing scan registered.
    assert [getattr(profile, "name", profile) for profile in seen] == ["racer"] * 6
    assert len({id(profile) for profile in seen}) == 1  # imported once, not once per thread
    assert _bound(secondary, lambda: get_provider_profile("racer-alias")) is seen[0]
    # The route a new chat takes: without the provider this raises "Unknown provider 'racer'".
    assert _bound(secondary, lambda: resolve_provider("racer")) == "racer"


def test_scans_of_two_homes_that_overlap_leave_no_scan_marked_in_flight(home):
    import providers
    from hermes_cli.auth import PROVIDER_REGISTRY

    first, second = home / "profiles" / "first", home / "profiles" / "second"
    _install(first, "quick", delay=0.2)
    _install(second, "slow", delay=0.6)
    seen: dict[str, object] = {}

    def lookup(profile_home: Path, name: str) -> None:
        seen[name] = _bound(profile_home, lambda: providers.get_provider_profile(name))

    # The second scan starts inside the first and ends after it: each used to restore the flag it found.
    outer = threading.Thread(target=lookup, args=(first, "quick"))
    outer.start()
    deadline = time.monotonic() + 10
    while not providers._discovering and time.monotonic() < deadline:
        time.sleep(0.005)
    inner = threading.Thread(target=lookup, args=(second, "slow"))
    inner.start()
    outer.join(30)
    inner.join(30)

    assert {name: getattr(profile, "name", profile) for name, profile in seen.items()} == {
        "quick": "quick", "slow": "slow"}
    assert not providers._discovering
    # A scan left marked in flight for good stops every later scan from mirroring what it found into the
    # auth registry.
    late = home / "profiles" / "late"
    _install(late, "latecomer")

    def scanned_and_mirrored() -> bool:
        providers.get_provider_profile("latecomer")
        return "latecomer" in PROVIDER_REGISTRY

    assert _bound(late, scanned_and_mirrored)


def test_a_tree_restored_to_an_earlier_release_keeps_its_providers(home):
    from providers import get_provider_profile, refresh_bound_home

    secondary = home / "profiles" / "scaleup"

    def release(stamp: int):
        _install(secondary, "rolled", stamp=stamp)
        _bound(secondary, refresh_bound_home)
        return _bound(secondary, lambda: get_provider_profile("rolled"))

    first = release(10**12)
    assert release(2 * 10**12) is not first  # an update: a new import of the new files
    # Rolled back with the first release's file times: that tree's module is still loaded, and importing it
    # again registers nothing. The layer must take back what it registered the first time.
    assert release(10**12) is first


def test_a_slow_process_root_setup_does_not_spend_the_import_wait(home, monkeypatch):
    """r32 review (Astra 6.2, pass 15): upstream's process-root setup waits up to 30 s on a cross-process
    setup lock. The first scan made that root only after it published its pending import, so a second
    scan's bounded wait ran out on the lock, not on the plugin, and its turn had no provider. The root is
    made before the import is published: the second scan queues on the root, as on plain upstream, and
    then finds the tree."""
    from providers import get_provider_profile, plugin_reload

    real_new_root = plugin_reload._new_root

    def slow_new_root():
        time.sleep(1.0)  # another process holding .setup.lock
        return real_new_root()

    monkeypatch.setattr(plugin_reload, "_snapshot_root", None)
    monkeypatch.setattr(plugin_reload, "_new_root", slow_new_root)
    monkeypatch.setattr(plugin_reload, "_IMPORT_WAIT_SECONDS", 0.5)
    secondary = home / "profiles" / "slowroot"
    _install(secondary, "rooted", delay=0.1)

    seen = _race([lambda: _bound(secondary, lambda: get_provider_profile("rooted"))] * 2)

    assert [getattr(profile, "name", profile) for profile in seen] == ["rooted"] * 2
