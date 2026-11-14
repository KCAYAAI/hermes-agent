"""A provider plugin's frozen copy outlives the scratch dir's idle retention while its process runs.

Provider plugins are imported from a frozen copy (#153300) made with ``tempfile``, so it lands in Hermes'
scratch dir, where an entry nothing wrote to for 24 h is deleted. A module file is read once, when it is first
imported, so a gateway up for more than a day lost the copy. A profile that had not yet built a client for
that provider (it ran on another model, or only its cron jobs had run since the restart) then failed its first
turn with ``No module named ...`` from ``create_client``'s lazy import, until the process restarted.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from hermes_constants import prune_scratch_dir

_INIT = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    class Profile(ProviderProfile):
        def create_client(self, **kw):
            from .client import Client  # imported on first use, like the DirectSDK plugin's `.directsdk`
            return Client()

    register_provider(Profile(name="lazy-provider", base_url="process://lazy-provider",
                              auth_type="external_process"))
    """
)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    import providers
    from providers import plugin_reload

    home = tmp_path / "home"
    scratch = home / "cache" / "scratch"
    scratch.mkdir(parents=True)
    plugin = home / "plugins" / "lazy-provider"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: lazy-provider\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_INIT, encoding="utf-8")
    (plugin / "client.py").write_text("class Client:\n    pass\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))  # what the boot hook's TMPDIR does
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    monkeypatch.setattr(plugin_reload, "_snapshot_root", None)
    monkeypatch.setattr(plugin_reload, "_keeper", None, raising=False)
    monkeypatch.setattr(plugin_reload, "_tree_imports", {}, raising=False)
    monkeypatch.setattr(plugin_reload, "_KEEP_IN_USE_SECONDS", 0.05, raising=False)
    yield scratch
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _age_by_two_days(root: Path) -> None:
    old = time.time() - 48 * 3600
    for directory, _, files in os.walk(root):
        for name in files:
            os.utime(os.path.join(directory, name), (old, old))
        os.utime(directory, (old, old))


def test_the_frozen_copy_survives_the_scratch_prune_while_its_process_runs(scratch):
    from providers import get_provider_profile, plugin_reload

    profile = get_provider_profile("lazy-provider")  # scanned and registered; its client file not imported yet
    copies = plugin_reload._snapshot_root
    assert profile is not None and copies is not None and copies.parent == scratch

    # Two days on, nothing has written to the copy; the process is still running and marks it as in use.
    _age_by_two_days(copies)
    marker = copies / ".in-use"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (marker.exists() and marker.stat().st_mtime > time.time() - 3600):
        time.sleep(0.02)

    prune_scratch_dir(scratch)

    assert copies.is_dir()
    assert type(profile.create_client()).__name__ == "Client"  # the lazy import still finds its file


def test_a_process_with_no_thread_to_spare_still_loads_the_plugin_and_marks_the_next_copy(scratch, monkeypatch):
    """Review: the keep-alive is a convenience for a day from now. Failing to start it must not fail the
    provider lookup that made the copy, and a later copy starts it."""
    import threading

    from providers import get_provider_profile, plugin_reload

    start = threading.Thread.start

    def no_thread(self):
        if self.name == "provider-plugin-copies":
            raise RuntimeError("can't start new thread")
        return start(self)

    monkeypatch.setattr(threading.Thread, "start", no_thread)
    profile = get_provider_profile("lazy-provider")
    copies = plugin_reload._snapshot_root
    assert profile is not None and copies is not None
    assert type(profile.create_client()).__name__ == "Client"
    assert not (copies / ".in-use").exists()

    monkeypatch.setattr(threading.Thread, "start", start)
    plugin_reload._snapshot(Path(os.environ["HERMES_HOME"]) / "plugins" / "lazy-provider", "a_later_copy")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (copies / ".in-use").exists():
        time.sleep(0.02)
    assert (copies / ".in-use").exists()
