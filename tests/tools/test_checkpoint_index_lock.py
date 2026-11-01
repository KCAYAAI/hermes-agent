"""An ``<index>.lock`` orphaned by a killed git must not wedge a project's checkpoints.

``_run_git`` SIGKILLs git on timeout and git can die on a signal (SIGBUS when a file it
mmaps is truncated, OOM, restart); git's own cleanup never runs, so the per-project
``indexes/<hash>.lock`` stays and every later ``git add -A`` fails with ``File exists``.
"""

import os
import time
from pathlib import Path

import pytest

from tools.checkpoint_manager import CheckpointManager, _index_path, _project_hash, _store_path


@pytest.fixture()
def work_dir(tmp_path):
    d = tmp_path / "project"
    d.mkdir()
    (d / "main.py").write_text("print('hello')\n")
    return d


@pytest.fixture()
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.checkpoint_manager.CHECKPOINT_BASE", tmp_path / "checkpoints")
    return CheckpointManager(enabled=True, max_snapshots=50)


def _orphan_lock(work_dir: Path, age_seconds: float) -> Path:
    index = _index_path(_store_path(), _project_hash(str(work_dir)))
    lock = index.with_name(index.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(b"")
    stamp = time.time() - age_seconds
    os.utime(lock, (stamp, stamp))
    return lock


def test_orphaned_index_lock_is_reclaimed(mgr, work_dir):
    assert mgr.ensure_checkpoint(str(work_dir), "first")
    lock = _orphan_lock(work_dir, age_seconds=3600)

    (work_dir / "main.py").write_text("print('changed')\n")
    mgr.new_turn()
    assert mgr.ensure_checkpoint(str(work_dir), "after killed git")
    assert not lock.exists()
    assert len(mgr.list_checkpoints(str(work_dir))) == 2


def test_fresh_index_lock_is_left_alone(mgr, work_dir):
    assert mgr.ensure_checkpoint(str(work_dir), "first")
    lock = _orphan_lock(work_dir, age_seconds=5)

    (work_dir / "main.py").write_text("print('changed')\n")
    mgr.new_turn()
    assert not mgr.ensure_checkpoint(str(work_dir), "lock may be live")
    assert lock.exists()


def test_hermes_scratch_dir_is_not_checkpointed(mgr, tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    scratch = home / "cache" / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "probe.py").write_text("x = 1\n")

    assert not mgr.ensure_checkpoint(str(scratch), "scratch write")
    assert not mgr.ensure_checkpoint(str(scratch / "nested"), "scratch write")
    assert mgr.list_checkpoints(str(scratch)) == []
