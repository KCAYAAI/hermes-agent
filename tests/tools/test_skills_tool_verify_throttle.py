"""skills.verify_interval_seconds: skill_view trusts a lookup index it verified within the
interval instead of re-statting the whole tree on every call (in a busy multi-profile gateway each
of those ~thousands of stats waits on the GIL). When the window ends one background thread per
directory re-checks (and past the TTL rebuilds) while lookups keep the last verified index; none is
served unchecked once its build or last check started _MAX_STALE_SECONDS ago. A miss or ambiguity
seen through a trusted index is always re-checked before it is reported. 0 keeps the per-call check."""

import json
import os
import threading
from pathlib import Path

import pytest

import tools.skills_tool as st
from agent.skill_utils import TIER_EXTERNAL, TIER_LOCAL


def _join_background():
    for t in threading.enumerate():
        if t.name == "skills-lookup-verify":
            t.join(5)
            assert not t.is_alive()


@pytest.fixture(autouse=True)
def _fresh_index():
    st.clear_skills_cache()
    yield
    _join_background()
    st.clear_skills_cache()


def _skill(root, rel, fm_name=None):
    d = root.joinpath(*rel.split("/"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {fm_name or d.name}\ndescription: d\n---\nbody\n", encoding="utf-8")
    return d


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "skills"
    _skill(root, "cat/alpha")
    _skill(root, "cat/beta")
    return root


@pytest.fixture
def clock(monkeypatch):
    """A controllable monotonic clock for the window and TTL arithmetic."""
    now = [1000.0]
    monkeypatch.setattr(st.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def interval(monkeypatch):
    """Throttle on at 5 s (the production KC setting)."""
    value = [5.0]
    monkeypatch.setattr(st, "_verify_interval", lambda: value[0])
    return value


@pytest.fixture
def verifications(monkeypatch):
    """Record every full tree check (the per-call cost the throttle removes) and its thread."""
    calls = []
    real = st._index_is_current

    def counted(key, index):
        calls.append(threading.current_thread().name)
        return real(key, index)
    monkeypatch.setattr(st, "_index_is_current", counted)
    return calls


@pytest.fixture
def builds(monkeypatch):
    """Record every full read of the tree (an index build)."""
    calls = []
    real = st._build_lookup_index

    def counted(key):
        calls.append(threading.current_thread().name)
        return real(key)
    monkeypatch.setattr(st, "_build_lookup_index", counted)
    return calls


@pytest.fixture
def held(monkeypatch):
    """Hold every tree check until released (it never proceeds on its own)."""
    started, release = threading.Event(), threading.Event()
    threads = []
    real = st._index_is_current

    def slow(key, index):
        threads.append(threading.current_thread().name)
        started.set()
        release.wait()
        return real(key, index)

    def arm():
        monkeypatch.setattr(st, "_index_is_current", slow)
    try:
        yield arm, started, release, threads
    finally:
        release.set()


def _locate(name, root):
    return st.locate_skill(name, None, [(TIER_LOCAL, root)])


def _warm(tree):
    assert _locate("alpha", tree)[0] is None  # build (published as verified)


def test_default_interval_is_zero_and_verifies_every_call(tree, verifications, monkeypatch, tmp_path):
    import agent.skill_utils as su
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))  # never read the host's real config
    su._raw_config_cache_clear()
    assert st._verify_interval() == 0.0
    for _ in range(3):
        error, _sd, smd = _locate("alpha", tree)
        assert error is None and smd == tree / "cat/alpha/SKILL.md"
    assert len(verifications) == 2  # first call builds; every later call re-checks the tree


def test_verify_interval_read_from_a_real_config(tmp_path, monkeypatch):
    import agent.skill_utils as su
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    su._raw_config_cache_clear()
    assert su.skills_verify_interval() == 0.0
    (home / "config.yaml").write_text("skills:\n  verify_interval_seconds: 5\n", encoding="utf-8")
    su._raw_config_cache_clear()
    assert su.skills_verify_interval() == 5.0


def _expire_window(tree, clock, seconds=10.0):
    """Let the window end, start the background check with one lookup, and wait for it."""
    clock[0] += seconds
    assert _locate("beta", tree)[0] is None
    _join_background()


def test_within_interval_the_verified_index_is_trusted(tree, verifications, clock, interval):
    _warm(tree)
    checks = len(verifications)
    for _ in range(5):
        clock[0] += 0.5
        assert _locate("beta", tree)[0] is None
    assert len(verifications) == checks  # no tree walk inside the window
    _expire_window(tree, clock, 5.0)
    assert verifications[checks:] == ["skills-lookup-verify"]  # one check, off the request path


def test_skill_added_right_after_a_check_is_found(tree, builds, clock, interval):
    """Astra r32 pass 2: there is no grace period. A skill another process adds 0.2 s after a check
    and requested at 0.5 s (a Kanban preload treats a miss as fatal) is found by the miss re-check."""
    _warm(tree)
    clock[0] += 0.2
    _skill(tree, "cat/newskill")
    clock[0] += 0.3
    before = len(builds)
    error, _sd, smd = _locate("newskill", tree)
    assert error is None and smd == tree / "cat/newskill/SKILL.md"
    assert len(builds) == before + 1  # the miss re-check is one full read


def test_forced_recheck_publishes_so_the_next_hit_stays_fast(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 2.0
    _skill(tree, "cat/gamma")
    assert _locate("gamma", tree)[0] is None  # forced re-check rebuilt the index
    before = len(verifications)
    clock[0] += 1.0
    assert _locate("beta", tree)[0] is None
    assert len(verifications) == before  # the rebuilt index was published as verified


def test_each_true_miss_is_checked_exactly_once(tree, builds, clock, interval):
    _warm(tree)
    for _ in range(3):
        clock[0] += 0.5
        before = len(builds)
        error, _sd, smd = _locate("no-such-skill", tree)
        assert smd is None and "not found" in error
        assert len(builds) == before + 1  # one forced full read, never more


def test_unthrottled_miss_is_not_rechecked(tree, verifications, clock):
    _warm(tree)
    before = len(verifications)
    error, _sd, _smd = _locate("no-such-skill", tree)
    assert "not found" in error
    assert len(verifications) == before + 1  # the ordinary per-call check only


def test_new_collision_is_reported_after_the_background_check(tree, clock, interval):
    _warm(tree)
    clock[0] += 2.0
    _skill(tree, "other/alpha")  # same name, same tier, added by "another process"
    assert _locate("alpha", tree)[0] is None  # inside the window: the trusted unique winner (documented)
    _expire_window(tree, clock, 5.0)
    error, _sd, smd = _locate("alpha", tree)
    assert smd is None and "alpha" in error


def test_index_past_the_ttl_is_rebuilt_by_the_background_check(tmp_path, clock, interval):
    """Astra r32 pass 2: a skill added below the signature depth is invisible to the quick check, so
    only the TTL rebuild finds it. A higher-precedence local copy must win over the external one once
    the TTL has passed; a positive (external) match is not re-checked as a miss."""
    local, external = tmp_path / "local", tmp_path / "external"
    local.joinpath("a", "b", "c", "d", "e", "f", "g").mkdir(parents=True)  # exists before the build
    _skill(external, "cat/x")
    roots = [(TIER_LOCAL, local), (TIER_EXTERNAL, external)]
    assert st.locate_skill("x", None, roots)[2] == external / "cat/x/SKILL.md"
    clock[0] += 1.0
    deep = "a/b/c/d/e/f/g/x"  # the new folder is below the depth the quick check watches
    _skill(local, deep)
    clock[0] += st._SKILLS_CACHE_TTL_SECONDS
    st.locate_skill("other", None, roots)  # window over: the background check rebuilds (past TTL)
    _join_background()
    error, _sd, smd = st.locate_skill("x", None, roots)
    assert error is None and smd == local / deep / "SKILL.md"


def test_index_past_the_bound_is_checked_before_it_is_served(tmp_path, clock, interval):
    """Astra r32 pass 2: idle past the bound, no background check ran; the lookup checks inline."""
    local, external = tmp_path / "local", tmp_path / "external"
    local.joinpath("a", "b", "c", "d", "e", "f", "g").mkdir(parents=True)
    _skill(external, "cat/x")
    roots = [(TIER_LOCAL, local), (TIER_EXTERNAL, external)]
    assert st.locate_skill("x", None, roots)[2] == external / "cat/x/SKILL.md"
    clock[0] += 1.0
    _skill(local, "a/b/c/d/e/f/g/x")
    clock[0] += st._MAX_STALE_SECONDS
    error, _sd, smd = st.locate_skill("x", None, roots)
    assert error is None and smd == local / "a/b/c/d/e/f/g/x/SKILL.md"
    assert not any(t.name == "skills-lookup-verify" for t in threading.enumerate())


def test_verification_never_extends_the_ttl(tree, clock, interval):
    _warm(tree)
    built = st._SKILL_LOOKUP_INDEX[os.path.realpath(tree)].built_at
    for _ in range(int(st._SKILLS_CACHE_TTL_SECONDS // 5) + 1):
        _expire_window(tree, clock, 5.0)
    rebuilt = st._SKILL_LOOKUP_INDEX[os.path.realpath(tree)].built_at
    assert rebuilt > built  # past the TTL the index was rebuilt, not kept alive by verifications


@pytest.mark.parametrize("advance", [10.0, st._SKILLS_CACHE_TTL_SECONDS + 5])
def test_expired_window_serves_now_and_checks_once_in_the_background(tree, clock, interval, held, advance):
    """Window expiry and TTL expiry (rebuild) alike: every caller is served at once, one thread checks."""
    arm, started, release, threads = held
    _warm(tree)
    clock[0] += advance
    arm()
    results = []
    callers = [threading.Thread(target=lambda: results.append(_locate("beta", tree)[0])) for _ in range(8)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(5)
    assert all(not t.is_alive() for t in callers) and results == [None] * 8  # served while held
    assert started.wait(5)
    release.set()
    _join_background()
    assert threads == ["skills-lookup-verify"]


def test_past_the_bound_callers_wait_for_one_inline_rebuild(tree, clock, interval, monkeypatch):
    """Past the bound nothing is served unchecked; an index built that long ago is rebuilt (not just
    re-signed) by one caller while the others wait for that rebuild and reuse it."""
    started, release = threading.Event(), threading.Event()
    threads = []
    real_build = st._build_lookup_index

    def held_build(key):
        threads.append(threading.current_thread().name)
        started.set()
        release.wait()
        return real_build(key)
    _warm(tree)
    clock[0] += st._MAX_STALE_SECONDS + 1
    monkeypatch.setattr(st, "_build_lookup_index", held_build)
    results = []
    callers = [threading.Thread(target=lambda: results.append(_locate("beta", tree)[0]), name=f"caller-{i}")
               for i in range(8)]
    for t in callers:
        t.start()
    assert started.wait(5)
    assert results == []  # nobody is served an index past the bound
    release.set()
    for t in callers:
        t.join(5)
    assert all(not t.is_alive() for t in callers) and results == [None] * 8
    assert len(threads) == 1 and threads[0].startswith("caller-")  # the others reused that check


def test_background_failure_drops_trust_and_the_next_caller_checks(tree, clock, interval, monkeypatch, caplog):
    _warm(tree)
    real = st._index_is_current
    caller_done = threading.Event()

    def boom(key, index):
        if threading.current_thread().name == "skills-lookup-verify":
            caller_done.wait(5)  # fail only after the lookup that started this check has returned
            raise PermissionError("unreadable skills dir")
        return real(key, index)
    monkeypatch.setattr(st, "_index_is_current", boom)
    clock[0] += 10.0
    assert _locate("beta", tree)[0] is None  # served while the check runs in the background
    caller_done.set()
    _join_background()
    key = os.path.realpath(tree)
    assert key not in st._LOOKUP_VERIFIED and key not in st._CHECK_REQUESTED
    assert not st._verify_lock(key).locked()
    assert "background re-check" in caplog.text
    monkeypatch.setattr(st, "_index_is_current", real)
    assert _locate("alpha", tree)[0] is None  # the next caller checks inline; trust is restored
    assert key in st._LOOKUP_VERIFIED


def test_thread_start_failure_after_launch_never_releases_a_lock_twice(tree, clock, interval, monkeypatch):
    """Astra r32 pass 2: Thread.start() can raise after the thread launched. The caller holds no
    lock across the start, so a second lookup checking inline keeps its own lock."""
    _warm(tree)
    real_start = threading.Thread.start

    def start_then_fail(self):
        real_start(self)
        if self.name == "skills-lookup-verify":
            raise RuntimeError("waiting for the started event failed")
    monkeypatch.setattr(threading.Thread, "start", start_then_fail)
    clock[0] += 10.0
    assert _locate("beta", tree)[0] is None
    clock[0] += st._MAX_STALE_SECONDS  # the next caller must check inline while the thread may run
    assert _locate("alpha", tree)[0] is None
    _join_background()
    key = os.path.realpath(tree)
    assert not st._verify_lock(key).locked() and key not in st._CHECK_REQUESTED
    assert _locate("beta", tree)[0] is None


def test_clear_during_a_background_check_is_not_undone(tree, clock, interval, held):
    arm, started, release, _threads = held
    _warm(tree)
    clock[0] += 10.0
    arm()
    assert _locate("beta", tree)[0] is None
    assert started.wait(5)
    st.clear_skill_lookup_index()  # an in-process skill write lands mid-check
    release.set()
    _join_background()
    assert os.path.realpath(tree) not in st._LOOKUP_VERIFIED


def test_two_spellings_of_one_directory_share_one_index(tree, verifications, clock, interval, tmp_path):
    link = tmp_path / "skills-link"
    link.symlink_to(tree)
    _warm(tree)
    before = len(verifications)
    _error, _sd, smd = st.locate_skill("beta", None, [(TIER_LOCAL, tree), (TIER_LOCAL, link)])
    assert smd is not None
    assert len(verifications) == before


def test_window_over_lookup_never_waits_on_a_rebuild_in_progress(tree, clock, interval, monkeypatch):
    """Astra r32 pass 4: a rebuild holds the global lookup lock for its whole run. A lookup whose
    window is over must be served from the bound at once, not queue behind that lock."""
    _warm(tree)
    building, release = threading.Event(), threading.Event()
    real_build = st._build_lookup_index

    def held_build(key):
        building.set()
        release.wait()
        return real_build(key)
    monkeypatch.setattr(st, "_build_lookup_index", held_build)
    clock[0] += st._SKILLS_CACHE_TTL_SECONDS + 1  # window and TTL over: the background check rebuilds
    try:
        assert _locate("beta", tree)[0] is None
        assert building.wait(5)  # the background rebuild now holds the global lookup lock
        done = []
        callers = [threading.Thread(target=lambda: done.append(_locate("alpha", tree)[0])) for _ in range(4)]
        for t in callers:
            t.start()
        for t in callers:
            t.join(5)
        assert all(not t.is_alive() for t in callers) and done == [None] * 4
    finally:
        release.set()
    _join_background()


def test_slow_rebuild_past_the_bound_is_rebuilt_not_resigned(tmp_path, clock, interval, monkeypatch):
    """Astra r32 pass 4: a rebuild read the tree at t=40 and finished at t=145. A skill added below
    the signature depth after that read must be found by a lookup past build start + bound, even
    though the index's TTL (counted from the end of the build) has not run out."""
    local, external = tmp_path / "local", tmp_path / "external"
    local.joinpath("a", "b", "c", "d", "e", "f", "g").mkdir(parents=True)
    _skill(local, "cat/y")
    _skill(external, "cat/x")
    roots = [(TIER_LOCAL, local), (TIER_EXTERNAL, external)]
    assert st.locate_skill("x", None, roots)[2] == external / "cat/x/SKILL.md"
    real_build = st._build_lookup_index

    def slow_build(key):
        index = real_build(key)  # reads the tree now
        if key == os.path.realpath(local):
            _skill(local, "a/b/c/d/e/f/g/x")  # added after the read, below the signature depth
            clock[0] += 105.0
            index = index._replace(built_at=clock[0])
        return index
    clock[0] += st._SKILLS_CACHE_TTL_SECONDS + 10
    monkeypatch.setattr(st, "_build_lookup_index", slow_build)
    assert st.locate_skill("y", None, roots)[2] == local / "cat/y/SKILL.md"  # a hit starts the rebuild
    _join_background()
    monkeypatch.setattr(st, "_build_lookup_index", real_build)
    clock[0] += st._MAX_STALE_SECONDS - 105.0 + 1.0  # past build start + bound, TTL not yet over
    error, _sd, smd = st.locate_skill("x", None, roots)
    assert error is None and smd == local / "a/b/c/d/e/f/g/x/SKILL.md"


def test_launcher_error_never_fails_the_lookup(tree, clock, interval, monkeypatch):
    """Astra r32 pass 4: a MemoryError (or any Exception) starting the optional background check
    is a check not scheduled; the lookup of an unchanged skill still succeeds."""
    _warm(tree)

    def no_thread(self):
        raise MemoryError("waiter allocation")
    monkeypatch.setattr(threading.Thread, "start", no_thread)
    clock[0] += 10.0
    error, _sd, smd = _locate("beta", tree)
    assert error is None and smd == tree / "cat/beta/SKILL.md"
    assert os.path.realpath(tree) not in st._CHECK_REQUESTED


def test_stale_positive_refused_by_a_gate_is_resolved_again_verified(tree, clock, interval, monkeypatch):
    """Astra r32 pass 4: within the window another process moves the name ``x`` from folder a to b
    and makes a unsupported here. The trusted index still says a; skill_view must not make that
    refusal final, but resolve once more verified and load b."""
    _skill(tree, "cat/a", fm_name="x")
    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([(TIER_LOCAL, tree)], tree))
    assert _locate("x", tree)[2] == tree / "cat/a/SKILL.md"  # warm: x is a
    clock[0] += 1.0
    (tree / "cat/a/SKILL.md").write_text(
        "---\nname: a\ndescription: d\nplatforms: [nonexistent-os]\n---\nbody\n", encoding="utf-8")
    _skill(tree, "cat/b", fm_name="x")
    clock[0] += 1.0  # still inside the window: the trusted index maps x to a
    result = json.loads(st.skill_view("x"))
    assert result.get("success") is True, result
    assert result["skill_dir"] == str(tree / "cat/b")


def test_a_miss_after_a_recent_rebuild_still_finds_a_deep_new_skill(tmp_path, clock, interval):
    """Astra r32 pass 5: a miss is final only on a full read of the tree. A skill added below the
    quick check's depth right after a rebuild must be found by the miss re-check, not missed twice."""
    local = tmp_path / "local"
    _skill(local, "cat/y")
    deep = local.joinpath("a", "b", "c", "d", "e", "f", "g")
    deep.mkdir(parents=True)
    roots = [(TIER_LOCAL, local)]
    assert st.locate_skill("y", None, roots)[0] is None  # built and verified now
    clock[0] += 1.0
    _skill(deep, "x")  # seven folders down: the signature does not see it
    clock[0] += 1.0  # inside the window
    error, _sd, smd = st.locate_skill("x", None, roots)
    assert error is None and smd == deep / "x/SKILL.md"


def test_stale_quarantined_winner_is_resolved_again(tmp_path, clock, interval, monkeypatch):
    """Astra r32 pass 5: quarantine refuses a located copy. When that copy came through a trusted
    index, the lookup re-resolves on a fresh read before the refusal stands."""
    import agent.skill_utils as su
    project, local = tmp_path / "project", tmp_path / "local"
    _skill(project, "cat/a", fm_name="x")
    _skill(local, "cat/other")
    roots = [(su.TIER_PROJECT, project), (TIER_LOCAL, local)]
    monkeypatch.setattr(su, "is_quarantined_project_skill", lambda md: Path(md).parent.name == "a")
    assert st.locate_skill("other", None, roots)[0] is None  # warm both roots
    clock[0] += 1.0
    (project / "cat/a/SKILL.md").write_text("---\nname: a\ndescription: d\n---\nbody\n", encoding="utf-8")
    _skill(local, "cat/b", fm_name="x")
    clock[0] += 1.0  # inside the window: the trusted project index still maps x to a
    error, _sd, smd = st.locate_skill("x", None, roots)
    assert error is None and smd == local / "cat/b/SKILL.md"


def test_a_fifo_left_at_a_stale_winner_is_refused_and_resolved_again(tree, clock, interval, monkeypatch):
    """Astra r32 pass 5: a FIFO where the stale winner's SKILL.md was must not block skill_view."""
    _skill(tree, "cat/a", fm_name="x")
    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([(TIER_LOCAL, tree)], tree))
    assert _locate("x", tree)[2] == tree / "cat/a/SKILL.md"
    clock[0] += 1.0
    md = tree / "cat/a/SKILL.md"
    md.unlink()
    os.mkfifo(md)
    _skill(tree, "cat/b", fm_name="x")
    clock[0] += 1.0
    out = []
    worker = threading.Thread(target=lambda: out.append(json.loads(st.skill_view("x"))), daemon=True)
    worker.start()
    worker.join(10)
    if worker.is_alive():  # unblock the reader before failing
        os.close(os.open(md, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(5)
        pytest.fail("skill_view blocked on a FIFO")
    assert out[0].get("success") is True, out[0]
    assert out[0]["skill_dir"] == str(tree / "cat/b")


def test_a_refusal_of_the_same_copy_stands_as_first_given(tree, clock, interval, monkeypatch):
    """Astra r32 pass 5: the verified retry must not weaken a refusal of an unchanged winner (for
    example when the disabled-skills config cannot be read the second time)."""
    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([(TIER_LOCAL, tree)], tree))
    _warm(tree)
    clock[0] += 1.0
    second = []
    real_locate = st.locate_skill

    def locate(*a, verify=False, **k):
        if verify:
            second.append(True)
        return real_locate(*a, verify=verify, **k)
    monkeypatch.setattr(st, "locate_skill", locate)
    monkeypatch.setattr(st, "_is_skill_disabled", lambda *a, **k: not second)
    result = json.loads(st.skill_view("beta"))
    assert result.get("success") is False and "disabled" in result.get("error", ""), result


def test_a_refusal_through_a_checked_index_is_not_retried(tree, clock, interval, monkeypatch):
    """Fable r32 pass 5: no second resolution when the first pass was checked."""
    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([(TIER_LOCAL, tree)], tree))
    monkeypatch.setattr(st, "_is_skill_disabled", lambda *a, **k: True)
    calls = []
    real_locate = st.locate_skill
    monkeypatch.setattr(st, "locate_skill", lambda *a, **k: calls.append(k.get("verify")) or real_locate(*a, **k))
    assert "disabled" in json.loads(st.skill_view("beta")).get("error", "")  # cold: built and checked
    assert calls == [False]


def test_an_interrupted_launcher_clears_its_request(tree, clock, interval, monkeypatch):
    """Fable/Astra r32 pass 5: a KeyboardInterrupt from start() propagates and leaves no request."""
    _warm(tree)

    def interrupted(self):
        raise KeyboardInterrupt
    monkeypatch.setattr(threading.Thread, "start", interrupted)
    clock[0] += 10.0
    with pytest.raises(KeyboardInterrupt):
        _locate("beta", tree)
    assert os.path.realpath(tree) not in st._CHECK_REQUESTED


def test_waiters_reuse_a_rebuild_that_outlived_the_bound(tree, clock, interval, monkeypatch):
    """Fable r32 pass 5: a rebuild that started after callers began waiting is the freshest read they
    could get, even when it outlived the bound; they reuse it instead of rebuilding in turn."""
    import time as real_time
    _warm(tree)
    clock[0] += st._MAX_STALE_SECONDS + 1
    key = os.path.realpath(tree)
    builds = []
    real_build = st._build_lookup_index

    def slow_build(k):
        builds.append(1)
        index = real_build(k)
        clock[0] += st._MAX_STALE_SECONDS + 10  # the build itself outlives the bound
        return index
    monkeypatch.setattr(st, "_build_lookup_index", slow_build)
    done = []
    callers = [threading.Thread(target=lambda: done.append(_locate("alpha", tree)[0])) for _ in range(3)]
    with st._verify_lock(key):  # the callers queue behind a check in progress
        for t in callers:
            t.start()
        real_time.sleep(0.3)  # all three are waiting before the build below starts
        clock[0] += 1.0
        st._checked_lookup_index(key, None, publish=True)
    for t in callers:
        t.join(10)
    assert done == [None] * 3 and len(builds) == 1


@pytest.mark.parametrize("raw, expected", [
    (None, 0.0), (0, 0.0), (5, 5.0), ("2.5", 2.5), (-1, 0.0), ("abc", 0.0),
    (float("nan"), 0.0), (float("inf"), 0.0), (True, 0.0), (10 ** 400, 0.0)])
def test_interval_config_parsing(raw, expected, monkeypatch):
    import agent.skill_utils as su
    monkeypatch.setattr(su, "_skills_cfg_get", lambda key: raw if key == "verify_interval_seconds" else None)
    assert su.skills_verify_interval() == expected
