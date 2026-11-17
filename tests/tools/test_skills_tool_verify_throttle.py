"""skills.verify_interval_seconds: skill_view trusts a lookup index it verified within the
interval instead of re-statting the whole tree on every call (in a busy multi-profile gateway each
of those ~thousands of stats waits on the GIL). Once verified, the re-check runs in one background
thread per directory while callers get the last verified index, never one older than
_MAX_STALE_SECONDS. A miss or ambiguity seen through a trusted index is re-checked once (not when a
check started under a second ago). 0 keeps the per-call check."""

import os
import threading

import pytest

import tools.skills_tool as st
from agent.skill_utils import TIER_LOCAL


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
    """A controllable monotonic clock for the window, floor, cap and TTL arithmetic."""
    now = [1000.0]
    monkeypatch.setattr(st.time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def interval(monkeypatch):
    """Throttle on at 5 s, re-checks inline so a test can assert the order of checks."""
    value = [5.0]
    monkeypatch.setattr(st, "_verify_interval", lambda: value[0])
    monkeypatch.setattr(st, "_BACKGROUND_VERIFY", False)
    return value


@pytest.fixture
def background(interval, monkeypatch):
    """The production shape: throttle on, re-checks in the background thread."""
    monkeypatch.setattr(st, "_BACKGROUND_VERIFY", True)
    return interval


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


def test_within_interval_the_verified_index_is_trusted(tree, verifications, clock, interval):
    _warm(tree)
    checks = len(verifications)
    for _ in range(5):
        clock[0] += 0.5
        assert _locate("beta", tree)[0] is None
    assert len(verifications) == checks  # no tree walk inside the window
    clock[0] += 5.0
    assert _locate("alpha", tree)[0] is None
    assert len(verifications) == checks + 1  # window over: one re-check


def test_new_skill_inside_window_is_found_by_the_forced_recheck(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 2.0  # inside the window, past the re-check floor
    _skill(tree, "cat/gamma")  # created by "another process": no in-process cache clear
    before = len(verifications)
    error, _sd, smd = _locate("gamma", tree)
    assert error is None and smd == tree / "cat/gamma/SKILL.md"
    assert len(verifications) == before + 1


def test_forced_recheck_publishes_so_the_next_hit_stays_fast(tree, verifications, clock, background):
    _warm(tree)
    clock[0] += 2.0
    _skill(tree, "cat/gamma")
    assert _locate("gamma", tree)[0] is None  # forced re-check rebuilt the index
    before = len(verifications)
    clock[0] += 1.0
    assert _locate("beta", tree)[0] is None
    assert len(verifications) == before  # the rebuilt index was published as verified


def test_true_miss_rechecks_exactly_once(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 2.0
    before = len(verifications)
    error, _sd, smd = _locate("no-such-skill", tree)
    assert smd is None and "not found" in error
    assert len(verifications) == before + 1


def test_miss_right_after_a_check_is_not_rechecked(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 0.5  # under the 1 s floor: a retry loop on a bad name stays cheap
    before = len(verifications)
    assert "not found" in _locate("no-such-skill", tree)[0]
    assert len(verifications) == before


def test_repeated_misses_reuse_the_forced_check(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 2.0
    assert "not found" in _locate("no-such-skill", tree)[0]
    before = len(verifications)
    clock[0] += 0.5
    assert "not found" in _locate("no-such-skill", tree)[0]
    assert len(verifications) == before  # the forced check restarted the floor


def test_unthrottled_miss_is_not_rechecked(tree, verifications, clock):
    _warm(tree)
    before = len(verifications)
    error, _sd, _smd = _locate("no-such-skill", tree)
    assert "not found" in error
    assert len(verifications) == before + 1  # the ordinary per-call check only


def test_new_collision_is_reported_once_the_window_ends(tree, verifications, clock, interval):
    _warm(tree)
    clock[0] += 2.0
    _skill(tree, "other/alpha")  # same name, same tier, added by "another process"
    assert _locate("alpha", tree)[0] is None  # inside the window: the trusted unique winner (documented)
    clock[0] += 5.0
    before = len(verifications)
    error, _sd, smd = _locate("alpha", tree)
    assert smd is None and "alpha" in error
    assert len(verifications) == before + 1


def test_idle_index_past_the_cap_is_checked_before_it_is_served(tree, verifications, clock, background):
    _warm(tree)
    _skill(tree, "other/alpha")  # a collision appears while the process is idle
    clock[0] += st._MAX_STALE_SECONDS + 1
    error, _sd, smd = _locate("alpha", tree)
    assert smd is None and "alpha" in error  # never the old unique winner
    assert verifications[-1] == threading.current_thread().name  # checked inline, not in background


def test_expired_window_is_rechecked_off_the_request_path(tree, clock, background, held):
    arm, started, release, threads = held
    _warm(tree)
    clock[0] += 10.0
    arm()
    done = []
    caller = threading.Thread(target=lambda: done.append(_locate("beta", tree)))
    caller.start()
    caller.join(5)
    assert not caller.is_alive() and not release.is_set()  # answered while the check is held
    assert done[0][0] is None and done[0][2] == tree / "cat/beta/SKILL.md"
    assert started.wait(5) and threads == ["skills-lookup-verify"]
    second = threading.Thread(target=lambda: done.append(_locate("alpha", tree)))
    second.start()
    second.join(5)
    assert not second.is_alive() and done[1][0] is None  # still served, no second walk
    release.set()
    _join_background()
    assert threads == ["skills-lookup-verify"]
    assert st._LOOKUP_VERIFIED[os.path.realpath(tree)][1] == clock[0]


def test_background_failure_drops_trust_and_releases_the_lock(tree, clock, background, monkeypatch):
    _warm(tree)
    clock[0] += 10.0
    real = st._index_is_current

    def boom(key, index):
        if threading.current_thread().name == "skills-lookup-verify":
            raise PermissionError("unreadable skills dir")
        return real(key, index)
    monkeypatch.setattr(st, "_index_is_current", boom)
    assert _locate("beta", tree)[0] is None  # served while the check fails in the background
    _join_background()
    key = os.path.realpath(tree)
    assert key not in st._LOOKUP_VERIFIED
    assert not st._verify_lock(key).locked()
    monkeypatch.setattr(st, "_index_is_current", real)
    assert _locate("alpha", tree)[0] is None  # the next caller checks inline; trust is restored
    assert key in st._LOOKUP_VERIFIED


def test_clear_during_background_check_is_not_undone(tree, clock, background, held):
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


def test_verification_never_extends_the_ttl(tree, verifications, clock, interval):
    _warm(tree)
    built = st._SKILL_LOOKUP_INDEX[os.path.realpath(tree)].built_at
    for _ in range(int(st._SKILLS_CACHE_TTL_SECONDS // 5) + 1):
        clock[0] += 5.0
        _locate("alpha", tree)
    rebuilt = st._SKILL_LOOKUP_INDEX[os.path.realpath(tree)].built_at
    assert rebuilt > built  # past the TTL the index was rebuilt, not kept alive by verifications


@pytest.mark.parametrize("advance", [10.0, st._SKILLS_CACHE_TTL_SECONDS + 5])
def test_concurrent_expiry_runs_one_check(tree, clock, background, held, advance):
    """Window expiry and TTL expiry (rebuild) alike: one thread checks, every caller is served."""
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
    assert all(not t.is_alive() for t in callers) and results == [None] * 8
    assert started.wait(5)
    release.set()
    _join_background()
    assert threads == ["skills-lookup-verify"]


def test_two_spellings_of_one_directory_share_one_index(tree, verifications, clock, interval, tmp_path):
    link = tmp_path / "skills-link"
    link.symlink_to(tree)
    _warm(tree)
    before = len(verifications)
    _error, _sd, smd = st.locate_skill("beta", None, [(TIER_LOCAL, tree), (TIER_LOCAL, link)])
    assert smd is not None
    assert len(verifications) == before


@pytest.mark.parametrize("raw, expected", [
    (None, 0.0), (0, 0.0), (5, 5.0), ("2.5", 2.5), (-1, 0.0), ("abc", 0.0),
    (float("nan"), 0.0), (float("inf"), 0.0), (True, 0.0), (10 ** 400, 0.0)])
def test_interval_config_parsing(raw, expected, monkeypatch):
    import agent.skill_utils as su
    monkeypatch.setattr(su, "_skills_cfg_get", lambda key: raw if key == "verify_interval_seconds" else None)
    assert su.skills_verify_interval() == expected
