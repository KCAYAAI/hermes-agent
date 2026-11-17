"""skill_view's per-process lookup index: same candidates as the tree walk, changes visible on
the next call, and an unchanged tree costs no frontmatter parse and no recursive glob."""

import os
import pathlib

import pytest

import tools.skills_tool as st


@pytest.fixture(autouse=True)
def _fresh_index():
    st.clear_skills_cache()
    yield
    st.clear_skills_cache()


def _skill(root, rel, fm_name=None, body="body"):
    d = root.joinpath(*rel.split("/"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {fm_name or d.name}\ndescription: d\n---\n{body}\n", encoding="utf-8")
    return d


def _flat(root, rel, text="# flat\n"):
    p = root.joinpath(*rel.split("/"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _cands(name, dirs, category=None):
    return st._collect_skill_candidates(name, category, dirs)


@pytest.fixture
def tree(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _skill(a, "cat/alpha")
    _skill(a, "cat/renamed-dir", fm_name="beta")
    _skill(a, "pkg")
    _flat(a, "pkg/docs/gamma.md")  # owned by the pkg skill: never a candidate
    _flat(a, "legacy/gamma.md")
    _flat(a, "deep/x/cat/delta.md")
    _skill(a, "one/dup")
    _skill(b, "dup")
    _flat(a, ".curator_backups/old/pkg/SKILL.md")
    _flat(a, ".curator_backups/old/pkg/epsilon.md")
    return a, b


def _call_twice(name, dirs, category=None):
    first = _cands(name, dirs, category)
    assert _cands(name, dirs, category) == first
    return first


def test_same_candidates_as_the_tree_walk(tree):
    a, b = tree
    assert _call_twice("alpha", [a, b]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    assert _call_twice("beta", [a, b]) == [(a / "cat/renamed-dir", a / "cat/renamed-dir/SKILL.md")]
    assert _call_twice("cat/alpha", [a, b]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    assert _call_twice("gamma", [a, b]) == [(None, a / "legacy/gamma.md")]
    assert _call_twice("cat/delta", [a, b]) == [(None, a / "deep/x/cat/delta.md")]
    assert _call_twice("x/delta", [a, b]) == []
    assert _call_twice("epsilon", [a, b]) == []
    assert _call_twice("dup", [a, b]) == [(a / "one/dup", a / "one/dup/SKILL.md"), (b / "dup", b / "dup/SKILL.md")]
    assert _call_twice("nope", [a, b]) == []


def test_changes_are_visible_on_the_next_call(tree):
    a, b = tree
    assert _cands("zeta", [a, b]) == []
    _skill(a, "cat/zeta")  # new skill
    assert _cands("zeta", [a, b]) == [(a / "cat/zeta", a / "cat/zeta/SKILL.md")]
    os.rename(a / "cat/zeta", a / "cat/zeta2")  # renamed folder
    assert _cands("zeta", [a, b]) == [(a / "cat/zeta2", a / "cat/zeta2/SKILL.md")]  # frontmatter still zeta
    assert _cands("zeta2", [a, b]) == [(a / "cat/zeta2", a / "cat/zeta2/SKILL.md")]
    md = a / "cat/zeta2/SKILL.md"
    stat = md.stat()
    md.write_text(md.read_text(encoding="utf-8").replace("name: zeta", "name: etaa"), encoding="utf-8")
    os.utime(md, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # same size, newer mtime
    assert _cands("zeta", [a, b]) == []
    assert _cands("etaa", [a, b]) == [(a / "cat/zeta2", a / "cat/zeta2/SKILL.md")]
    md.unlink()
    (a / "cat/zeta2").rmdir()  # deleted skill
    assert _cands("etaa", [a, b]) == []
    assert _cands("zeta2", [a, b]) == []


def _count_costs(monkeypatch):
    calls = {"frontmatter": 0, "rglob": []}
    real_safe, real_parse, real_rglob = st._safe_frontmatter, st._parse_frontmatter, pathlib.Path.rglob

    def safe(*args, **kwargs):
        calls["frontmatter"] += 1
        return real_safe(*args, **kwargs)

    def parse(*args, **kwargs):
        calls["frontmatter"] += 1
        return real_parse(*args, **kwargs)

    def rglob(self, pattern, *args, **kwargs):
        calls["rglob"].append(pattern)
        return real_rglob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(st, "_safe_frontmatter", safe)
    monkeypatch.setattr(st, "_parse_frontmatter", parse)
    monkeypatch.setattr(pathlib.Path, "rglob", rglob)
    return calls


def test_unchanged_tree_parses_no_frontmatter_and_runs_no_rglob(tree, monkeypatch):
    a, b = tree
    first = _cands("beta", [a, b])
    calls = _count_costs(monkeypatch)
    for name in ("beta", "gamma", "cat/delta", "nope"):
        _cands(name, [a, b])
    assert _cands("beta", [a, b]) == first
    assert calls == {"frontmatter": 0, "rglob": []}


def test_glob_characters_take_the_tree_walk(tree, monkeypatch):
    a, b = tree
    _cands("gam*", [a, b])
    calls = _count_costs(monkeypatch)
    assert _cands("gam*", [a, b]) == [(None, a / "legacy/gamma.md")]
    assert calls["rglob"] == ["gam*.md", "gam*.md"]


def test_copies_of_a_qualified_flat_file_keep_the_tree_walk_order(tmp_path):
    """``rglob("cat/eta.md")`` orders copies by the directory it matched under, not by the file's
    own directory; with several copies the candidate order must still be the glob's."""
    root = tmp_path / "r"
    for rel in ("cat/eta.md", "a/x/z/cat/eta.md", "b/cat/eta.md", "c/cat/eta.md", "b/y/cat/eta.md"):
        _flat(root, rel)
    expected = [(None, md) for md in root.rglob("cat/eta.md")]
    assert len(expected) == 5
    assert _call_twice("cat/eta", [root]) == expected


def test_a_qualified_flat_file_is_found_through_a_symlinked_directory(tmp_path, monkeypatch):
    """rglob("link/doc.md") follows a symlinked directory named in the pattern; the index walk enters
    real directories only, so such a name keeps the real glob."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    _flat(a, "real/doc.md")
    (a / "other").mkdir()
    os.symlink(a / "real", a / "other" / "link", target_is_directory=True)
    assert _call_twice("link/doc", [a]) == [(None, a / "other" / "link" / "doc.md")]
    assert _call_twice("real/doc", [a]) == [(None, a / "real" / "doc.md")]
    # A name that does not go through the link still costs no glob.
    _cands("real/doc", [a])
    monkeypatch.setattr(pathlib.Path, "rglob", lambda *a, **k: pytest.fail("rglob ran"))
    assert _cands("real/doc", [a]) == [(None, a / "real" / "doc.md")]
    assert _cands("alpha", [a]) == [(a / "cat" / "alpha", a / "cat" / "alpha" / "SKILL.md")]


def test_a_clear_between_the_check_and_the_rebuild_rebuilds(tree, monkeypatch):
    """A skill_manage write clears the index. A lookup that had already found its index stale and was
    about to rebuild it must build, not read the entry the clear removed."""
    a, b = tree
    _cands("alpha", [a, b])
    _skill(a, "cat/theta")  # the index of a is stale now
    real, fired = st._dir_tree_signature, []

    def signature_then_clear(root, sig, *args, **kwargs):
        real(root, sig, *args, **kwargs)
        if not fired:
            fired.append(1)
            st.clear_skill_lookup_index()

    monkeypatch.setattr(st, "_dir_tree_signature", signature_then_clear)
    assert _cands("theta", [a, b]) == [(a / "cat/theta", a / "cat/theta/SKILL.md")]
    assert fired


def test_a_directory_that_becomes_a_skill_hides_its_support_dirs_at_once(tmp_path):
    """The catalog walk skips references/ (and the other support dirs) of a directory that holds a
    SKILL.md. Below the signature depth a directory can gain one without the index noticing, and a
    skill under its references/ must stop being a candidate on the next call all the same."""
    a = tmp_path / "a"
    owner = a.joinpath(*"1/2/3/4/5/6/g".split("/"))
    child = _skill(owner, "references/child")
    assert _cands("child", [a]) == [(child, child / "SKILL.md")]
    index = st._SKILL_LOOKUP_INDEX[str(a)]
    (owner / "SKILL.md").write_text("---\nname: g\ndescription: d\n---\nbody\n", encoding="utf-8")
    assert _cands("child", [a]) == []
    assert st._SKILL_LOOKUP_INDEX[str(a)] is index, "the index did not notice the change: the rule was read from disk"
    (owner / "SKILL.md").unlink()
    assert _cands("child", [a]) == [(child, child / "SKILL.md")]


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists every directory")
def test_a_flat_file_in_a_directory_that_cannot_be_listed_is_still_found(tmp_path):
    """rglob finds <dir>/doc.md by asking for that path, also in a directory it may enter but not
    list. The index walk cannot see the file, so such a tree keeps the real glob for flat files."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    doc = _flat(a, "outer/sealed/doc.md")
    os.chmod(a / "outer" / "sealed", 0o111)
    try:
        assert _call_twice("doc", [a]) == [(None, doc)]
        assert _call_twice("sealed/doc", [a]) == [(None, doc)]
        assert _call_twice("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    finally:
        os.chmod(a / "outer" / "sealed", 0o755)


def test_a_directory_that_fails_while_it_is_read_does_not_fail_the_lookup(tree, monkeypatch):
    """The tree walk and rglob skip a directory that fails while it is read (EIO in a backup
    directory); the index build must too."""
    a, b = tree
    real = os.scandir

    class Failing:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            raise OSError(5, "Input/output error")

        def close(self):
            pass

    monkeypatch.setattr(os, "scandir", lambda path=".": Failing() if str(path).rstrip("/").endswith(".curator_backups") else real(path))
    assert _call_twice("gamma", [a, b]) == [(None, a / "legacy/gamma.md")]
    assert _call_twice("alpha", [a, b]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]


def test_the_index_keeps_a_bounded_number_of_search_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "_LOOKUP_INDEX_MAX_DIRS", 2)
    dirs = [tmp_path / n for n in ("a", "b", "c")]
    for d in dirs:
        _skill(d, f"cat/in-{d.name}")
    for d in dirs:
        assert _cands(f"in-{d.name}", [d]) == [(d / f"cat/in-{d.name}", d / f"cat/in-{d.name}" / "SKILL.md")]
    assert list(st._SKILL_LOOKUP_INDEX) == [str(dirs[1]), str(dirs[2])]
    assert _cands("in-a", [dirs[0]]) == [(dirs[0] / "cat/in-a", dirs[0] / "cat/in-a/SKILL.md")]
    assert list(st._SKILL_LOOKUP_INDEX) == [str(dirs[2]), str(dirs[0])]


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters every directory")
def test_a_flat_file_in_a_directory_that_cannot_be_entered_is_not_found(tmp_path):
    """rglob asks whether <dir>/doc.md exists, and in a directory that can be listed but not entered
    the answer is no. The index knows the name from the listing and must ask too."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    _flat(a, "sealed/doc.md")
    os.chmod(a / "sealed", 0o444)
    try:
        assert _call_twice("doc", [a]) == []
        assert _call_twice("sealed/doc", [a]) == []
    finally:
        os.chmod(a / "sealed", 0o755)


def test_a_deleted_flat_file_stops_shadowing_the_next_copy_at_once(tmp_path):
    """A flat file deleted below the signature depth leaves the index as it was. It must still not be
    a candidate, or it would shadow the copy in the next search dir and the lookup would fail."""
    a, b = tmp_path / "a", tmp_path / "b"
    deep = _flat(a, "1/2/3/4/5/6/7/doc.md")
    other = _flat(b, "doc.md")
    assert _cands("doc", [a, b]) == [(None, deep), (None, other)]
    index = st._SKILL_LOOKUP_INDEX[str(a)]
    deep.unlink()
    assert _cands("doc", [a, b]) == [(None, other)]
    assert st._SKILL_LOOKUP_INDEX[str(a)] is index, "the index did not notice the change: the file was checked on disk"


def test_a_skill_load_does_not_drop_the_index(tmp_path, monkeypatch):
    """Every skill load rewrites <skills root>/.usage.json (temp file, then rename), which moves the
    root's mtime. The index must survive that, or every lookup after a load would rebuild it."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    assert _cands("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    index = st._SKILL_LOOKUP_INDEX[str(a)]
    before = os.stat(a).st_mtime_ns
    for n in range(3):
        (a / f".usage_{n}.tmp").write_text("{}", encoding="utf-8")
        os.replace(a / f".usage_{n}.tmp", a / ".usage.json")
        os.utime(a, ns=(before, before + (n + 1) * 1_000_000_000))  # a coarse clock must not hide the move
        calls = _count_costs(monkeypatch)
        assert _cands("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
        assert calls == {"frontmatter": 0, "rglob": []}
        assert st._SKILL_LOOKUP_INDEX[str(a)] is index
    # What a lookup can return from the root is still seen at once: a flat file and a new folder.
    flat = _flat(a, "rootflat.md")
    assert _cands("rootflat", [a]) == [(None, flat)]
    _skill(a, "newcat/beta")
    assert _cands("beta", [a]) == [(a / "newcat/beta", a / "newcat/beta/SKILL.md")]
    flat.unlink()
    assert _cands("rootflat", [a]) == []


def test_the_real_skill_view_keeps_the_index_across_loads(monkeypatch):
    """The tool as the gateway calls it: each successful view bumps the usage file in the profile's
    skills directory. The second and third loads must reuse the index the first one built."""
    from hermes_constants import get_hermes_home
    root = get_hermes_home() / "skills"
    for name in ("alpha", "beta", "gamma"):
        _skill(root, f"cat/{name}", body=f"body of {name}")
    import json
    assert json.loads(st._skill_view_with_bump({"name": "alpha"}))["success"] is True
    assert (root / ".usage.json").exists(), "the load did not write the usage file this test is about"
    assert json.loads(st._skill_view_with_bump({"name": "beta"}))["success"] is True
    index = st._SKILL_LOOKUP_INDEX[str(root)]
    calls = _count_costs(monkeypatch)
    assert json.loads(st._skill_view_with_bump({"name": "gamma"}))["success"] is True
    assert st._SKILL_LOOKUP_INDEX[str(root)] is index
    assert calls["rglob"] == []


def test_a_skill_file_that_is_a_pipe_does_not_hang_the_lookup(tmp_path):
    """Building the index reads every SKILL.md for its frontmatter name. One that is a FIFO with no
    writer would block the open forever, under the lock every other lookup needs."""
    import threading
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    (a / "cat" / "pipe").mkdir()
    fifo = a / "cat" / "pipe" / "SKILL.md"
    os.mkfifo(fifo)
    out = {}
    worker = threading.Thread(target=lambda: out.update(alpha=_cands("alpha", [a]), pipe=_cands("pipe", [a])), daemon=True)
    worker.start()
    worker.join(timeout=20)
    hung = worker.is_alive()
    if hung:  # let the blocked open finish so the lock is released for the tests that follow
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        worker.join(timeout=20)
    assert not hung, "the lookup blocked on a FIFO SKILL.md"
    assert out["alpha"] == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    assert out["pipe"] == [(a / "cat/pipe", fifo)]  # still found by its folder name, as before


def test_a_case_folding_directory_keeps_the_real_glob(tree, monkeypatch, tmp_path):
    """In a directory that matches names without regard to case the literal glob finds Foo.md for
    foo. The index is keyed by exact name, so such a tree keeps the real glob for flat files."""
    a, b = tree
    assert st._folds_case(str(a / "legacy"), {"gamma.md"}) is False  # the test filesystem is case-sensitive
    assert st._folds_case(str(a), {"123", "_"}) is False  # nothing to probe with
    monkeypatch.setattr(st, "_folds_case", lambda directory, names: directory == str(a / "legacy"))
    _cands("gamma", [a, b])
    calls = _count_costs(monkeypatch)
    assert _cands("gamma", [a, b]) == [(None, a / "legacy/gamma.md")]
    assert calls["rglob"] == ["gamma.md"]  # a only; b folds no case and uses its index
    assert calls["frontmatter"] == 0
