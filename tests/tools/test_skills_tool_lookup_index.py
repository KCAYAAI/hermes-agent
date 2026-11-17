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
    index = st._SKILL_LOOKUP_INDEX[os.path.realpath(a)]
    (owner / "SKILL.md").write_text("---\nname: g\ndescription: d\n---\nbody\n", encoding="utf-8")
    assert _cands("child", [a]) == []
    assert st._SKILL_LOOKUP_INDEX[os.path.realpath(a)] is index, "the index did not notice the change: the rule was read from disk"
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
    assert list(st._SKILL_LOOKUP_INDEX) == [os.path.realpath(dirs[1]), os.path.realpath(dirs[2])]
    assert _cands("in-a", [dirs[0]]) == [(dirs[0] / "cat/in-a", dirs[0] / "cat/in-a/SKILL.md")]
    assert list(st._SKILL_LOOKUP_INDEX) == [os.path.realpath(dirs[2]), os.path.realpath(dirs[0])]


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
    index = st._SKILL_LOOKUP_INDEX[os.path.realpath(a)]
    deep.unlink()
    assert _cands("doc", [a, b]) == [(None, other)]
    assert st._SKILL_LOOKUP_INDEX[os.path.realpath(a)] is index, "the index did not notice the change: the file was checked on disk"


def test_a_skill_load_does_not_drop_the_index(tmp_path, monkeypatch):
    """Every skill load rewrites <skills root>/.usage.json (temp file, then rename), which moves the
    root's mtime. The index must survive that, or every lookup after a load would rebuild it."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    assert _cands("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    index = st._SKILL_LOOKUP_INDEX[os.path.realpath(a)]
    before = os.stat(a).st_mtime_ns
    for n in range(3):
        (a / f".usage_{n}.tmp").write_text("{}", encoding="utf-8")
        os.replace(a / f".usage_{n}.tmp", a / ".usage.json")
        os.utime(a, ns=(before, before + (n + 1) * 1_000_000_000))  # a coarse clock must not hide the move
        calls = _count_costs(monkeypatch)
        assert _cands("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
        assert calls == {"frontmatter": 0, "rglob": []}
        assert st._SKILL_LOOKUP_INDEX[os.path.realpath(a)] is index
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
    index = st._SKILL_LOOKUP_INDEX[os.path.realpath(root)]
    calls = _count_costs(monkeypatch)
    assert json.loads(st._skill_view_with_bump({"name": "gamma"}))["success"] is True
    assert st._SKILL_LOOKUP_INDEX[os.path.realpath(root)] is index
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


def test_links_to_one_tree_share_one_index(tmp_path, monkeypatch):
    """KC's profiles each have a skills dir that is a link to one shared tree. They must share one
    index (or each link would rebuild its own every 30 s), and each caller still gets paths under
    the dir it asked for."""
    real = tmp_path / "shared"
    _skill(real, "cat/alpha")
    _flat(real, "legacy/gamma.md")
    one, two = tmp_path / "one", tmp_path / "two"
    os.symlink(real, one, target_is_directory=True)
    os.symlink(real, two, target_is_directory=True)
    assert _cands("alpha", [one]) == [(one / "cat/alpha", one / "cat/alpha/SKILL.md")]
    calls = _count_costs(monkeypatch)
    assert _cands("alpha", [two]) == [(two / "cat/alpha", two / "cat/alpha/SKILL.md")]
    assert _cands("gamma", [two]) == [(None, two / "legacy/gamma.md")]
    assert _cands("alpha", [real]) == [(real / "cat/alpha", real / "cat/alpha/SKILL.md")]
    assert calls == {"frontmatter": 0, "rglob": []}
    assert list(st._SKILL_LOOKUP_INDEX) == [os.path.realpath(real)]


def test_a_search_dir_that_is_itself_a_skill_is_found_by_the_callers_dir_name(tmp_path):
    """The folder name of a SKILL.md in the search dir itself is the name of the dir the caller
    asked for, not the name of the real directory the index is keyed by."""
    real = tmp_path / "shared"
    _skill(real, "")
    one = tmp_path / "one"
    os.symlink(real, one, target_is_directory=True)
    assert _call_twice("one", [one]) == [(one, one / "SKILL.md")]
    assert _call_twice("shared", [one]) == [(one, one / "SKILL.md")]  # its frontmatter name
    assert _call_twice("shared", [real]) == [(real, real / "SKILL.md")]
    assert _call_twice("two", [one]) == []


def test_two_search_dirs_that_are_one_directory_are_checked_once_per_lookup(tmp_path, monkeypatch):
    real = tmp_path / "shared"
    _skill(real, "cat/alpha")
    link = tmp_path / "link"
    os.symlink(real, link, target_is_directory=True)
    _cands("alpha", [link, real])
    checks = []
    current = st._index_is_current
    monkeypatch.setattr(st, "_index_is_current", lambda key, index: checks.append(key) or current(key, index))
    assert _cands("alpha", [link, real]) == [(link / "cat/alpha", link / "cat/alpha/SKILL.md")]
    assert checks == [os.path.realpath(real)]


def test_a_root_folder_swapped_for_a_link_is_seen_at_once(tmp_path):
    """The glob does not walk into a linked directory. Swapping a folder of the root for a link to
    it changes only the root's mtime, which the signature no longer holds, so the signature keeps
    which root entries are links."""
    a, outside = tmp_path / "a", tmp_path / "outside"
    doc = _flat(a, "cat/doc.md")
    other = _flat(a, "dog/doc.md")
    assert sorted(_cands("doc", [a])) == sorted([(None, doc), (None, other)])
    before = (os.stat(a / "cat").st_mtime_ns, os.stat(a).st_mtime_ns)
    outside.mkdir()
    os.rename(a / "cat", outside / "cat")
    os.symlink(outside / "cat", a / "cat", target_is_directory=True)
    os.utime(outside / "cat", ns=(before[0], before[0]))
    assert _cands("doc", [a]) == [(None, other)]
    (a / "cat").unlink()
    os.rename(outside / "cat", a / "cat")
    assert sorted(_cands("doc", [a])) == sorted([(None, doc), (None, other)])


def test_the_case_probe_changes_ascii_letters_only(tmp_path):
    """A non-ASCII letter may have no partner a folding filesystem folds it to (dotless i), so the
    probe flips ASCII letters only; a directory it cannot probe counts as folding."""
    d = tmp_path / "d"
    d.mkdir()
    probed = []
    real = os.lstat
    try:
        def probe(path, *a, **kw):
            if os.path.dirname(path) == str(d):
                probed.append(os.path.basename(path))
            return real(path, *a, **kw)
        os.lstat = probe
        assert st._folds_case(str(d), {"\u0131Foo.md"}) is False
        assert probed and set(probed) == {"\u0131fOO.MD"}
        assert st._folds_case(str(d), {"a.md", "A.MD"}) is False  # both spellings exist: no folding
        assert st._folds_case(str(d), {"\u043a\u0430\u0442"}) is True  # cased, no ASCII letter: cannot tell
        assert st._folds_case(str(d), {"2024", "_", "\u65e5\u672c"}) is False  # nothing with case
    finally:
        os.lstat = real


def test_a_skill_file_that_cannot_be_read_has_no_frontmatter_name(tmp_path, monkeypatch):
    """Any failure to read one SKILL.md means no frontmatter name for it, as before; it must not
    fail the build and with it every lookup in that search dir."""
    a = tmp_path / "a"
    _skill(a, "cat/alpha")
    _skill(a, "cat/renamed", fm_name="beta")
    real_open = open

    def failing_open(file, *args, **kwargs):
        if isinstance(file, int):
            if kwargs.get("closefd", True):
                os.close(file)
            raise MemoryError("too large")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    assert _cands("alpha", [a]) == [(a / "cat/alpha", a / "cat/alpha/SKILL.md")]
    assert _cands("beta", [a]) == []
    monkeypatch.undo()
    st.clear_skill_lookup_index()
    assert _cands("beta", [a]) == [(a / "cat/renamed", a / "cat/renamed/SKILL.md")]


@pytest.mark.parametrize("names", [["\udcffX", "Foo.md"], ["Foo.md", "\udcffA.md"]])
def test_opaque_filename_cannot_prove_a_directory_is_case_sensitive(tmp_path, names):
    """Non-strict Linux casefold treats undecodable names as opaque, even when valid names
    in the same directory fold case. No probe in that directory can clear the glob fallback."""
    directory = tmp_path / "cat"
    directory.mkdir()
    for name in names:
        (directory / name).write_text("body")
    assert st._folds_case(str(directory), names) is True
    assert _call_twice("Foo", [tmp_path]) == [(None, directory / "Foo.md")]
    assert st._SKILL_LOOKUP_INDEX[os.path.realpath(tmp_path)].plain is False


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="Linux has a 40-link resolution limit")
def test_canonical_index_does_not_shadow_a_readable_skill_with_an_unreachable_alias(tmp_path):
    """Two links in the caller's root plus 39 in foo exceed the OS limit. The canonical index
    can reach foo but the caller's walk cannot. The readable lower-priority copy must win."""
    from agent.skill_utils import iter_skill_index_files

    real = tmp_path / "real"
    real.mkdir()
    target = _skill(tmp_path, "target", fm_name="foo")
    links = tmp_path / "links"
    links.mkdir()
    previous = target
    for n in range(38):
        link = links / f"link{n}"
        link.symlink_to(previous, target_is_directory=True)
        previous = link
    (real / "foo").symlink_to(previous, target_is_directory=True)
    middle, caller = tmp_path / "middle", tmp_path / "caller"
    middle.symlink_to(real, target_is_directory=True)
    caller.symlink_to(middle, target_is_directory=True)
    external = tmp_path / "external"
    other = _skill(external, "foo")
    assert (real / "foo/SKILL.md").is_file()
    assert not os.path.lexists(caller / "foo/SKILL.md")
    assert list(iter_skill_index_files(caller, "SKILL.md")) == []
    assert _call_twice("foo", [caller, external]) == [(other, other / "SKILL.md")]
    assert _cands("foo", [real]) == [(real / "foo", real / "foo/SKILL.md")]


@pytest.mark.platforms("linux")
def test_a_final_skill_file_symlink_does_not_invent_a_frontmatter_match(tmp_path):
    """lexists stops before the final symlink; frontmatter lookup must use caller-path access."""
    real = tmp_path / "real"
    (real / "renamed").mkdir(parents=True)
    target = _flat(tmp_path, "outside/target.md", "---\nname: foo\n---\nbody")
    previous = target
    for n in range(38):
        link = tmp_path / "outside" / f"filelink{n}"
        link.symlink_to(previous)
        previous = link
    (real / "renamed/SKILL.md").symlink_to(previous)
    middle, caller = tmp_path / "middle", tmp_path / "caller"
    middle.symlink_to(real, target_is_directory=True)
    caller.symlink_to(middle, target_is_directory=True)
    external = tmp_path / "external"
    good = _skill(external, "foo")
    assert os.path.lexists(caller / "renamed/SKILL.md")
    assert not (caller / "renamed/SKILL.md").exists()
    assert st._safe_frontmatter(caller / "renamed/SKILL.md") == {}
    assert _call_twice("foo", [caller, external]) == [(good, good / "SKILL.md")]
    # Folder-name hits retain the original walk's broken/unreadable-link filename semantics.
    assert _cands("renamed", [caller]) == [(caller / "renamed", caller / "renamed/SKILL.md")]


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("root_length", [4080, 4090])
@pytest.mark.parametrize("via_directory_link", [False, True])
def test_short_root_alias_is_used_when_canonical_paths_exceed_the_os_limit(tmp_path, root_length, via_directory_link):
    """A short caller alias can read a file that an absolute canonical walk cannot open or list."""
    real = tmp_path / "long"
    real.mkdir()
    while len(os.fsencode(real)) < root_length:
        size = min(200, root_length - len(os.fsencode(real)) - 1)
        assert size > 0
        real = real / ("d" * size)
        real.mkdir()
    caller = tmp_path / "short"
    caller.symlink_to(real, target_is_directory=True)
    base = caller
    outside = tmp_path / "outside-dir"
    if via_directory_link:
        outside.mkdir()
        (caller / "link").symlink_to(outside, target_is_directory=True)
        base = caller / "link"
    good = _skill(base, "nested", fm_name="beta")
    try:
        assert st._safe_frontmatter(good / "SKILL.md")["name"] == "beta"
        assert _call_twice("beta", [caller]) == [(good, good / "SKILL.md")]
    finally:
        # Use the short spelling during cleanup too.
        (good / "SKILL.md").unlink()
        good.rmdir()
        if via_directory_link:
            (caller / "link").unlink()


@pytest.mark.platforms("linux")
def test_overlong_case_probe_requests_the_caller_walk(tmp_path):
    """No casefold mount is required to verify that ENAMETOOLONG is not a negative probe.
    The original literal lookup through the short alias still reads the same flat file."""
    import errno
    real = tmp_path / "long-probe"
    real.mkdir()
    while len(os.fsencode(real)) < 4090:
        size = min(200, 4090 - len(os.fsencode(real)) - 1)
        real = real / ("d" * size)
        real.mkdir()
    caller = tmp_path / "short-probe"
    caller.symlink_to(real, target_is_directory=True)
    doc = _flat(caller, "cat/Foo.md")
    try:
        with pytest.raises(OSError) as exc:
            st._folds_case(str(real / "cat"), {"Foo.md"})
        assert exc.value.errno == errno.ENAMETOOLONG
        assert _call_twice("Foo", [caller]) == [(None, doc)]
        # A partial index from the canonical path was not published.
        assert os.path.realpath(caller) not in st._SKILL_LOOKUP_INDEX
    finally:
        doc.unlink()
        doc.parent.rmdir()


@pytest.mark.parametrize("error", [13, 5])
def test_inconclusive_case_probe_keeps_the_glob(tmp_path, monkeypatch, error):
    """Permission and I/O failures do not establish case-sensitive matching."""
    directory = tmp_path / "uncertain"
    directory.mkdir()
    real = os.lstat

    def denied(path, *a, **kw):
        if os.fspath(path) == str(directory / "fOO.MD"):
            raise OSError(error, "inconclusive probe")
        return real(path, *a, **kw)

    monkeypatch.setattr(os, "lstat", denied)
    assert st._folds_case(str(directory), {"Foo.md"}) is True


def test_close_failure_discards_the_frontmatter_alias(tmp_path, monkeypatch):
    """read_text fails as a whole if close fails. A cached alias must not survive that failure."""
    local, external = tmp_path / "local", tmp_path / "external"
    bad = _skill(local, "renamed", fm_name="foo") / "SKILL.md"
    good = _skill(external, "foo")
    real_open, real_close = os.open, os.close
    bad_fds = set()

    def track_open(path, *a, **kw):
        fd = real_open(path, *a, **kw)
        if os.fspath(path) == str(bad):
            bad_fds.add(fd)
        return fd

    def failing_close(fd):
        real_close(fd)
        if fd in bad_fds:
            bad_fds.remove(fd)
            raise OSError(5, "close EIO")

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", failing_close)
    assert st._frontmatter_name(bad) is None
    assert _call_twice("foo", [local, external]) == [(good, good / "SKILL.md")]
    assert not bad_fds


def test_failed_text_wrapper_does_not_close_a_reused_descriptor(tmp_path, monkeypatch):
    """A failed owning text wrapper can close the supplied fd before raising. Another reader
    can then acquire that number. Cleanup must not close the other reader's descriptor."""
    md = _skill(tmp_path, "alpha") / "SKILL.md"
    other = _flat(tmp_path, "other.md", "other reader")
    real_open = open
    acquired = []

    def failed_wrapper(file, *args, **kwargs):
        if not isinstance(file, int):
            return real_open(file, *args, **kwargs)
        if kwargs.get("closefd", True):
            os.close(file)  # what the owning wrapper is permitted to do during failure
        replacement = os.open(other, os.O_RDONLY)  # the interleaved reader
        acquired.append(replacement)
        raise MemoryError("wrapper construction failed")

    monkeypatch.setattr("builtins.open", failed_wrapper)
    try:
        assert st._frontmatter_name(md) is None
        assert os.read(acquired[0], 100) == b"other reader"
    finally:
        for fd in acquired:
            try:
                os.close(fd)
            except OSError:
                pass


@pytest.mark.platforms("linux")
@pytest.mark.skipif(os.geteuid() == 0, reason="root can traverse mode-000 parents")
def test_descriptor_root_alias_remains_usable_when_canonical_parent_is_inaccessible(tmp_path):
    """An open directory descriptor remains usable after an ancestor loses search permission.
    Its /proc alias can find a skill even when canonical-root discovery cannot list the root."""
    private = tmp_path / "private"
    shared = private / "shared"
    _skill(shared, "renamed", fm_name="foo")
    fd = os.open(shared, os.O_RDONLY | os.O_DIRECTORY)
    caller = tmp_path / "caller"
    caller.symlink_to(f"/proc/self/fd/{fd}", target_is_directory=True)
    private.chmod(0)
    try:
        with pytest.raises(PermissionError):
            os.scandir(shared)
        assert st._safe_frontmatter(caller / "renamed/SKILL.md")["name"] == "foo"
        assert _call_twice("foo", [caller]) == [(caller / "renamed", caller / "renamed/SKILL.md")]
    finally:
        private.chmod(0o700)
        os.close(fd)
