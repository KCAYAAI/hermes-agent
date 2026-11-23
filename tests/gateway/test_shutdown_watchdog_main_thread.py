"""KC carry: a watchdog dump with no monitored loop thread still writes the main thread first.

faulthandler stops at 100 threads, newest first, so on a busy gateway the event loop's stack
(the main thread) can fall off the end of the all-threads dump.
"""
from gateway import shutdown_watchdog as wd


def test_a_dump_without_a_loop_thread_writes_the_main_thread_stack_first(tmp_path):
    dump = tmp_path / "watchdog.log"
    wd._write_watchdog_dump(dump, delay_s=1.0, snapshot=None)
    text = dump.read_text(encoding="utf-8")
    head, _, rest = text.partition("--- faulthandler dump (all threads) ---")
    assert "--- main thread (event loop) ---" in head
    assert "--- end main thread ---" in head
    assert rest  # the all-threads dump still follows


def test_a_dump_with_a_loop_thread_keeps_the_loop_section_only(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "_format_loop_thread_section", lambda _tid: "--- loop section ---\n")
    dump = tmp_path / "watchdog.log"
    wd._write_watchdog_dump(dump, delay_s=1.0, snapshot=None, loop_thread_id=123)
    head = dump.read_text(encoding="utf-8").partition("--- faulthandler dump (all threads) ---")[0]
    assert "--- loop section ---" in head
    assert "--- main thread (event loop) ---" not in head


def test_the_main_thread_section_reads_no_source_and_no_locals(monkeypatch):
    """The main-thread section renders like upstream's loop section: file, line and function only,
    no linecache read on the kill path and no frame locals."""
    source_reads = []
    with monkeypatch.context() as patched:  # undone before the asserts: pytest reads source
        patched.setattr("linecache.getline", lambda *args, **_: source_reads.append(args) or "")
        patched.setattr("linecache.updatecache", lambda *args, **_: source_reads.append(args) or [])
        section = wd._main_thread_stack()
    assert source_reads == [], "the main-thread section read a source file on the kill path"
    assert section.startswith("--- main thread (event loop) ---\n")
    assert section.endswith("--- end main thread ---\n")
    assert 'File "' in section and ", line " in section
