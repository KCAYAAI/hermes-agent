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
