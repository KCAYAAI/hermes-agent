"""Plain-language Slack progress: card steps, collapsed history, and the 3-minute check-in."""

import time

from gateway import progress_text
from gateway.progress_text import friendly_step, heartbeat_text
from gateway.run_turn_runner import TurnRunner


def test_friendly_step_hides_commands_and_paths():
    assert friendly_step("terminal", "cd /home/aya && rm -rf x") == "Running"
    assert friendly_step("read_file", "/home/aya/secret.env") == "Reading"
    assert friendly_step("web_search", "pump pressure switch") == "Searching the web for pump pressure switch"
    assert friendly_step("mcp__highlevel_busybee__search_contacts", "Jane Doe") == "Using Highlevel Busybee"
    assert friendly_step("custom_tool", "anything") == "Using custom tool"


def test_heartbeat_text_is_plain_and_includes_extra_lines():
    text = heartbeat_text(6, "terminal, read_file", ["🔀 Subagents: 1 of 2 done"])
    assert text == "⏳ Still working · 6 min\nNow: Running\n🔀 Subagents: 1 of 2 done"
    assert heartbeat_text(3, "_thinking") == "⏳ Still working · 3 min"


def test_progress_line_providers_are_isolated():
    calls = []

    def good(**turn):
        calls.append(turn)
        return ["line A"]

    def bad(**turn):
        raise RuntimeError("boom")

    progress_text.register_progress_line_provider(good)
    progress_text.register_progress_line_provider(bad)
    try:
        assert progress_text.extra_progress_lines(chat_id="C1") == ["line A"]
        assert calls == [{"chat_id": "C1"}]
    finally:
        progress_text._providers[:] = [p for p in progress_text._providers if p not in (good, bad)]


def test_task_card_uses_friendly_steps_and_collapses_history():
    st = TurnRunner._TaskCardState(adapter=None)
    for i in range(7):
        st.apply_event({"type": "tool.started", "tool_call_id": f"c{i}", "tool_name": "terminal", "preview": "cd /x"})
        st.apply_event({"type": "tool.completed", "tool_call_id": f"c{i}", "is_error": i == 0})
    st.apply_event({"type": "tool.started", "tool_call_id": "w", "tool_name": "web_search", "preview": "pumps"})
    tasks = st.visible_tasks()
    assert tasks[0] == {"id": "earlier_steps", "title": "3 earlier steps (1 failed)", "status": "error"}
    assert [t["title"] for t in tasks[1:]] == ["Running"] * 4 + ["Searching the web for pumps"]
    assert "cd /x" not in str(tasks)
    assert st.title() == "Working"
    st.started = time.monotonic() - 125
    assert st.title() == "Working · 2 min"
    assert st.fallback_text().startswith("Working · 2 min\n- 3 earlier steps (1 failed) - failed")
