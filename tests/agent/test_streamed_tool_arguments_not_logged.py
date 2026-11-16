"""Tool-call arguments as streamed never reach the log through the stream assembler.

A provider that streams a tool call as the model writes it (the Claude subscription plugin from 0.4.0)
hands Hermes the arguments of a call that was cut at the output cap, did not parse, or was refused
mid-call. The assembler used to pass them to the argument repair, which logs what it was given: an
80-character preview when the repair works, up to 100,000 characters when it cannot. A cut
``write_file`` put the file's content in the log.

Two rules, both tested here:
- ``finish_reason == "length"`` is not repaired at all. The turn loop never runs such a call (it re-sends
  the request: ``agent/turn_truncation.py::_retry_truncated_tool_call``), so a repair changed nothing.
- Any other finished reply is repaired exactly as before, but the assembler's repair logs the tool name
  and the length only.
The repair function's other callers (stored history on the send path) log as upstream wrote them.
"""
import json
import logging

import pytest

from agent.chat_completion_helpers import _StreamingCall
from agent.message_sanitization import _repair_tool_call_arguments

PRIVATE = "private customer note 4411 Elm Street"
CUT_IN_STRING = '{"path": "note.txt", "content": "' + PRIVATE          # cannot be repaired
CUT_AFTER_VALUE = '{"content": "' + PRIVATE + '", "path": "note.txt"'   # repairable by closing the object
NOT_JSON = '{"path": /srv/note.txt, "content": ' + PRIVATE + '}'         # not a prefix of valid JSON
BROKEN = {"cut inside a string": CUT_IN_STRING, "cut after a value": CUT_AFTER_VALUE, "not json": NOT_JSON}


def _accumulated(arguments):
    return {0: {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": arguments}}}


def _assemble(raw, finish_reason, caplog):
    with caplog.at_level(logging.DEBUG):
        calls, truncated = _StreamingCall._assemble_tool_calls(_accumulated(raw), finish_reason)
    return calls[0].function, truncated


@pytest.mark.parametrize("raw", BROKEN.values(), ids=BROKEN.keys())
@pytest.mark.parametrize("finish_reason", ["length", "tool_calls", "stop", "content_filter", None])
def test_no_finish_reason_puts_streamed_arguments_in_the_log(raw, finish_reason, caplog):
    _assemble(raw, finish_reason, caplog)
    assert PRIVATE not in caplog.text and "note.txt" not in caplog.text


@pytest.mark.parametrize("raw", BROKEN.values(), ids=BROKEN.keys())
def test_a_call_cut_at_the_output_cap_is_flagged_and_kept_as_received(raw, caplog):
    function, truncated = _assemble(raw, "length", caplog)
    assert truncated is True and function.arguments == raw and function.args_repaired is False
    assert not caplog.text


@pytest.mark.parametrize("finish_reason", ["tool_calls", "stop", "content_filter"])
def test_a_finished_reply_is_repaired_as_before_and_the_warning_names_tool_and_length(finish_reason, caplog):
    # Repairable: the call the assembler returns is the one upstream's repair produces.
    function, truncated = _assemble(CUT_AFTER_VALUE, finish_reason, caplog)
    assert truncated is False and function.args_repaired is True
    assert json.loads(function.arguments) == {"content": PRIVATE, "path": "note.txt"}
    assert f"Repaired malformed tool_call arguments for write_file ({len(CUT_AFTER_VALUE)} characters)" in caplog.text
    caplog.clear()
    # Unrepairable: flagged, arguments kept as received, as before.
    function, truncated = _assemble(CUT_IN_STRING, finish_reason, caplog)
    assert truncated is True and function.arguments == CUT_IN_STRING
    assert f"Unrepairable tool_call arguments for write_file — replaced with empty object ({len(CUT_IN_STRING)} characters)" in caplog.text


def test_the_repair_result_does_not_depend_on_the_log_switch():
    for raw in (*BROKEN.values(), "", "None", '{"a": 1,}', '{"a": "x\ny"}'):
        assert _repair_tool_call_arguments(raw, "t", log_arguments=False) == _repair_tool_call_arguments(raw, "t")


def test_other_callers_of_the_repair_log_as_before(caplog):
    # Control: the default (the send path over stored history) is upstream's log line, text included.
    with caplog.at_level(logging.WARNING):
        assert _repair_tool_call_arguments(CUT_IN_STRING, "write_file") == "{}"
    assert PRIVATE in caplog.text


def test_valid_arguments_pass_through_untouched_and_unlogged(caplog):
    function, truncated = _assemble('{"path": "a.py"}', "length", caplog)
    assert truncated is False and function.arguments == '{"path": "a.py"}' and not caplog.text
