"""Regression tests for ``ClientLifecycleMixin._api_kwargs_have_image_parts``.

Pins the image-part detection contract and guards against the unhashable-type
crash: a message part whose ``type`` field holds a ``dict`` or ``list`` used to
raise ``TypeError: unhashable type`` because the value was tested for membership
in the image-type set without an ``isinstance(str)`` check (a ``set`` value did
not raise: set membership retries it as a ``frozenset``; it is kept as extra
coverage). The detector now only compares string types and lets non-string
parts fall through to the recursive traversal.
"""

from __future__ import annotations

import pytest

from agent.client_lifecycle import ClientLifecycleMixin


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url"},
        {"type": "input_image"},
    ],
)
def test_detects_string_image_types(part):
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [part]}) is True


@pytest.mark.parametrize(
    "part",
    [
        {"type": {"foo": "bar"}},
        {"type": ["image_url", "input_image"]},
        {"type": {"image_url", "input_image"}},
    ],
)
def test_unhashable_type_field_does_not_raise(part):
    # Regression: dict and list type fields previously raised TypeError.
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [part]}) is False


def test_detects_nested_image_part():
    api_kwargs = {"messages": [{"content": [{"type": "text"}, {"type": "image_url"}]}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True


def test_input_field_is_scanned():
    api_kwargs = {"input": [{"type": "input_image"}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is True


def test_no_image_part_returns_false():
    api_kwargs = {"messages": [{"type": "text", "content": "hello"}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False


@pytest.mark.parametrize("api_kwargs", [None, [], "not-a-dict"])
def test_non_dict_api_kwargs_returns_false(api_kwargs):
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False


def test_non_list_messages_field_returns_false():
    # A dict-valued "messages" field is not scanned (only list fields are).
    api_kwargs = {"messages": {"type": "image_url"}}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts(api_kwargs) is False


def test_non_string_type_still_scans_nested_image():
    # The guard must fall through to traversal, not stop at a non-string ``type``.
    part = {"type": {"slug": "x"}, "content": [{"type": "image_url"}]}
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [part]}) is True


def test_tool_call_arguments_with_object_type_field_do_not_raise():
    # Providers that keep assistant tool-call arguments as dicts in ``messages``
    # (for example a tool argument object with a nested ``type`` object) used to
    # crash the scan before the request was built, which activated the fallback
    # model for every later turn of the session.
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_details": [{"input": {"calls": [{"arguments": {
            "post": {"title": "Sales rep", "type": {"slug": "full-time"}},
        }}]}}],
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "update_post", "arguments": "{}"}}],
    }
    assert ClientLifecycleMixin._api_kwargs_have_image_parts({"messages": [message]}) is False
