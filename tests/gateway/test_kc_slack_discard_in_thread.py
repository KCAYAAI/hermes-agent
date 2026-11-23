"""KC: the Slack pause notice names commands a person can send from inside a thread.

Slack rejects slash commands in a thread, so `/hermes discard` cannot reach a paused thread's session.
The adapter rewrites a leading `!cmd` (also after the bot mention) to the gateway `/cmd`.
"""
from plugins.platforms.slack.adapter import _rewrite_known_bang_command


def test_the_words_the_slack_pause_notice_names_reach_the_gateway_commands():
    assert _rewrite_known_bang_command("!discard") == "/discard"
    assert _rewrite_known_bang_command("!reset") == "/reset"
