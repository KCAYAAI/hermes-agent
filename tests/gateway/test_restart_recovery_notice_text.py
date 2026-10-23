"""The person-visible restart-recovery notice comes from config (gateway.restart_recovery_notice*)."""
from gateway.session_restart_recovery import Verdict, thread_notice
from hermes_cli.config_defaults import DEFAULT_CONFIG

DEFAULT_TEXT = DEFAULT_CONFIG["gateway"]["restart_recovery_notice"]
DEFAULT_STEPS = DEFAULT_CONFIG["gateway"]["restart_recovery_notice_steps"]


def _v(*unsettled):
    return Verdict("uncertain", unsettled=tuple(unsettled), anchored=True)


def test_default_text_without_steps():
    assert thread_notice(_v()) == DEFAULT_TEXT
    assert DEFAULT_TEXT.startswith("♻️ ")
    assert "Hermes" not in DEFAULT_TEXT


def test_default_text_names_unsettled_steps():
    text = thread_notice(_v("jobber_update", "send_message"))
    assert text == f"{DEFAULT_TEXT} {DEFAULT_STEPS.replace('{steps}', 'jobber_update, send_message')}"


def test_config_overrides_both_parts():
    cfg = {"gateway": {"restart_recovery_notice": "Back online. Reply continue.",
                       "restart_recovery_notice_steps": "Unfinished: {steps}."}}
    assert thread_notice(_v(), cfg) == "Back online. Reply continue."
    assert thread_notice(_v("a"), cfg) == "Back online. Reply continue. Unfinished: a."


def test_stray_braces_never_raise():
    cfg = {"gateway": {"restart_recovery_notice": "{oops} restarted", "restart_recovery_notice_steps": "{x} {steps}"}}
    assert thread_notice(_v("a"), cfg) == "{oops} restarted {x} a"


def test_blank_or_wrong_type_falls_back_to_default():
    for bad in ({"gateway": {"restart_recovery_notice": "  "}}, {"gateway": {"restart_recovery_notice": 5}},
                {"gateway": None}, None, "nonsense"):
        assert thread_notice(_v(), bad) == DEFAULT_TEXT


def test_steps_template_without_placeholder_still_lists_steps():
    cfg = {"gateway": {"restart_recovery_notice_steps": "Some steps may have run."}}
    assert thread_notice(_v("a"), cfg) == f"{DEFAULT_TEXT} Some steps may have run. (a)"
