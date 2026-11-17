"""``session_store_max_age_days`` set in config.yaml reaches the gateway (KC behavior 40).

The gateway builds its settings from a flat dict and only copies the config.yaml keys listed in
``gateway/config_loader.py::_TOPLEVEL_BRIDGE``. This key was missing from that list, so a value in
config.yaml (top level or under ``gateway:``) was silently ignored and only the legacy gateway.json
file could change the routing-index prune age. The tests load a real config.yaml through
``load_gateway_config`` (the gateway's startup path), not ``GatewayConfig.from_dict``.
"""
import json
from unittest.mock import patch

import pytest

import hermes_yaml as yaml


def _load(tmp_path, yaml_dict=None, gateway_json=None):
    from gateway.config import load_gateway_config

    if yaml_dict is not None:
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(yaml_dict), encoding="utf-8")
    if gateway_json is not None:
        (tmp_path / "gateway.json").write_text(json.dumps(gateway_json), encoding="utf-8")
    with patch("gateway.config.get_hermes_home", return_value=tmp_path):
        return load_gateway_config()


def test_default_is_90_when_nothing_sets_it(tmp_path):
    assert _load(tmp_path, {"model": {}}).session_store_max_age_days == 90


def test_top_level_yaml_key_is_read(tmp_path):
    assert _load(tmp_path, {"session_store_max_age_days": 7}).session_store_max_age_days == 7


def test_nested_gateway_yaml_key_is_read(tmp_path):
    # `hermes config set gateway.session_store_max_age_days 7` writes this shape.
    cfg = _load(tmp_path, {"gateway": {"session_store_max_age_days": 7}})
    assert cfg.session_store_max_age_days == 7


def test_top_level_wins_over_nested(tmp_path):
    cfg = _load(tmp_path, {"session_store_max_age_days": 5, "gateway": {"session_store_max_age_days": 30}})
    assert cfg.session_store_max_age_days == 5


def test_yaml_wins_over_legacy_gateway_json(tmp_path):
    cfg = _load(tmp_path, {"session_store_max_age_days": 7}, gateway_json={"session_store_max_age_days": 45})
    assert cfg.session_store_max_age_days == 7


def test_nested_yaml_wins_over_legacy_gateway_json(tmp_path):
    cfg = _load(tmp_path, {"gateway": {"session_store_max_age_days": 7}}, gateway_json={"session_store_max_age_days": 45})
    assert cfg.session_store_max_age_days == 7


def test_gateway_json_still_applies_when_yaml_is_silent(tmp_path):
    cfg = _load(tmp_path, {"model": {}}, gateway_json={"session_store_max_age_days": 45})
    assert cfg.session_store_max_age_days == 45


@pytest.mark.parametrize("value", [0, 7])
@pytest.mark.parametrize("nested", [False, True])
def test_a_malformed_platform_block_keeps_the_yaml_value(tmp_path, value, nested):
    # The platform merge after the bridge raises (``extra: null``) and the loader falls back to the
    # gateway.json view. The valid retention the bridge already read stays in force: falling back to
    # the default of 90 would prune routes an owner set 0 (never prune) to keep, as live did not.
    retention = {"session_store_max_age_days": value}
    yaml_dict = ({"gateway": {**retention, "platforms": {"discord": {"extra": None}}}} if nested
                 else {**retention, "platforms": {"discord": {"extra": None}}})
    cfg = _load(tmp_path, yaml_dict, gateway_json={"session_store_max_age_days": 45})
    assert cfg.session_store_max_age_days == value


@pytest.mark.parametrize("legacy, expected", [("1", 90), ('"text"', 90), ("[1, 2]", 90),
                                              ("null", 7)])  # null reads as no gateway.json
def test_a_non_mapping_gateway_json_still_boots_as_on_live(tmp_path, legacy, expected):
    # A gateway.json that is a number, string or list makes the bridge fail and the loader fall back;
    # the gateway still starts, as on live, with the default (the YAML value never reached a mapping).
    (tmp_path / "gateway.json").write_text(legacy, encoding="utf-8")
    cfg = _load(tmp_path, {"gateway": {"session_store_max_age_days": 7}})
    assert cfg.session_store_max_age_days == expected


def test_a_malformed_platform_block_without_a_yaml_value_keeps_gateway_json(tmp_path):
    cfg = _load(tmp_path, {"platforms": {"discord": {"extra": None}}}, gateway_json={"session_store_max_age_days": 45})
    assert cfg.session_store_max_age_days == 45


@pytest.mark.parametrize("raw, expected", [("abc", 90), (-3, 0), (0, 0), ("14", 14),
                                          (float("inf"), 90), (float("-inf"), 90)])
def test_yaml_values_are_coerced_like_gateway_json(tmp_path, raw, expected):
    # Same coercion as before (GatewayConfig.from_dict): bad -> 90, negative -> 0 (prune off).
    # YAML .inf / -.inf: int() raises OverflowError; the gateway must still start, with 90.
    assert _load(tmp_path, {"session_store_max_age_days": raw}).session_store_max_age_days == expected
