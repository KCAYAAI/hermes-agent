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


@pytest.mark.parametrize("raw, expected", [("abc", 90), (-3, 0), (0, 0), ("14", 14),
                                          (float("inf"), 90), (float("-inf"), 90)])
def test_yaml_values_are_coerced_like_gateway_json(tmp_path, raw, expected):
    # Same coercion as before (GatewayConfig.from_dict): bad -> 90, negative -> 0 (prune off).
    # YAML .inf / -.inf: int() raises OverflowError; the gateway must still start, with 90.
    assert _load(tmp_path, {"session_store_max_age_days": raw}).session_store_max_age_days == expected
