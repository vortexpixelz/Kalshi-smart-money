import textwrap

import pytest

from smart_money_detection.config import Config, load_config, validate_config


def test_load_config_merges_yaml_and_env(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            detector:
              zscore_threshold: 2.5
            kalshi:
              demo_mode: false
            """
        )
    )

    monkeypatch.setenv("KALSHI_API_KEY", "demo-key")
    monkeypatch.setenv("SMART_MONEY_ENSEMBLE__WEIGHTING_METHOD", "uniform")

    config = load_config(env_path="", yaml_path=str(yaml_path))

    assert config.detector.zscore_threshold == 2.5
    assert config.ensemble.weighting_method == "uniform"
    assert config.kalshi.api_key == "demo-key"
    assert config.kalshi.demo_mode is False


def test_validate_config_requires_kalshi_credentials_when_not_demo():
    config = Config()
    config.kalshi.demo_mode = False

    with pytest.raises(ValueError):
        validate_config(config)


def test_load_config_returns_independent_instances():
    config_one = load_config(env_path="")
    config_one.detector.zscore_threshold = 1.0

    config_two = load_config(env_path="")

    assert config_two.detector.zscore_threshold == 3.0
