# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import os
import pathlib
from typing import Any
from unittest import mock

import pytest

from gt4py.next._config import Config, ConfigManager, OptionDescriptor, UpdateScope


class TestOptionDescriptorBasics:
    """Test basic OptionDescriptor functionality."""

    def test_descriptor_attribute_access(self) -> None:
        """Test attribute-style access to configuration options."""

        class TestConfig(ConfigManager):
            debug = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        assert cfg.debug is False

    def test_descriptor_with_default_value(self) -> None:
        """Test that descriptor stores and returns default values."""

        class TestConfig(ConfigManager):
            name = OptionDescriptor(option_type=str, default="test")

        cfg = TestConfig()
        assert cfg.name == "test"

    def test_descriptor_with_default_factory(self) -> None:
        """Test that descriptor uses default_factory to compute defaults."""

        class TestConfig(ConfigManager):
            base = OptionDescriptor(option_type=int, default=10)
            derived = OptionDescriptor(
                option_type=int, default_factory=lambda cfg: cfg.get("base") * 2
            )

        cfg = TestConfig()
        assert cfg.derived == 20


class TestStringValueParsing:
    """Test environment variable parsing and configuration."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("False", False),
            ("false", False),
            ("0", False),
            ("off", False),
            ("True", True),
            ("true", True),
            ("1", True),
            ("on", True),
        ],
    )
    def test_parse_bool(self, value, expected) -> None:
        """Test parsing boolean environment variables."""
        with mock.patch.dict(os.environ, {"GT4PY_TESTING_OPT": value}):

            class TestConfig(ConfigManager):
                testing_opt = OptionDescriptor(option_type=bool, default=False)

            cfg = TestConfig()
            assert cfg.testing_opt is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("42", 42),
            ("-5", -5),
            ("0", 0),
        ],
    )
    def test_parse_int(self, value, expected) -> None:
        """Test parsing integer environment variables."""
        with mock.patch.dict(os.environ, {"GT4PY_TESTING_OPT": value}):

            class TestConfig(ConfigManager):
                testing_opt = OptionDescriptor(option_type=int, default=0)

            cfg = TestConfig()
            assert cfg.testing_opt == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("/tmp/test", pathlib.Path("/tmp/test")),
            ("./relative/path", pathlib.Path("./relative/path")),
            ("~/user/path", pathlib.Path(os.environ["HOME"]) / "user" / "path"),
        ],
    )
    def test_parse_path(self, value, expected) -> None:
        """Test parsing pathlib.Path environment variables."""
        with mock.patch.dict(os.environ, {"GT4PY_TESTING_OPT": value}):

            class TestConfig(ConfigManager):
                testing_opt = OptionDescriptor(option_type=pathlib.Path, default=pathlib.Path("/"))

            cfg = TestConfig()
            assert cfg.testing_opt == expected

    def test_parse_enum(self) -> None:
        """Test parsing enum options from environment variables."""

        class Mode(enum.Enum):
            DEBUG = "debug"
            RELEASE = "release"

        with mock.patch.dict(os.environ, {"GT4PY_TESTING_OPT": "DEBUG"}):

            class TestConfig(ConfigManager):
                testing_opt = OptionDescriptor(option_type=Mode, default=Mode.RELEASE)

            cfg = TestConfig()
            assert cfg.testing_opt == Mode.DEBUG

    def test_custom_parser(self) -> None:
        """Test custom parser for environment variables."""

        def parse_list(s: str) -> list[str]:
            return s.split(",")

        with mock.patch.dict(os.environ, {"GT4PY_ITEMS": "a,b,c"}):

            class TestConfig(ConfigManager):
                items = OptionDescriptor(option_type=list, default=[], env_var_parser=parse_list)

            cfg = TestConfig()
            assert cfg.items == ["a", "b", "c"]

    def test_invalid_environment_variable_raises_error(self) -> None:
        """Test that invalid environment variables raise RuntimeError."""
        with mock.patch.dict(os.environ, {"GT4PY_NUM": "not_a_number"}):
            with pytest.raises(RuntimeError, match="Parsing"):

                class TestConfig(ConfigManager):
                    num = OptionDescriptor(option_type=int, default=0)

                TestConfig()


class TestConfigManagerBasics:
    """Test ConfigManager basic functionality."""

    def test_set_changes_global_value(self) -> None:
        """Test that set() changes the global configuration value."""

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10)

        cfg = TestConfig()
        cfg.set("value", 20)
        assert cfg.value == 20

    def test_set_via_attribute_assignment(self) -> None:
        """Test that setting via attribute assignment works."""

        class TestConfig(ConfigManager):
            debug = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        cfg.debug = True
        assert cfg.debug is True

    def test_get_rejects_unrecognized_option(self) -> None:
        """Test that get() raises AttributeError for unknown options."""

        class TestConfig(ConfigManager):
            opt = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        with pytest.raises(AttributeError, match="Unrecognized config option"):
            cfg.get("nonexistent")

    def test_set_rejects_unrecognized_option(self) -> None:
        """Test that set() raises AttributeError for unknown options."""

        class TestConfig(ConfigManager):
            opt = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        with pytest.raises(AttributeError, match="Unrecognized config option"):
            cfg.set("nonexistent", True)

    def test_set_blocked_during_context_override(self) -> None:
        """Test that set() is blocked while option is overridden in context."""

        class TestConfig(ConfigManager):
            opt = OptionDescriptor(option_type=int, default=10)

        cfg = TestConfig()
        with cfg.overrides(opt=20):
            with pytest.raises(AttributeError, match="overridden in a context manager"):
                cfg.set("opt", 30)

    def test_as_dict_returns_all_options(self) -> None:
        """Test that as_dict() returns all configuration options."""

        class TestConfig(ConfigManager):
            opt1 = OptionDescriptor(option_type=int, default=1)
            opt2 = OptionDescriptor(option_type=str, default="test")

        cfg = TestConfig()
        config_dict = cfg.as_dict()
        assert config_dict["opt1"] == 1
        assert config_dict["opt2"] == "test"

    def test_as_dict_reflects_context_overrides(self) -> None:
        """Test that as_dict() reflects active context overrides."""

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10)

        cfg = TestConfig()
        with cfg.overrides(value=99):
            assert cfg.as_dict()["value"] == 99


class TestConfigurationPrecedence:
    """Test configuration value precedence rules."""

    def test_environment_variable_overrides_default(self) -> None:
        """Test that environment variables override descriptor defaults."""
        with mock.patch.dict(os.environ, {"GT4PY_VALUE": "999"}):

            class TestConfig(ConfigManager):
                value = OptionDescriptor(option_type=int, default=100)

            cfg = TestConfig()
            assert cfg.value == 999

    def test_context_override_precedence_chain(self) -> None:
        """Test complete precedence: context > global > environment > default."""
        with mock.patch.dict(os.environ, {"GT4PY_NUM": "50"}):

            class TestConfig(ConfigManager):
                num = OptionDescriptor(option_type=int, default=10)

            cfg = TestConfig()
            assert cfg.num == 50  # Environment overrides default

            cfg.set("num", 100)
            assert cfg.num == 100  # Global overrides environment

            with cfg.overrides(num=200):
                assert cfg.num == 200  # Context overrides global

            assert cfg.num == 100  # Back to global after context

    def test_multiple_option_override(self) -> None:
        """Test overriding multiple options simultaneously."""

        class TestConfig(ConfigManager):
            opt1 = OptionDescriptor(option_type=int, default=1)
            opt2 = OptionDescriptor(option_type=str, default="a")
            opt3 = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        with cfg.overrides(opt1=10, opt2="b", opt3=True):
            assert cfg.opt1 == 10
            assert cfg.opt2 == "b"
            assert cfg.opt3 is True

    def test_nested_context_overrides(self) -> None:
        """Test nested context overrides."""

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=1)

        cfg = TestConfig()
        with cfg.overrides(value=10):
            assert cfg.value == 10
            with cfg.overrides(value=20):
                assert cfg.value == 20
            assert cfg.value == 10
        assert cfg.value == 1

    def test_override_rejects_unrecognized_options(self) -> None:
        """Test that overrides reject unknown option names."""

        class TestConfig(ConfigManager):
            opt = OptionDescriptor(option_type=bool, default=False)

        cfg = TestConfig()
        with pytest.raises(AttributeError, match="Unrecognized config options"):
            with cfg.overrides(nonexistent=True):
                pass

    def test_override_no_change_for_same_value(self) -> None:
        """Test that overriding with same value doesn't trigger unnecessary changes."""

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10)

        cfg = TestConfig()
        with cfg.overrides(value=10):
            assert cfg.value == 10


class TestValidation:
    """Test configuration option validation."""

    def test_validator_rejects_invalid_values(self) -> None:
        """Test that validators reject invalid values."""

        def positive_int(val: Any) -> None:
            if not isinstance(val, int) or val <= 0:
                raise ValueError("Must be positive")

        class TestConfig(ConfigManager):
            count = OptionDescriptor(option_type=int, default=1, validator=positive_int)

        cfg = TestConfig()
        with pytest.raises(ValueError, match="Must be positive"):
            cfg.set("count", -5)

    def test_type_check_validator(self) -> None:
        """Test that 'type_check' validator validates types."""

        class TestConfig(ConfigManager):
            name = OptionDescriptor(option_type=str, default="test", validator="type_check")

        cfg = TestConfig()
        with pytest.raises(TypeError):
            cfg.set("name", 123)

    def test_validator_accepts_valid_values(self) -> None:
        """Test that validators accept valid values."""

        def even_int(val: Any) -> None:
            if not isinstance(val, int) or val % 2 != 0:
                raise ValueError("Must be even")

        class TestConfig(ConfigManager):
            num = OptionDescriptor(option_type=int, default=2, validator=even_int)

        cfg = TestConfig()
        cfg.set("num", 42)
        assert cfg.num == 42

    def test_validator_applied_during_context_override(self) -> None:
        """Test that validators are applied during context overrides."""

        def positive(val: Any) -> None:
            if val <= 0:
                raise ValueError("Must be positive")

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=1, validator=positive)

        cfg = TestConfig()
        with pytest.raises(ValueError, match="Must be positive"):
            with cfg.overrides(value=-1):
                pass


class TestUpdateCallbacks:
    """Test option update callbacks."""

    def test_callback_invoked_on_global_set(self) -> None:
        """Test that callbacks are invoked when using set()."""
        callback_calls: list[tuple[Any, Any, UpdateScope]] = []

        def track_changes(new_val: Any, old_val: Any, scope: UpdateScope) -> None:
            callback_calls.append((new_val, old_val, scope))

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10, update_callback=track_changes)

        cfg = TestConfig()
        cfg.set("value", 20)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (20, 10, UpdateScope.GLOBAL)

    def test_callback_invoked_on_context_override(self) -> None:
        """Test that callbacks are invoked during context overrides."""
        callback_calls: list[tuple[Any, Any, UpdateScope]] = []

        def track_changes(new_val: Any, old_val: Any, scope: UpdateScope) -> None:
            callback_calls.append((new_val, old_val, scope))

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10, update_callback=track_changes)

        cfg = TestConfig()
        with cfg.overrides(value=20):
            pass

        # Should have one call on enter and one on exit
        assert any(call[2] == UpdateScope.CONTEXT for call in callback_calls)

    def test_no_callback_for_no_change(self) -> None:
        """Test that callbacks are not invoked when override value equals current value."""
        callback_calls: list[Any] = []

        def track_changes(new_val: Any, old_val: Any, scope: UpdateScope) -> None:
            callback_calls.append("called")

        class TestConfig(ConfigManager):
            value = OptionDescriptor(option_type=int, default=10, update_callback=track_changes)

        cfg = TestConfig()
        with cfg.overrides(value=10):  # Same as default
            pass

        assert len(callback_calls) == 0


def test_gt4py_config_class() -> None:
    """Test the actual Config class for GT4Py."""

    assert isinstance(Config, type)
    cfg = Config()

    assert "debug" in cfg._option_descriptors_()
    assert isinstance(cfg.debug, bool)

    assert isinstance(cfg.build_cache_dir, pathlib.Path)
    assert str(cfg.build_cache_dir).endswith(".gt4py_cache")
