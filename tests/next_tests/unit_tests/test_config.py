# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os

import pytest

from gt4py.next import config


@pytest.fixture
def env_var():
    """Just in case another test will ever use that environment variable."""
    env_var_name = "GT4PY_TEST_ENV_VAR"
    saved = os.environ.get(env_var_name, None)
    yield env_var_name
    if saved is not None:
        os.environ[env_var_name] = saved
    else:
        _ = os.environ.pop(env_var_name, None)


@pytest.mark.parametrize("value", ["False", "false", "0", "off"])
def test_env_flag_to_bool_false(env_var, value):
    os.environ[env_var] = value
    assert config.env_flag_to_bool(env_var, default=True) is False


@pytest.mark.parametrize("value", ["True", "true", "1", "on"])
def test_env_flag_to_bool_true(env_var, value):
    os.environ[env_var] = value
    assert config.env_flag_to_bool(env_var, default=False) is True


def test_env_flag_to_bool_invalid(env_var):
    os.environ[env_var] = "invalid value"
    with pytest.raises(ValueError):
        config.env_flag_to_bool(env_var, default=False)


def test_env_flag_to_bool_unset(env_var):
    _ = os.environ.pop(env_var, None)
    assert config.env_flag_to_bool(env_var, default=False) is False


def _load_fresh_config():
    """Execute a fresh copy of the `config` module, leaving `gt4py.next.config` untouched."""
    spec = importlib.util.spec_from_file_location("_fresh_gt4py_next_config", config.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "debug, format_sources, expected",
    [
        (None, None, False),
        ("1", None, True),
        ("0", "1", True),
        ("1", "0", False),
    ],
)
def test_format_sources_precedence(monkeypatch, debug, format_sources, expected):
    # avoid adding global warning filters when executing the module
    monkeypatch.setenv("GT4PY_SKIP_DACE_WARNINGS", "0")
    for name, value in (("GT4PY_DEBUG", debug), ("GT4PY_FORMAT_SOURCES", format_sources)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert _load_fresh_config().FORMAT_SOURCES is expected
