# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    del os.environ[env_var]
    assert config.env_flag_to_bool(env_var, default=False) is False
