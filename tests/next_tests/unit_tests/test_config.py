# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

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
