# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from gt4py.eve import exceptions


def test_exception_types():
    with pytest.raises(TypeError):
        raise exceptions.EveTypeError()

    with pytest.raises(ValueError):
        raise exceptions.EveValueError()

    with pytest.raises(RuntimeError):
        raise exceptions.EveRuntimeError()


def test_default_exception_message():
    with pytest.raises(TypeError, match="my_data=23, your_data=42"):
        raise exceptions.EveTypeError(my_data=23, your_data=42)

    with pytest.raises(ValueError, match="my_data=23, your_data=42"):
        raise exceptions.EveValueError(my_data=23, your_data=42)

    with pytest.raises(RuntimeError, match="my_data=23, your_data=42"):
        raise exceptions.EveRuntimeError(my_data=23, your_data=42)


def test_custom_exception_message():
    with pytest.raises(TypeError, match="custom message"):
        raise exceptions.EveTypeError("This is a custom message", my_data=23, your_data=42)

    with pytest.raises(ValueError, match="custom message"):
        raise exceptions.EveValueError("This is a custom message", my_data=23, your_data=42)

    with pytest.raises(RuntimeError, match="custom message"):
        raise exceptions.EveRuntimeError("This is a custom message", my_data=23, your_data=42)
