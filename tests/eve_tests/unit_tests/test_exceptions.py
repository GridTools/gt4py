# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
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


from __future__ import annotations

import pytest

from eve import exceptions


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
