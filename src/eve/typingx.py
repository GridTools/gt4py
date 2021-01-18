# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

"""Python version independent typings."""

# flake8: noqa
from typing import *  # isort:skip

import sys  # isort:skip
from typing import *
from typing import IO, BinaryIO, TextIO


if sys.version_info < (3, 8):
    from typing_extensions import Final, Literal, Protocol, TypedDict, runtime_checkable

del sys


T = TypeVar("T")
FrozenList = Tuple[T, ...]

AnyCallable = Callable[..., Any]
AnyNoneCallable = Callable[..., None]
AnyNoArgCallable = Callable[[], Any]

RootValidatorValuesType = Dict[str, Any]
RootValidatorType = Callable[[Type, RootValidatorValuesType], RootValidatorValuesType]
