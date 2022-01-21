# GT4Py Project - GridTools Framework
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

from numpy import float32, float64, int32, int64

from functional.common import Field


__all__ = ["Field", "float32", "float64", "int32", "int64"]

TYPE_BUILTINS = [Field, float, float32, float64, int, int32, int64, bool, tuple]
TYPE_BUILTIN_NAMES = [t.__name__ for t in TYPE_BUILTINS]

EXTERNALS_MODULE_NAME = "__externals__"
MODULE_BUILTIN_NAMES = [EXTERNALS_MODULE_NAME]

ALL_BUILTIN_NAMES = TYPE_BUILTIN_NAMES + MODULE_BUILTIN_NAMES

BUILTINS = {name: globals()[name] for name in __all__}
