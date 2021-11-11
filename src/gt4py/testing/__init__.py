# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

__all__ = ["field", "global_name", "none", "parameter", "StencilTestSuite"]
try:
    from .input_strategies import field, global_name, none, parameter
    from .suites import StencilTestSuite
except ModuleNotFoundError as e:
    if "pytest" in str(e) or "hypothesis" in str(e):
        raise RuntimeError(
            "Missing 'testing' extra dependencies. Reinstall GT4Py as 'gt4py[testing]'"
        ) from e
    else:
        raise e
