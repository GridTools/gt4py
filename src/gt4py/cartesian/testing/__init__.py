# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

__all__ = ["StencilTestSuite", "field", "global_name", "none", "parameter"]
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
