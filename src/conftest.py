# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the doctests collected from `src/` (`pytest --doctest-modules src`)."""


def pytest_configure(config: object) -> None:
    # As in `tests/next_tests/conftest.py`: fail loudly on a `DimensionRef` compared with a
    # dimension class (off by default for users).
    from gt4py.next import config as gtx_config

    gtx_config.STRICT_DIMENSION_REF_COMPARISON = True
