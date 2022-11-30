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


import gt4py


def test_version():
    assert isinstance(gt4py.version.__version__, str)
    assert len(gt4py.version.__version__) and all(
        len(p) for p in gt4py.version.__version__.split(".")
    )
    assert gt4py.version.__version__ == gt4py.__version__


def test_version_info():
    from packaging.version import Version

    assert isinstance(gt4py.version.__version_info__, Version)
    assert gt4py.version.__version_info__.release == tuple(
        int(p) for p in gt4py.version.__version__.split(".")
    )
    assert (0, 1) <= gt4py.version.__version_info__.release < (0, 2)
    assert gt4py.version.__version_info__ == gt4py.__version_info__
