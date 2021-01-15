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

"""Version specification."""


from packaging.version import parse
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("gt4py").version
    __versioninfo__ = parse(__version__)

except DistributionNotFound:
    __version__ = "unknown"
    __versioninfo__ = None

finally:
    del DistributionNotFound, get_distribution, parse
