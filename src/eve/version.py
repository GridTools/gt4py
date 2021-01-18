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

"""Version specification."""

from typing import Optional, Union

from packaging.version import LegacyVersion, Version, parse
from pkg_resources import DistributionNotFound, get_distribution


__version__: str = "unknown"
__versioninfo__: Optional[Union[LegacyVersion, Version]] = None

try:
    __version__ = get_distribution("gt4py").version
    __versioninfo__ = parse(__version__)
except DistributionNotFound:
    pass

finally:
    del DistributionNotFound, LegacyVersion, Version, get_distribution, parse
