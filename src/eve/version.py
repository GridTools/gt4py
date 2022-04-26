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


from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Final, Optional, Union

from packaging.version import LegacyVersion, Version, parse


# TODO(egparedes): Set up proper versioning scheme after migrating repo to new location
try:
    _version: str = version("gt4py-functional")
except PackageNotFoundError:
    _version = "X.X.X.unknown"


__version__: Final = _version
__versioninfo__: Final[Optional[Union[LegacyVersion, Version]]] = parse(__version__)
