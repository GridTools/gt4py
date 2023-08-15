# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import dataclasses
from typing import TypeAlias

from gt4py._core import definitions as core_defs


@dataclasses.dataclass(frozen=True)
class StorageInfo:  # TODO this needs to be designed (not sure if device is part of it)
    device: core_defs.Device


StorageInfoInterface: TypeAlias = StorageInfo  # and more?
