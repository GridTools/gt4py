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
from __future__ import annotations

import enum
import os
import pathlib
import tempfile


class BuildCacheLifetime(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


class CMakeBuildType(enum.Enum):
    def _generate_next_value_(name, start, count, last_values):
        return "".join(part.capitalize() for part in name.split("_"))

    DEBUG = enum.auto()
    RELEASE = enum.auto()
    REL_WITH_DEB_INFO = enum.auto()
    MIN_SIZE_REL = enum.auto()


_PREFIX = "GT4PY"
DEBUG = ...
BUILD_CACHE_DIR = (
    pathlib.Path(os.environ.get(f"{PREFIX}_BUILD_CACHE_DIR", tempfile.gettempdir())) / "gt4py_cache"
)

BUILD_CACHE_LIFETIME = getattr(
    BuildCacheLifetime, os.environ.get(f"{PREFIX}_BUILD_CACHE_LIFETIME", "session").upper()
)

CMAKE_BUILD_TYPE = getattr(CMakeBuildType, os.environ.get(f"{PREFIX}_BUILD_TYPE", "debug" if DEBUG else "release").upper())
