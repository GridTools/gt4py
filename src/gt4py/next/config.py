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
from typing import Final


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


def env_flag_to_bool(flag_value: str) -> bool:
    """Like in gt4py.cartesian, env vars for flags should be set to '0' or '1'."""
    match flag_value:
        case "0" | "1":
            return bool(int(flag_value))
        case _:
            raise ValueError("GT4Py flag environment variables must have value '0' or '1'.")


_PREFIX: Final[str] = "GT4PY"

#: Master debug flag
#: Changes defaults for all the other options to be as helpful for debugging as possible.
#: Does not override values set in environment variables.
DEBUG: Final[bool] = env_flag_to_bool(os.environ.get(f"{_PREFIX}_DEBUG", "0"))

#: Where generated code projects should be persisted.
#: Only active if BUILD_CACHE_LIFETIME is set to PERSISTENT
BUILD_CACHE_DIR: Final[pathlib.Path] = (
    pathlib.Path(os.environ.get(f"{_PREFIX}_BUILD_CACHE_DIR", tempfile.gettempdir()))
    / "gt4py_cache"
)


#: Whether generated code projects should be kept around between runs.
#: - SESSION: generated code projects get destroyed when the interpreter shuts down
#: - PERSISTENT: generated code projects are written to BUILD_CACHE_DIR and persist between runs
BUILD_CACHE_LIFETIME: Final[BuildCacheLifetime] = getattr(
    BuildCacheLifetime,
    os.environ.get(f"{_PREFIX}_BUILD_CACHE_LIFETIME", "persistent" if DEBUG else "session").upper(),
)

#: Build type to be used when CMake is used to compile generated code.
#: Might have no effect when CMake is not used as part of the toolchain.
CMAKE_BUILD_TYPE: Final[CMakeBuildType] = getattr(
    CMakeBuildType,
    os.environ.get(f"{_PREFIX}_CMAKE_BUILD_TYPE", "debug" if DEBUG else "release").upper(),
)
