# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
    """
    CMake build types enum.

    Member values have to be valid CMake syntax.
    """

    DEBUG = "Debug"
    RELEASE = "Release"
    REL_WITH_DEB_INFO = "RelWithDebInfo"
    MIN_SIZE_REL = "MinSizeRel"


def env_flag_to_bool(name: str, default: bool) -> bool:
    """Recognize true or false signaling string values."""
    flag_value = None
    if name in os.environ:
        flag_value = os.environ[name].lower()
    match flag_value:
        case None:
            return default
        case "0" | "false" | "off":
            return False
        case "1" | "true" | "on":
            return True
        case _:
            raise ValueError(
                "Invalid GT4Py environment flag value: use '0 | false | off' or '1 | true | on'."
            )


_PREFIX: Final[str] = "GT4PY"

#: Master debug flag
#: Changes defaults for all the other options to be as helpful for debugging as possible.
#: Does not override values set in environment variables.
DEBUG: Final[bool] = env_flag_to_bool(f"{_PREFIX}_DEBUG", default=False)


#: Verbose flag for DSL compilation errors
VERBOSE_EXCEPTIONS: bool = env_flag_to_bool(
    f"{_PREFIX}_VERBOSE_EXCEPTIONS", default=True if DEBUG else False
)


#: Where generated code projects should be persisted.
#: Only active if BUILD_CACHE_LIFETIME is set to PERSISTENT
BUILD_CACHE_DIR: pathlib.Path = (
    pathlib.Path(os.environ.get(f"{_PREFIX}_BUILD_CACHE_DIR", tempfile.gettempdir()))
    / "gt4py_cache"
)


#: Whether generated code projects should be kept around between runs.
#: - SESSION: generated code projects get destroyed when the interpreter shuts down
#: - PERSISTENT: generated code projects are written to BUILD_CACHE_DIR and persist between runs
BUILD_CACHE_LIFETIME: BuildCacheLifetime = BuildCacheLifetime[
    os.environ.get(f"{_PREFIX}_BUILD_CACHE_LIFETIME", "persistent" if DEBUG else "session").upper()
]

#: Build type to be used when CMake is used to compile generated code.
#: Might have no effect when CMake is not used as part of the toolchain.
CMAKE_BUILD_TYPE: CMakeBuildType = CMakeBuildType[
    os.environ.get(f"{_PREFIX}_CMAKE_BUILD_TYPE", "debug" if DEBUG else "release").upper()
]
