# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceCodeConfig:
    """
    Basic settings for any source programming language.

    Formatting will happen through ``eve.codegen.format_source``.
    For available formatting options, check the options of the
    specific formatter used depending on ``.formatter_key``.
    """

    name: str
    file_extension: str
    formatter_key: str | None = None
    formatter_options: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceCodeAndHeaderConfig(SourceCodeConfig):
    """Add a header file extension setting on top of the basic set."""

    header_extension: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class PythonCodeConfig(SourceCodeConfig):
    """Settings for Python language."""

    name: str = "python"
    file_extension: str = "py"
    formatter_key: str = "python"


@dataclasses.dataclass(frozen=True, kw_only=True)
class SDFGCodeConfig(SourceCodeConfig):
    """Settings for SDFGs."""

    name: str = "SDFG"
    file_extension: str = "sdfg"


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPPLikeCodeConfig(SourceCodeAndHeaderConfig):
    """Settings for C++-like language."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPPCodeConfig(CPPLikeCodeConfig):
    """Settings for C++ language."""

    name: str = "CXX"
    file_extension: str = "cpp"
    header_extension: str = "hpp"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CUDACodeConfig(CPPLikeCodeConfig):
    """Settings for CUDA language."""

    name: str = "CUDA"
    file_extension: str = "cu"
    header_extension: str = "cuh"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HIPCodeConfig(CPPLikeCodeConfig):
    """Settings for HIP language."""

    name: str = "HIP"
    file_extension: str = "hip"
    header_extension: str = "h"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )
