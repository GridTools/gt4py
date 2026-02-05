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
class SourceLanguageSettings:
    """
    Basic settings for any language.

    Formatting will happen through ``eve.codegen.format_source``.
    For available formatting styles check, which formatter is used (depends on ``.formatter_key``)
    and then check which styles are available for that (if any).
    """

    name: str
    file_extension: str
    formatter_key: str | None = None
    formatter_options: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class SourceAndHeaderLanguageSettings(SourceLanguageSettings):
    """Add a header file extension setting on top of the basic set."""

    header_extension: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class PythonLanguageSettings(SourceLanguageSettings):
    """Settings for Python language."""

    name: str = "python"
    file_extension: str = "py"
    formatter_key: str = "python"


@dataclasses.dataclass(frozen=True, kw_only=True)
class SDFGLanguageSettings(SourceLanguageSettings):
    """Settings for SDFGs."""

    name: str = "SDFG"
    file_extension: str = "sdfg"


@dataclasses.dataclass(frozen=True, kw_only=True)
class CLikeLanguageSettings(SourceAndHeaderLanguageSettings):
    """Settings for C++-like language."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class CPPLanguageSettings(CLikeLanguageSettings):
    """Settings for C++ language."""

    name: str = "CXX"
    file_extension: str = "cpp"
    header_extension: str = "hpp"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CUDALanguageSettings(CLikeLanguageSettings):
    """Settings for CUDA language."""

    name: str = "CUDA"
    file_extension: str = "cu"
    header_extension: str = "cuh"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HIPLanguageSettings(CLikeLanguageSettings):
    """Settings for HIP language."""

    name: str = "HIP"
    file_extension: str = "hip"
    header_extension: str = "h"
    formatter_key: str = "cpp"
    formatter_options: Mapping[str, Any] = dataclasses.field(
        default_factory=functools.partial(dict, formatter_style="LLVM")
    )
