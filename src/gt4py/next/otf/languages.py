# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import ClassVar, Optional


class LanguageTag:
    """
    Represent a programming language.

    ``.settings_class`` should be set to the ``LanguageSettings`` subclass
    with the minimum amount of settings required for the language.
    """

    settings_class: ClassVar[type[LanguageSettings]]
    ...


@dataclasses.dataclass(frozen=True)
class LanguageSettings:
    """
    Basic settings for any language.

    Formatting will happen through ``eve.codegen.format_source``.
    For available formatting styles check, which formatter is used (depends on ``.formatter_key``)
    and then check which styles are available for that (if any).
    """

    formatter_key: str
    formatter_style: Optional[str]
    file_extension: str


@dataclasses.dataclass(frozen=True)
class LanguageWithHeaderFilesSettings(LanguageSettings):
    """Add a header file extension setting on top of the basic set."""

    header_extension: str


class Python(LanguageTag):
    settings_class = LanguageSettings
    ...


class SDFG(LanguageTag):
    settings_class = LanguageSettings
    ...


class NanobindSrcL(LanguageTag): ...


class Cpp(NanobindSrcL):
    settings_class = LanguageWithHeaderFilesSettings
    ...


class Cuda(NanobindSrcL):
    settings_class = LanguageWithHeaderFilesSettings


class Hip(NanobindSrcL):
    settings_class = LanguageWithHeaderFilesSettings
