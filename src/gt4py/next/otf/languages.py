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


class Cpp(LanguageTag):
    settings_class = LanguageWithHeaderFilesSettings
    ...
