# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

import gt4py.next.type_system.type_specifications as ts
from gt4py.eve import codegen
from gt4py.next.otf import code_specs


def format_source(settings: code_specs.SourceCodeSpec, source: str) -> str:
    assert settings.formatter_key is not None, "No formatter key specified in language settings."
    return codegen.format_source(
        settings.formatter_key, source, **(settings.formatter_options or {})
    )


@dataclasses.dataclass(frozen=True)
class Parameter:
    name: str
    type_: ts.TypeSpec


@dataclasses.dataclass(frozen=True)
class Function:
    name: str
    parameters: tuple[Parameter, ...]
    returns: bool = False


@dataclasses.dataclass(frozen=True)
class LibraryDependency:
    name: str
    version: str
