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

"""Eve framework with general utils for development of DSL toolchains in Python.

The internal dependencies between modules are the following (each module depends
on some of the previous ones):

  0. extended_typing
  1. exceptions, pattern_matching, type_definitions
  2. utils
  3. type_validation
  4. datamodels
  5. trees
  6. concepts
  7. visitors
  8. traits
  9. codegen

"""

from __future__ import annotations  # isort:skip

from .version import __version__, __version_info__  # isort:skip


from .concepts import (
    AnnexManager,
    AnySourceLocation,
    FrozenNode,
    GenericNode,
    Node,
    RootNode,
    SourceLocation,
    SourceLocationGroup,
    SymbolName,
    SymbolRef,
    VType,
    register_annex_user,
)
from .datamodels import (
    Coerced,
    DataModel,
    FrozenModel,
    GenericDataModel,
    Unchecked,
    concretize,
    datamodel,
    field,
    frozenmodel,
)
from .traits import SymbolTableTrait, ValidatedSymbolTableTrait, VisitorWithSymbolTableTrait
from .trees import (
    bfs_walk_items,
    bfs_walk_values,
    post_walk_items,
    post_walk_values,
    pre_walk_items,
    pre_walk_values,
    walk_items,
    walk_values,
)
from .type_definitions import NOTHING, ConstrainedStr, Enum, IntEnum, NothingType, StrEnum
from .visitors import NodeTranslator, NodeVisitor


__all__ = [
    # version
    "__version__",
    "__version_info__",
    # concepts
    "AnnexManager",
    "AnySourceLocation",
    "FrozenNode",
    "RootNode",
    "GenericNode",
    "Node",
    "SourceLocation",
    "SourceLocationGroup",
    "SymbolName",
    "SymbolRef",
    "VType",
    "register_annex_user",
    "# datamodels" "Coerced",
    "DataModel",
    "FrozenModel",
    "GenericDataModel",
    "Unchecked",
    "concretize",
    "datamodel",
    "field",
    "frozenmodel",
    # datamodels
    "Coerced",
    "DataModel",
    "FrozenModel",
    "GenericDataModel",
    "Unchecked",
    "concretize",
    "datamodel",
    "field",
    "frozenmodel",
    # traits
    "SymbolTableTrait",
    "ValidatedSymbolTableTrait",
    "VisitorWithSymbolTableTrait",
    # trees
    "bfs_walk_items",
    "bfs_walk_values",
    "post_walk_items",
    "post_walk_values",
    "pre_walk_items",
    "pre_walk_values",
    "walk_items",
    "walk_values",
    "# type_definition",
    "NOTHING",
    "ConstrainedStr",
    "Enum",
    "IntEnum",
    "NothingType",
    "StrEnum",
    # visitors
    "NodeTranslator",
    "NodeVisitor",
]
