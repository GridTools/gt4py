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

The internal dependencies between modules are the following (each line depends
on some of the previous ones):

  - extended_typing (no dependencies)
  - exceptions, pattern_matching, type_definitions
  - datamodels, utils
  - trees
  - concepts
  - visitors
  - traits
  - codegen

"""

from __future__ import annotations  # isort:skip

from .version import __version__, __versioninfo__  # isort:skip

from .concepts import (
    AnnexManager,
    AnySourceLocation,
    FrozenNode,
    GenericDataModel,
    Node,
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
from .type_definitions import (
    NOTHING,
    ConstrainedStr,
    Enum,
    IntEnum,
    NothingType,
    SourceLocation,
    StrEnum,
    SymbolName,
    SymbolRef,
)
from .visitors import NodeTranslator, NodeVisitor


__all__ = [
    # version
    "__version__",
    "__versioninfo__",
    # datamodels
    "Coerced",
    "DataModel",
    "concretize",
    "datamodel",
    "field",
    #
    "Bool",
    "Enum",
    "Float",
    "Int",
    "IntEnum",
    "FieldKind",
    "FrozenModel",
    "FrozenNode",
    "GenericNode",
    "Model",
    "NegativeFloat",
    "NegativeInt",
    "NOTHING",
    "Node",
    "NodeMutator",
    "NodeTranslator",
    "NodeVisitor",
    "PositiveFloat",
    "PositiveInt",
    "SourceLocation",
    "Str",
    "StrEnum",
    "SymbolName",
    "SymbolRef",
    "SymbolTableTrait",
    "VType",
    "field",
    "iter_tree",
    "in_field",
    "out_field",
]
