# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

from __future__ import annotations

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
from .traits import (
    PreserveLocationVisitor,
    SymbolTableTrait,
    ValidatedSymbolTableTrait,
    VisitorWithSymbolTableTrait,
)
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


__all__ = [  # noqa: RUF022 `__all__` is not sorted
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
    "PreserveLocationVisitor",
    # trees
    "bfs_walk_items",
    "bfs_walk_values",
    "post_walk_items",
    "post_walk_values",
    "pre_walk_items",
    "pre_walk_values",
    "walk_items",
    "walk_values",
    # type_definitions
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
