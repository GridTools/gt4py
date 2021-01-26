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

"""Eve: a stencil toolchain in pure Python."""


# flake8: noqa  # disable flake8 because of non-used imports warnings
from .version import __version__, __versioninfo__  # isort:skip

# Internal dependencies between modules (each line depends on some of the previous ones):
#
#   - typingx  (no dependencies)
#   - exceptions, type_definitions
#   - utils
#   - concepts <-> iterators  (circular dependency only inside methods, it should be safe)
#   - traits, visitors
#   - codegen
#

from .concepts import (
    FieldKind,
    FrozenModel,
    FrozenNode,
    GenericNode,
    Model,
    Node,
    VType,
    field,
    in_field,
    out_field,
)
from .iterators import iter_tree
from .traits import SymbolTableTrait
from .type_definitions import (
    NOTHING,
    Bool,
    Enum,
    Float,
    Int,
    IntEnum,
    NegativeFloat,
    NegativeInt,
    PositiveFloat,
    PositiveInt,
    SourceLocation,
    Str,
    StrEnum,
    SymbolName,
    SymbolRef,
)
from .visitors import NodeMutator, NodeTranslator, NodeVisitor
