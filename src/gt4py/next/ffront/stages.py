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
import inspect
import types
from typing import Any, Generic, Optional, TypeVar

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past
from gt4py.next.type_system import type_specifications as ts


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperatorDefinition(Generic[OperatorNodeT]):
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    node_class: type[OperatorNodeT] = dataclasses.field(default=foast.FieldOperator)  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        state = self.__dict__.copy()
        state["name"] = self.definition.__name__
        state["source"] = inspect.getsource(self.definition)
        state |= self.attributes
        del state["definition"]
        return state


def hash_field_operator_definition(fieldop_definition: FieldOperatorDefinition) -> str:
    return eve_utils.content_hash(fieldop_definition)


@dataclasses.dataclass(frozen=True)
class FoastOperatorDefinition(Generic[OperatorNodeT]):
    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        hashable_closure_vars = {}
        for k, v in self.closure_vars.items():
            if hasattr(v, "definition_stage"):
                hashable_closure_vars[k] = v.definition_stage
                hashable_closure_vars[f"{k}_backend"] = v.backend.__name__ if v.backend else "None"
        state = self.__dict__.copy()
        state["closure_vars"] = hashable_closure_vars
        return state


def hash_foast_operator_definition(foast_definition: FoastOperatorDefinition) -> str:
    return eve_utils.content_hash(
        foast_definition.foast_node,
        foast_definition.grid_type,
        tuple(foast_definition.attributes.items()),
    )


@dataclasses.dataclass(frozen=True)
class FoastWithTypes(Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    arg_types: tuple[ts.TypeSpec]
    kwarg_types: dict[str, ts.TypeSpec]
    closure_vars: dict[str, Any]

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        hashable_closure_vars = {}
        for k, v in self.closure_vars.items():
            if hasattr(v, "definition_stage"):
                hashable_closure_vars[k] = v.definition_stage
                hashable_closure_vars[f"{k}_backend"] = v.backend.__name__ if v.backend else "None"
        state = self.__dict__.copy()
        state["closure_vars"] = hashable_closure_vars
        return state


@dataclasses.dataclass(frozen=True)
class FoastClosure(Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    closure_vars: dict[str, Any]

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        hashable_closure_vars = {}
        for k, v in self.closure_vars.items():
            if hasattr(v, "definition_stage"):
                hashable_closure_vars[k] = v.definition_stage
                hashable_closure_vars[f"{k}_backend"] = v.backend.__name__ if v.backend else "None"
        state = self.__dict__.copy()
        state["closure_vars"] = hashable_closure_vars
        return state


def hash_foast_with_types(foast_with_types: FoastWithTypes) -> str:
    return eve_utils.content_hash(foast_with_types)
    # return eve_utils.content_hash(
    #     (
    #         foast_with_types.foast_op_def,
    #         foast_with_types.arg_types,
    #         tuple((name, arg) for name, arg in foast_with_types.kwarg_types.items()),
    #     )
    # )


@dataclasses.dataclass(frozen=True)
class ProgramDefinition:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        state = self.__dict__.copy()
        state["name"] = self.definition.__name__
        state["source"] = inspect.getsource(self.definition)
        del state["definition"]
        return state


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition:
    past_node: past.Program
    closure_vars: dict[str, Any]
    grid_type: Optional[common.GridType] = None

    def __getstate__(self) -> dict[str, Any]:
        """Make the stage pickleable (but not unpickleable) for use with content_hash."""
        hashable_closure_vars = {}
        for k, v in self.closure_vars.items():
            if hasattr(v, "definition_stage"):
                hashable_closure_vars[k] = v.definition_stage
                hashable_closure_vars[f"{k}_backend"] = v.backend.__name__ if v.backend else "None"
        state = self.__dict__.copy()
        state["closure_vars"] = hashable_closure_vars
        return state


def hash_past_program_definition(past_definition: PastProgramDefinition) -> str:
    return eve_utils.content_hash(past_definition)


@dataclasses.dataclass(frozen=True)
class PastClosure:
    closure_vars: dict[str, Any]
    past_node: past.Program
    grid_type: Optional[common.GridType]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
