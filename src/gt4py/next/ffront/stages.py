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

from gt4py.next import common
from gt4py.next.ffront import field_operator_ast as foast, program_ast as past
from gt4py.next.type_system import type_specifications as ts


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


class ContentHashableMixin:
    """
    Allows deriving dataclasses to modify what goes into the content hash per-field.

    Warning: Using this will modify how the class gets pickled. If unpickling is desired,
    extra care has to be taken. The hasher must not remove crucial data and any modifications
    have to be undone while loading in the __setstate__ method (which needs to be implemented).

    In fact, when unpickling is required, it is probably best to implement both
    __setstate__ and __getstate__ by hand for the entire class rather than per-field.
    """

    def __getstate__(self) -> dict[str, Any]:
        if not dataclasses.is_dataclass(self):
            raise TypeError(f"'{self.__class__}' is not a dataclass.")
        state = self.__dict__.copy()
        for field in dataclasses.fields(self):
            if "content_hasher" in field.metadata:
                field.metadata["content_hasher"](state, getattr(self, field.name), field.name)
        return state


def function_type_hasher(state: dict[str, Any], value: types.FunctionType, name: str) -> None:
    state[f"{name}__name"] = value.__name__
    state[f"{name}__source"] = inspect.getsource(value)
    del state[name]


def simple_dict_hasher(state: dict[str, Any], value: dict[str, Any], name: str) -> None:
    for k, v in value.items():
        state[f"{name}__{k}"] = v


def closure_vars_hasher(state: dict[str, Any], value: dict[str, Any], name: str) -> None:
    hashable_closure_vars = {}
    for k, v in value.items():
        # replace the decorator with the earliest canonical dsl representation available
        if hasattr(v, "definition_stage"):
            hashable_closure_vars[k] = v.definition_stage
        elif hasattr(v, "foast_stage"):
            hashable_closure_vars[k] = v.foast_stage
        elif hasattr(v, "past_stage"):
            hashable_closure_vars[k] = v.past_stage
        # put the backend into the hash because it may influence the toolchain
        # TODO(ricoh): This is not perfect, since backend names are allowed to clash (low priority).
        if hasattr(v, "backend"):
            hashable_closure_vars[f"{k}_backend"] = v.backend.__name__ if v.backend else "None"
    state[name] = hashable_closure_vars


@dataclasses.dataclass(frozen=True)
class FieldOperatorDefinition(ContentHashableMixin, Generic[OperatorNodeT]):
    definition: types.FunctionType = dataclasses.field(
        metadata={"content_hasher": function_type_hasher}
    )
    grid_type: Optional[common.GridType] = None
    node_class: type[OperatorNodeT] = dataclasses.field(default=foast.FieldOperator)  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
    attributes: dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"content_hasher": simple_dict_hasher}
    )


@dataclasses.dataclass(frozen=True)
class FoastOperatorDefinition(ContentHashableMixin, Generic[OperatorNodeT]):
    foast_node: OperatorNodeT
    closure_vars: dict[str, Any] = dataclasses.field(
        metadata={"content_hasher": closure_vars_hasher}
    )
    grid_type: Optional[common.GridType] = None
    attributes: dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"content_hasher": simple_dict_hasher}
    )


@dataclasses.dataclass(frozen=True)
class FoastWithTypes(ContentHashableMixin, Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    arg_types: tuple[ts.TypeSpec, ...]
    kwarg_types: dict[str, ts.TypeSpec]
    closure_vars: dict[str, Any] = dataclasses.field(
        metadata={"content_hasher": closure_vars_hasher}
    )


@dataclasses.dataclass(frozen=True)
class FoastClosure(ContentHashableMixin, Generic[OperatorNodeT]):
    foast_op_def: FoastOperatorDefinition[OperatorNodeT]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    closure_vars: dict[str, Any] = dataclasses.field(
        metadata={"content_hasher": closure_vars_hasher}
    )


@dataclasses.dataclass(frozen=True)
class ProgramDefinition(ContentHashableMixin):
    definition: types.FunctionType = dataclasses.field(
        metadata={"content_hasher": function_type_hasher}
    )
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastProgramDefinition(ContentHashableMixin):
    past_node: past.Program
    closure_vars: dict[str, Any] = dataclasses.field(
        metadata={"content_hasher": closure_vars_hasher}
    )
    grid_type: Optional[common.GridType] = None


@dataclasses.dataclass(frozen=True)
class PastClosure(ContentHashableMixin):
    closure_vars: dict[str, Any] = dataclasses.field(
        metadata={"content_hasher": closure_vars_hasher}
    )
    past_node: past.Program
    grid_type: Optional[common.GridType]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
