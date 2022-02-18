# GT4Py Project - GridTools Framework
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
import dataclasses
import types
from collections import ChainMap
from functools import cached_property
from typing import Any, Optional

from functional.common import GTTypeError
from functional.ffront import common_types as ct
from functional.ffront import field_operator_ast as foast
from functional.ffront import program_ast as past
from functional.ffront import symbol_makers
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.source_utils import ClosureRefs
from functional.iterator import ir as itir
from functional.iterator.backend_executor import execute_program


uid = 0


def gensym():
    global uid
    uid += 1
    return f"__sym_{uid}"


default_backend = "roundtrip"


@dataclasses.dataclass
class Program:
    past_node: past.Program
    closure_refs: ClosureRefs
    externals: dict[str, Any]
    backend: str
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        closure_refs = ClosureRefs.from_function(definition)
        past_node = ProgramParser.apply_to_function(definition)
        return Program(
            past_node=past_node,
            closure_refs=closure_refs,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
        )

    def __post_init__(self):
        if self.backend is None:
            self.backend = default_backend

    @cached_property
    def itir(self) -> itir.Program:
        if self.externals:
            raise NotImplementedError("Externals are not supported yet.")

        fencil_itir_node = ProgramLowering.apply(self.past_node)

        func_names = []
        for closure_var in self.past_node.closure:
            if isinstance(closure_var.type, ct.FunctionType):
                func_names.append(closure_var.id)
            else:
                raise NotImplementedError("Only function closure vars are allowed currently.")

        vars_ = ChainMap(self.closure_refs.globals, self.closure_refs.nonlocals)
        if not all(func in vars_ for func in func_names):
            undef_funcs = [func not in vars_ for func in func_names]
            raise RuntimeError(f"Reference to undefined symbol(s) `{', '.join(undef_funcs)}`")
        funcs_lowered = [vars_[func_names].__gt_itir__() for func_names in func_names]

        return itir.Program(
            function_definitions=funcs_lowered, fencil_definitions=[fencil_itir_node], setqs=[]
        )

    def _validate_args(self, *args, **kwargs):
        # validate signature
        if len(args) != len(self.past_node.params):
            raise GTTypeError(
                f"Function takes {len(self.past_node.params)} arguments, but {len(args)} were given."
            )
        if kwargs:
            raise NotImplementedError("Keyword arguments are not supported yet.")

    def __call__(self, *args, offset_provider, **kwargs):
        self._validate_args(*args, **kwargs)

        # extract size of all field arguments
        size_args = []
        for param_idx, param in enumerate(self.past_node.params):
            if not isinstance(param.type, ct.FieldType):
                continue
            for dim_idx in range(0, len(param.type.dims)):
                size_args.append(args[param_idx].shape[dim_idx])

        backend = self.backend if self.backend else default_backend

        execute_program(
            self.itir, *args, *size_args, **kwargs, offset_provider=offset_provider, backend=backend
        )


def program(
    definition: types.FunctionType, externals: Optional[dict] = None, backend: Optional[str] = None
) -> Program:
    return Program.from_function(definition, externals, backend)


@dataclasses.dataclass(frozen=True)
class FieldOperator:
    foast_node: foast.FieldOperator
    closure_refs: ClosureRefs
    externals: dict[str, Any]
    backend: str  # note: backend is only used if directly called
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls, definition: types.FunctionType, externals: Optional[dict] = None, backend: str = None
    ):
        closure_refs = ClosureRefs.from_function(definition)
        foast_node = FieldOperatorParser.apply_to_function(definition)
        return FieldOperator(
            foast_node=foast_node,
            closure_refs=closure_refs,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.FunctionType:
        type_ = symbol_makers.make_symbol_type_from_value(self.definition)
        assert isinstance(type_, ct.FunctionType)
        return type_

    def __gt_itir__(self):
        return FieldOperatorLowering.apply(self.foast_node)

    def as_program(self) -> Program:
        if any(param.id == "out" for param in self.foast_node.params):
            raise Exception(
                "Direct call to Field operator whose signature contains an argument `out` is not permitted."
            )

        name = self.foast_node.id
        loc = self.foast_node.location

        type_ = self.__gt_type__()
        stencil_sym = past.Symbol(id=name, type=type_, namespace=ct.Namespace.CLOSURE, location=loc)

        params_decl = [
            past.Symbol(id=gensym(), type=arg_type, namespace=ct.Namespace.LOCAL, location=loc)
            for arg_type in type_.args
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym = past.Symbol(
            id="out", type=type_.returns, namespace=ct.Namespace.LOCAL, location=loc
        )
        out_ref = past.Name(id="out", location=loc)

        untyped_past_node = past.Program(
            id=f"__field_operator_{name}",
            params=params_decl + [out_sym],
            body=[
                past.Call(
                    func=past.Name(id=name, location=loc),
                    args=params_ref,
                    kwargs={"out": out_ref},
                    location=loc,
                )
            ],
            closure=[stencil_sym],
            location=loc,
        )
        past_node = ProgramTypeDeduction.apply(untyped_past_node)

        # inject stencil as a closure var into program
        #  since ClosureRefs is immutable we have to resort to this rather ugly way of doing a copy
        closure_refs = ClosureRefs(
            **{
                **{
                    field.name: getattr(self.closure_refs, field.name)
                    for field in dataclasses.fields(self.closure_refs)
                },
                "globals": {**self.closure_refs.globals, name: self},
            }
        )

        return Program(
            past_node=past_node,
            closure_refs=closure_refs,
            externals=self.externals,
            backend=self.backend,
        )

    def __call__(self, *args, out, offset_provider, **kwargs):
        return self.as_program()(*args, out, offset_provider=offset_provider, **kwargs)


def field_operator(
    definition: types.FunctionType, externals: Optional[dict] = None, backend: Optional[str] = None
) -> FieldOperator:
    return FieldOperator.from_function(definition, externals, backend)
