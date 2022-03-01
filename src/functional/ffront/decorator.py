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
# TODO(tehrengruber): This file contains to many different components. Split
#  into components for each dialect.
import abc
import collections
import dataclasses
import functools
import inspect
import types
import typing
from typing import Any, Optional, Protocol

from eve.utils import UIDs
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
from functional.iterator.runtime import Offset

DEFAULT_BACKEND = "roundtrip"

@typing.runtime_checkable
class GTCallable(Protocol):
    """
    Typing Protocol (abstract base class) defining the interface for subroutines.

    Any class implementing the methods defined in this protocol can be called
    from ``ffront`` programs or operators.
    """

    def __gt_closure_refs__(self) -> Optional[ClosureRefs]:
        return None

    @abc.abstractmethod
    def __gt_type__(self) -> ct.FunctionType:
        """
        Return symbol type, i.e. signature and return type.

        The type is used internally to populate the closure vars of the
        various dialects root nodes (i.e. FOAST Field Operator, PAST Program)
        """
        ...

    @abc.abstractmethod
    def __gt_itir__(self) -> itir.FunctionDefinition:
        """
        Return iterator IR function definition representing the callable.

        Used internally by the Program decorator to populate the function
        definitions of the iterator IR.
        """
        ...

    # TODO(tehrengruber): For embedded execution a `__call__` method and for
    #  "truely" embedded execution arguably also a `from_function` method is
    #  required. Since field operators currently have a `__gt_type__` with a
    #  Field return value, but it's `__call__` method being void (result via
    #  out arg) there is no good / consistent definition on what signature a
    #  protocol implementer is expected to provide. Skipping for now.


# TODO(tehrengruber): Decide if and how programs can call other programs. As a
#  result Program could become a GTCallable.
@dataclasses.dataclass(frozen=True)
class Program:
    """
    Construct a program object from a PAST node.

    A call to the resulting object executes the program as expressed
    by the PAST node.

    Attributes:
        past_node: The node representing the program.
        closure_refs: Mapping from names referenced in the program to the
            actual values.
        externals: Dictionary of externals.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
    """

    past_node: past.Program
    closure_refs: ClosureRefs
    externals: dict[str, Any]
    backend: Optional[str]
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
        return cls(
            past_node=past_node,
            closure_refs=closure_refs,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
        )

    def _lowered_funcs_from_closureref(self, closure_refs: ClosureRefs) -> list[itir.FunctionDefinition]:
        lowered_funcs = []

        vars_ = collections.ChainMap(closure_refs.globals,
                                     closure_refs.nonlocals)
        for name, val in vars_.items():
            # With respect to the frontend offsets are singleton types, i.e.
            #  they do not store any runtime information, but only type
            #  information. As such we do not need their value.
            if isinstance(val, Offset):
                continue
            if not isinstance(val, GTCallable):
                raise NotImplementedError(
                    "Only function closure vars are allowed currently.")
            lowered_funcs.append(val.__gt_itir__())
            # if the closure ref has closure refs by itself, also add them
            if val.__gt_closure_refs__():
                lowered_funcs.extend(self._lowered_funcs_from_closureref(val.__gt_closure_refs__()))
        return lowered_funcs

    @functools.cached_property
    def itir(self) -> itir.Program:
        if self.externals:
            raise NotImplementedError("Externals are not supported yet.")

        fencil_itir_node = ProgramLowering.apply(self.past_node)

        func_names = []
        for closure_var in self.past_node.closure:
            if isinstance(closure_var.type, ct.FunctionType):
                func_names.append(closure_var.id)
            else:
                raise NotImplementedError(
                    "Only function closure vars are allowed currently.")

        vars_ = collections.ChainMap(self.closure_refs.globals,
                                     self.closure_refs.nonlocals)
        if undefined := (set(vars_) - set(func_names)):
            raise RuntimeError(
                f"Reference to undefined symbol(s) `{', '.join(undefined)}`.")
        if not_callable := [name for name in func_names if
                            not isinstance(vars_[name], GTCallable)]:
            raise RuntimeError(
                f"The following function(s) are not valid GTCallables `{', '.join(not_callable)}`."
            )

        lowered_funcs = self._lowered_funcs_from_closureref(self.closure_refs)

        return itir.Program(
            function_definitions=lowered_funcs, fencil_definitions=[fencil_itir_node], setqs=[]
        )

    def _validate_args(self, *args, **kwargs) -> None:
        # TODO(tehrengruber): better error messages
        if len(args) != len(self.past_node.params):
            raise GTTypeError(
                f"Function takes {len(self.past_node.params)} arguments, but {len(args)} were given."
            )
        if kwargs:
            raise NotImplementedError("Keyword arguments are not supported yet.")

    def __call__(self, *args, offset_provider, **kwargs) -> None:
        self._validate_args(*args, **kwargs)

        # extract size of all field arguments
        size_args = []
        for param_idx, param in enumerate(self.past_node.params):
            if not isinstance(param.type, ct.FieldType):
                continue
            for dim_idx in range(0, len(param.type.dims)):
                size_args.append(args[param_idx].shape[dim_idx])

        backend = self.backend if self.backend else DEFAULT_BACKEND

        execute_program(
            self.itir, *args, *size_args, **kwargs, offset_provider=offset_provider, backend=backend
        )


def program(
    definition: types.FunctionType, externals: Optional[dict] = None, backend: Optional[str] = None
) -> Program:
    """
    Generate an implementation of a program from a Python function object.

    Examples:
        >>> @program  # noqa: F821 # doctest: +SKIP
        ... def program(in_field: Field[..., float64], out_field: Field[..., float64]): # noqa: F821
        ...     field_op(in_field, out=out_field)
        >>> program(in_field, out=out_field) # noqa: F821 # doctest: +SKIP
    """
    return Program.from_function(definition, externals, backend)


@dataclasses.dataclass(frozen=True)
class FieldOperator(GTCallable):
    """
    Construct a field operator object from a PAST node.

    A call to the resulting object executes the field operator as expressed
    by the FOAST node and with the signature as if it would appear inside
    a program.

    Attributes:
        foast_node: The node representing the field operator.
        closure_refs: Mapping from names referenced in the program to the
            actual values.
        externals: Dictionary of externals.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
    """

    foast_node: foast.FieldOperator
    closure_refs: ClosureRefs
    externals: dict[str, Any]
    backend: Optional[str]  # note: backend is only used if directly called
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[str] = None,
    ):
        closure_refs = ClosureRefs.from_function(definition)
        foast_node = FieldOperatorParser.apply_to_function(definition)
        return cls(
            foast_node=foast_node,
            closure_refs=closure_refs,
            externals=externals or {},
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.FunctionType:
        type_ = symbol_makers.make_symbol_type_from_value(self.definition)
        assert isinstance(type_, ct.FunctionType)
        return type_

    def __gt_itir__(self) -> itir.FunctionDefinition:
        return FieldOperatorLowering.apply(self.foast_node)

    def __gt_closure_refs__(self) -> ClosureRefs:
        return self.closure_refs

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
            past.Symbol(
                id=UIDs.sequential_id(prefix="__sym"),
                type=arg_type,
                namespace=ct.Namespace.LOCAL,
                location=loc,
            )
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
        closure_refs = dataclasses.replace(
            self.closure_refs, globals={**self.closure_refs.globals, name: self}
        )

        return Program(
            past_node=past_node,
            closure_refs=closure_refs,
            externals=self.externals,
            backend=self.backend,
        )

    def __call__(self, *args, out, offset_provider, **kwargs) -> None:
        return self.as_program()(*args, out, offset_provider=offset_provider, **kwargs)


def field_operator(
    definition: types.FunctionType, externals: Optional[dict] = None, backend: Optional[str] = None
) -> FieldOperator:
    """
    Generate an implementation of the field operator from a Python function object.

    Examples:
        >>> @field_operator  # doctest: +SKIP
        ... def field_op(in_field: Field[..., float64]) -> Field[..., float64]: # noqa: F821
        ...     ...
        >>> field_op(in_field, out=out_field)  # noqa: F821 # doctest: +SKIP
    """
    return FieldOperator.from_function(definition, externals, backend)


@dataclasses.dataclass(frozen=True)
class RawItIRStencil(GTCallable):
    itir_node: itir.FunctionDefinition
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(cls, definition: types.FunctionType):
        sig = inspect.signature(definition)
        if any(
            param.kind
            not in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]
            for param in sig.parameters.values()
        ):
            raise GTTypeError("Only positional arguments allowed.")
        param_decls = [itir.Sym(id=param.name) for param in sig.parameters.values()]
        param_refs = [itir.SymRef(id=param.name) for param in sig.parameters.values()]
        body = definition(*param_refs)
        if not isinstance(body, itir.Expr):
            raise GTTypeError(f"Expected an ITIR expression, but got `{body}`.")
        itir_node = itir.FunctionDefinition(id=definition.__name__, params=param_decls, expr=body)
        return RawItIRStencil(itir_node=itir_node, definition=definition)

    def __gt_itir__(self):
        return self.itir_node

    def __gt_type__(self):
        type_ = symbol_makers.make_symbol_type_from_value(self.definition)
        assert isinstance(type_, ct.FunctionType)
        return type_


def raw_itir_stencil(definition):
    return RawItIRStencil.from_function(definition)
