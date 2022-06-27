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


from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import types
import typing
import warnings
from typing import Callable, ClassVar, TypeVar, Generic, Type
from numbers import Number

from eve.extended_typing import Any, Optional, Protocol
from eve.utils import UIDs
from functional.common import GTTypeError
from functional.ffront import (
    common_types as ct,
    field_operator_ast as foast,
    program_ast as past,
    symbol_makers,
    type_info
)
from functional.ffront.fbuiltins import BUILTINS, Dimension
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.source_utils import CapturedVars
from functional.iterator import ir as itir
from functional.iterator.backend_executor import execute_fencil
from functional.iterator.embedded import constant_field

DEFAULT_BACKEND = "roundtrip"


def _collect_capture_vars(captured_vars: CapturedVars) -> CapturedVars:
    new_captured_vars = captured_vars
    flat_captured_vars = collections.ChainMap(captured_vars.globals, captured_vars.nonlocals)

    for value in flat_captured_vars.values():
        if isinstance(value, GTCallable):
            # if the closure ref has closure refs by itself, also add them
            if vars_of_val := value.__gt_captured_vars__():
                vars_of_val = _collect_capture_vars(vars_of_val)

                flat_vars_of_val = collections.ChainMap(vars_of_val.globals, vars_of_val.nonlocals)
                collisions: list[str] = []
                for potential_collision in set(flat_captured_vars) & set(flat_vars_of_val):
                    if (
                        flat_captured_vars[potential_collision]
                        != flat_vars_of_val[potential_collision]
                    ):
                        collisions.append(potential_collision)
                if collisions:
                    raise NotImplementedError(
                        f"Using closure vars with same name, but different value "
                        f"across functions is not implemented yet. \n"
                        f"Collisions: {'`,  `'.join(collisions)}"
                    )

                new_captured_vars = dataclasses.replace(
                    new_captured_vars,
                    globals={**new_captured_vars.globals, **vars_of_val.globals},
                    nonlocals={**new_captured_vars.nonlocals, **vars_of_val.nonlocals},
                )
    return new_captured_vars


@typing.runtime_checkable
class GTCallable(Protocol):
    """
    Typing Protocol (abstract base class) defining the interface for subroutines.

    Any class implementing the methods defined in this protocol can be called
    from ``ffront`` programs or operators.
    """

    def __gt_captured_vars__(self) -> Optional[CapturedVars]:
        """
        Return all external variables referenced inside the callable.

        Note that in addition to the callable itself all captured variables
        are also lowered such that they can be used in the lowered callable.
        """
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
        captured_vars: Mapping from names referenced in the program to the
            actual values.
        externals: Dictionary of externals.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
    """

    past_node: past.Program
    captured_vars: CapturedVars
    externals: dict[str, Any]
    backend: Optional[str]
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[str] = None,
    ) -> "Program":
        captured_vars = _collect_capture_vars(CapturedVars.from_function(definition))
        past_node = ProgramParser.apply_to_function(definition)
        return cls(
            past_node=past_node,
            captured_vars=captured_vars,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
        )

    def with_backend(self, backend: str) -> "Program":
        return Program(
            past_node=self.past_node,
            captured_vars=self.captured_vars,
            externals=self.externals,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    def _lowered_funcs_from_captured_vars(
        self, captured_vars: CapturedVars
    ) -> list[itir.FunctionDefinition]:
        lowered_funcs = []

        all_captured_vars = collections.ChainMap(captured_vars.globals, captured_vars.nonlocals)

        for name, value in all_captured_vars.items():
            if not isinstance(value, GTCallable):
                continue
            itir_node = value.__gt_itir__()
            if itir_node.id != name:
                raise RuntimeError(
                    "Name of the closure reference and the function it holds do not match."
                )
            lowered_funcs.append(itir_node)
        return lowered_funcs

    @functools.cached_property
    def itir(self) -> itir.FencilDefinition:
        if self.externals:
            raise NotImplementedError("Externals are not supported yet.")

        capture_vars = _collect_capture_vars(self.captured_vars)

        referenced_var_names: set[str] = set()
        for captured_var in self.past_node.captured_vars:
            if isinstance(captured_var.type, (ct.CallableType, ct.OffsetType, ct.DimensionType)):
                referenced_var_names.add(captured_var.id)
            else:
                raise NotImplementedError("Only function closure vars are allowed currently.")
        defined_var_names = set(capture_vars.globals) | set(capture_vars.nonlocals)
        if undefined := referenced_var_names - defined_var_names:
            raise RuntimeError(f"Reference to undefined symbol(s) `{', '.join(undefined)}`.")

        lowered_funcs = self._lowered_funcs_from_captured_vars(capture_vars)

        return ProgramLowering.apply(self.past_node, function_definitions=lowered_funcs)

    def _validate_args(self, *args, **kwargs) -> None:
        # TODO(tehrengruber): better error messages, check argument types
        if len(args) != len(self.past_node.params):
            raise GTTypeError(
                f"Function takes {len(self.past_node.params)} arguments, but {len(args)} were given."
            )
        if kwargs:
            raise NotImplementedError("Keyword arguments are not supported yet.")

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs) -> None:
        self._validate_args(*args, **kwargs)

        # extract size of all field arguments
        size_args: list[Optional[tuple[int, ...]]] = []
        rewritten_args = list(args)
        for param_idx, param in enumerate(self.past_node.params):
            if not isinstance(param.type, ct.FieldType):
                continue
            if args[param_idx].array is None:
                size_args.append(None)
                continue
            for dim_idx in range(0, len(param.type.dims)):
                size_args.append(args[param_idx].shape[dim_idx])

        if not self.backend:
            warnings.warn(
                UserWarning(
                    f"Field View Program '{self.itir.id}': Using default ({DEFAULT_BACKEND}) backend."
                )
            )
        backend = self.backend if self.backend else DEFAULT_BACKEND

        execute_fencil(
            self.itir,
            *rewritten_args,
            *size_args,
            **kwargs,
            offset_provider=offset_provider,
            backend=backend,
            column_axis=Dimension(value="K"),
        )


@typing.overload
def program(definition: types.FunctionType) -> Program:
    ...


@typing.overload
def program(
    *, externals: Optional[dict], backend: Optional[str]
) -> Callable[[types.FunctionType], Program]:
    ...


def program(
    definition=None,
    *,
    externals=None,
    backend=None,
) -> Program | Callable[[types.FunctionType], Program]:
    """
    Generate an implementation of a program from a Python function object.

    Examples:
        >>> @program  # noqa: F821 # doctest: +SKIP
        ... def program(in_field: Field[..., float64], out_field: Field[..., float64]): # noqa: F821
        ...     field_op(in_field, out=out_field)
        >>> program(in_field, out=out_field) # noqa: F821 # doctest: +SKIP

        >>> # the backend can optionally be passed if already decided
        >>> # not passing it will result in embedded execution by default
        >>> # the above is equivalent to
        >>> @program(backend="roundtrip")  # noqa: F821 # doctest: +SKIP
        ... def program(in_field: Field[..., float64], out_field: Field[..., float64]): # noqa: F821
        ...     field_op(in_field, out=out_field)
        >>> program(in_field, out=out_field) # noqa: F821 # doctest: +SKIP
    """

    def program_inner(definition: types.FunctionType) -> Program:
        return Program.from_function(definition, externals, backend)

    return program_inner if definition is None else program_inner(definition)


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperator(GTCallable, Generic[OperatorNodeT]):
    """
    Construct a field operator object from a PAST node.

    A call to the resulting object executes the field operator as expressed
    by the FOAST node and with the signature as if it would appear inside
    a program.

    Attributes:
        foast_node: The node representing the field operator.
        captured_vars: Mapping from names referenced in the program to the
            actual values.
        externals: Dictionary of externals.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
    """
    foast_node: OperatorNodeT
    captured_vars: CapturedVars
    externals: dict[str, Any]
    backend: Optional[str]  # note: backend is only used if directly called
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[str] = None,
        *,
        operator_node_cls: Type[OperatorNodeT] = foast.FieldOperator,
        operator_attributes: Optional[dict[str, Any]] = None,
    ) -> FieldOperator[OperatorNodeT]:
        captured_vars = CapturedVars.from_function(definition)
        foast_definition_node = FieldOperatorParser.apply_to_function(definition)
        loc = foast_definition_node.location
        operator_attribute_nodes = {
            key: foast.Constant(value=value,
                                type=symbol_makers.make_symbol_type_from_value(
                                    value), location=loc)
            for key, value in operator_attributes.items()
        } if operator_attributes else {}
        untyped_foast_node = operator_node_cls(
            id=foast_definition_node.id,
            definition=foast_definition_node,
            location=loc,
            **(operator_attribute_nodes)
        )
        foast_node = FieldOperatorTypeDeduction.apply(untyped_foast_node)
        return cls(
            foast_node=foast_node,
            captured_vars=captured_vars,
            externals=externals or {},
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.FunctionType:
        type_ = self.foast_node.type
        assert type_info.is_concrete(type_)
        return type_

    def with_backend(self, backend: str) -> FieldOperator:
        return FieldOperator(
            foast_node=self.foast_node,
            captured_vars=self.captured_vars,
            externals=self.externals,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    def __gt_itir__(self) -> itir.FunctionDefinition:
        return typing.cast(itir.FunctionDefinition, FieldOperatorLowering.apply(self.foast_node))

    def __gt_captured_vars__(self) -> CapturedVars:
        return self.captured_vars

    def as_program(self, arg_types: list[ct.SymbolType], kwarg_types: dict[str, ct.SymbolType]) -> Program:
        # todo(tehrengruber): add comment why arg types are needed
        # todo(tehrengruber): use kwargs for check
        #if any(param.id == "out" for param in self.foast_node.params):
        #    raise Exception(
        #        "Direct call to Field operator whose signature contains an argument `out` is not permitted."
        #    )

        name = self.foast_node.id
        loc = self.foast_node.location

        type_ = self.__gt_type__()
        params_decl: list[past.Symbol] = [
            past.DataSymbol(
                id=UIDs.sequential_id(prefix="__sym"),
                type=arg_type,
                namespace=ct.Namespace.LOCAL,
                location=loc,
            )
            for arg_type in arg_types
            # for arg_type in type_.args # TODO(check it agrees with arg_types)
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym: past.Symbol = past.DataSymbol(
            id="out", type=type_info.return_type(type_, with_args=arg_types, with_kwargs=kwarg_types), namespace=ct.Namespace.LOCAL, location=loc
        )
        out_ref = past.Name(id="out", location=loc)

        # inject stencil as a closure var into program. Since CapturedVars is
        #  immutable we have to resort to this rather ugly way of doing a copy.
        captured_vars = dataclasses.replace(
            self.captured_vars, globals={**self.captured_vars.globals, name: self}
        )
        all_captured_vars = collections.ChainMap(captured_vars.globals, captured_vars.nonlocals)

        captured_symbols: list[past.Symbol] = []
        for name, val in all_captured_vars.items():  # type: ignore
            captured_symbols.append(
                past.Symbol(
                    id=name,
                    type=symbol_makers.make_symbol_type_from_value(val),
                    namespace=ct.Namespace.CLOSURE,
                    location=loc,
                )
            )

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
            captured_vars=captured_symbols,
            location=loc,
        )
        past_node = ProgramTypeDeduction.apply(untyped_past_node)

        return Program(
            past_node=past_node,
            captured_vars=captured_vars,
            externals=self.externals,
            backend=self.backend,
        )

    def __call__(self, *args, out, offset_provider: dict[str, Dimension],
                 **kwargs) -> None:
        # TODO(tehrengruber): check all offset providers are given
        # deduce argument types
        arg_types = []
        for arg in args:
            arg_types.append(symbol_makers.make_symbol_type_from_value(arg))
        kwarg_types = {}
        for name, arg in kwargs.items():
            kwarg_types[name] = symbol_makers.make_symbol_type_from_value(arg)

        return self.as_program(arg_types, kwarg_types)(*args, out,
                                                       offset_provider=offset_provider,
                                                       **kwargs)


@typing.overload
def field_operator(definition: types.FunctionType) -> FieldOperator[foast.FieldOperator]:
    ...


@typing.overload
def field_operator(
    *, externals: Optional[dict], backend: Optional[str]
) -> Callable[[types.FunctionType], FieldOperator[foast.FieldOperator]]:
    ...


def field_operator(
    definition=None,
    *,
    externals=None,
    backend=None,
):
    """
    Generate an implementation of the field operator from a Python function object.

    Examples:
        >>> @field_operator  # doctest: +SKIP
        ... def field_op(in_field: Field[..., float64]) -> Field[..., float64]: # noqa: F821
        ...     ...
        >>> field_op(in_field, out=out_field)  # noqa: F821 # doctest: +SKIP

        >>> # the backend can optionally be passed if already decided
        >>> # not passing it will result in embedded execution by default
        >>> @field_operator(backend="roundtrip")  # doctest: +SKIP
        ... def field_op(in_field: Field[..., float64]) -> Field[..., float64]: # noqa: F821
        ...     ...
    """

    def field_operator_inner(definition: types.FunctionType) -> FieldOperator[foast.FieldOperator]:
        return FieldOperator.from_function(definition, externals, backend)

    return field_operator_inner if definition is None else field_operator_inner(definition)


@typing.overload
def scan_operator(definition: types.FunctionType) -> FieldOperator[foast.ScanOperator]:
    ...


@typing.overload
def scan_operator(
    *, externals: Optional[dict], backend: Optional[str]
) -> Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]:
    ...


def scan_operator(
    definition=None,
    *,
    axis: Dimension,
    forward: bool = True,
    init: Number = 0.,
    externals=None,
    backend=None,
):
    """
    Generate an implementation of the scan operator from a Python function object.+

    Arguments:
        definition: Function from scalars to a scalar.

    Keyword Arguments:
        axis: A :ref:`Dimension` to reduce over.
        forward: Boolean specifying the direction.
        init: Initial value for the carry argument of the scan pass.

    Examples:
        >>>
        >>> @scan_operator(axis=KDim, forward=True, init=0.)
        >>> def scan_operator(carry: float, val: float) -> float:
        >>>     return carry+val.
        >>> scan_operator(field, out=out)
    """

    def scan_operator_inner(definition: types.FunctionType) -> FieldOperator:
        return FieldOperator.from_function(
            definition,
            externals,
            backend,
            operator_node_cls=foast.ScanOperator,
            operator_attributes={
                "axis": axis,
                "forward": forward,
                "init": init
            }
        )

    return scan_operator_inner if definition is None else scan_operator_inner(definition)