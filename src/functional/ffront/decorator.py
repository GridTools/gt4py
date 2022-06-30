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
from typing import Callable, Iterable, Protocol

from devtools import debug

from eve.extended_typing import Any, Optional
from eve.utils import UIDs
from functional.common import GridType, GTTypeError
from functional.fencil_processors import roundtrip
from functional.ffront import (
    common_types as ct,
    field_operator_ast as foast,
    program_ast as past,
    symbol_makers,
)
from functional.ffront.fbuiltins import BUILTINS, Dimension, FieldOffset
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.source_utils import CapturedVars
from functional.iterator import ir as itir
from functional.iterator.embedded import constant_field
from functional.iterator.processor_interface import (
    FencilExecutor,
    FencilFormatter,
    ensure_executor,
    ensure_formatter,
)


DEFAULT_BACKEND: Callable = roundtrip.executor


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
    #  "truly" embedded execution arguably also a `from_function` method is
    #  required. Since field operators currently have a `__gt_type__` with a
    #  Field return value, but it's `__call__` method being void (result via
    #  out arg) there is no good / consistent definition on what signature a
    #  protocol implementer is expected to provide. Skipping for now.


# TODO(tehrengruber): Decide if and how programs can call other programs. As a
#  result Program could become a GTCallable.
# TODO(ricoh): factor out the generated ITIR together with arguments rewriting
# so that using fencil processors on `some_program.itir` becomes trivial without
# prior knowledge of the fencil signature rewriting done by `Program`.
# After that, drop the `.format_itir()` method, since it won't be needed.
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
    backend: Optional[FencilExecutor]
    definition: Optional[types.FunctionType] = None
    grid_type: Optional[GridType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[FencilExecutor] = None,
        grid_type: Optional[GridType] = None,
    ) -> "Program":
        captured_vars = _collect_capture_vars(CapturedVars.from_function(definition))
        past_node = ProgramParser.apply_to_function(definition)
        return cls(
            past_node=past_node,
            captured_vars=captured_vars,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
            grid_type=grid_type,
        )

    def with_backend(self, backend: FencilExecutor) -> "Program":
        return Program(
            past_node=self.past_node,
            captured_vars=self.captured_vars,
            externals=self.externals,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    @staticmethod
    def _deduce_grid_type(
        requested_grid_type: Optional[GridType],
        offsets_and_dimensions: set[FieldOffset | Dimension],
    ):
        """
        Derive grid type from actually occurring dimensions and check against optional user request.

        Unstructured grid type is consistent with any kind of offset, cartesian is easier to optimize for but only
        allowed in the absence of unstructured dimensions and offsets.
        """

        def is_cartesian_offset(o: FieldOffset):
            return len(o.target) == 1 and o.source == o.target[0]

        deduced_grid_type = GridType.CARTESIAN
        for o in offsets_and_dimensions:
            if isinstance(o, FieldOffset) and not is_cartesian_offset(o):
                deduced_grid_type = GridType.UNSTRUCTURED
                break
            if isinstance(o, Dimension) and o.local:
                deduced_grid_type = GridType.UNSTRUCTURED
                break

        if requested_grid_type == GridType.CARTESIAN and deduced_grid_type == GridType.UNSTRUCTURED:
            raise GTTypeError(
                "grid_type == GridType.CARTESIAN was requested, but unstructured `FieldOffset` or local `Dimension` was found."
            )

        return deduced_grid_type if requested_grid_type is None else requested_grid_type

    def _gt_callables_from_captured_vars(self, captured_vars: CapturedVars) -> list[GTCallable]:
        all_captured_vars = collections.ChainMap(captured_vars.globals, captured_vars.nonlocals)

        gt_callables = []
        for name, value in all_captured_vars.items():
            if isinstance(value, GTCallable):
                if value.__gt_itir__().id != name:
                    raise RuntimeError(
                        "Name of the closure reference and the function it holds do not match."
                    )
                gt_callables.append(value)
        return gt_callables

    def _offsets_and_dimensions_from_gt_callables(
        self, gt_callables: Iterable[GTCallable]
    ) -> set[FieldOffset | Dimension]:
        offsets_and_dimensions: set[FieldOffset | Dimension] = set()
        for gt_callable in gt_callables:
            if (captured := gt_callable.__gt_captured_vars__()) is not None:
                for c in (captured.globals | captured.nonlocals).values():
                    if isinstance(c, FieldOffset):
                        offsets_and_dimensions.add(c)
                    if isinstance(c, Dimension):
                        offsets_and_dimensions.add(c)
        return offsets_and_dimensions

    def _lowered_funcs_from_gt_callables(
        self, gt_callables: Iterable[GTCallable]
    ) -> list[itir.FunctionDefinition]:
        return [gt_callable.__gt_itir__() for gt_callable in gt_callables]

    @functools.cached_property
    def itir(self) -> itir.FencilDefinition:
        if self.externals:
            raise NotImplementedError("Externals are not supported yet.")

        capture_vars = _collect_capture_vars(self.captured_vars)

        referenced_var_names: set[str] = set()
        for captured_var in self.past_node.captured_vars:
            if isinstance(captured_var.type, (ct.FunctionType, ct.OffsetType, ct.DimensionType)):
                referenced_var_names.add(captured_var.id)
            else:
                raise NotImplementedError("Only function closure vars are allowed currently.")
        defined_var_names = set(capture_vars.globals) | set(capture_vars.nonlocals)
        if undefined := referenced_var_names - defined_var_names:
            raise RuntimeError(f"Reference to undefined symbol(s) `{', '.join(undefined)}`.")

        referenced_gt_callables = self._gt_callables_from_captured_vars(capture_vars)
        grid_type = self._deduce_grid_type(
            self.grid_type, self._offsets_and_dimensions_from_gt_callables(referenced_gt_callables)
        )

        lowered_funcs = self._lowered_funcs_from_gt_callables(referenced_gt_callables)
        return ProgramLowering.apply(
            self.past_node, function_definitions=lowered_funcs, grid_type=grid_type
        )

    def _validate_args(self, *args, **kwargs) -> None:
        # TODO(tehrengruber): better error messages, check argument types
        if len(args) != len(self.past_node.params):
            raise GTTypeError(
                f"Function takes {len(self.past_node.params)} arguments, but {len(args)} were given."
            )
        if kwargs:
            raise NotImplementedError("Keyword arguments are not supported yet.")

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs) -> None:
        rewritten_args, size_args, kwargs = self._process_args(args, kwargs)

        if not self.backend:
            warnings.warn(
                UserWarning(
                    f"Field View Program '{self.itir.id}': Using default ({DEFAULT_BACKEND}) backend."
                )
            )
        backend = self.backend if self.backend else DEFAULT_BACKEND

        ensure_executor(backend)
        if "debug" in kwargs:
            debug(self.itir)

        backend(
            self.itir,
            *rewritten_args,
            *size_args,
            **kwargs,
            offset_provider=offset_provider,
        )

    def format_itir(
        self, *args, formatter: FencilFormatter, offset_provider: dict[str, Dimension], **kwargs
    ) -> str:
        ensure_formatter(formatter)
        rewritten_args, size_args, kwargs = self._process_args(args, kwargs)
        if "debug" in kwargs:
            debug(self.itir)
        return formatter(
            self.itir,
            *rewritten_args,
            *size_args,
            **kwargs,
            offset_provider=offset_provider,
        )

    def _process_args(self, args: tuple, kwargs: dict) -> tuple[tuple, tuple, dict[str, Any]]:
        self._validate_args(*args, **kwargs)

        # extract size of all field arguments
        size_args: list[Optional[tuple[int, ...]]] = []
        rewritten_args = list(args)
        for param_idx, param in enumerate(self.past_node.params):
            if isinstance(param.type, ct.ScalarType):
                rewritten_args[param_idx] = constant_field(
                    args[param_idx],
                    dtype=BUILTINS[param.type.kind.name.lower()],
                )
            if not isinstance(param.type, ct.FieldType):
                continue
            if not hasattr(args[param_idx], "__array__"):
                size_args.append(None)
                continue
            for dim_idx in range(0, len(param.type.dims)):
                size_args.append(args[param_idx].shape[dim_idx])

        return tuple(rewritten_args), tuple(size_args), kwargs


@typing.overload
def program(definition: types.FunctionType) -> Program:
    ...


@typing.overload
def program(
    *, externals: Optional[dict], backend: Optional[FencilExecutor]
) -> Callable[[types.FunctionType], Program]:
    ...


def program(
    definition=None,
    *,
    externals=None,
    backend=None,
    grid_type=None,
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
        return Program.from_function(definition, externals, backend, grid_type)

    return program_inner if definition is None else program_inner(definition)


@dataclasses.dataclass(frozen=True)
class FieldOperator(GTCallable):
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

    foast_node: foast.FieldOperator
    captured_vars: CapturedVars
    externals: dict[str, Any]
    backend: Optional[FencilExecutor]  # note: backend is only used if directly called
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[FencilExecutor] = None,
    ) -> "FieldOperator":
        captured_vars = CapturedVars.from_function(definition)
        foast_node = FieldOperatorParser.apply_to_function(definition)
        return cls(
            foast_node=foast_node,
            captured_vars=captured_vars,
            externals=externals or {},
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.FunctionType:
        type_ = symbol_makers.make_symbol_type_from_value(self.definition)
        assert isinstance(type_, ct.FunctionType)
        return type_

    def with_backend(self, backend: FencilExecutor) -> "FieldOperator":
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

    def as_program(self) -> Program:
        if any(param.id == "out" for param in self.foast_node.params):
            raise Exception(
                "Direct call to Field operator whose signature contains an argument `out` is not permitted."
            )

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
            for arg_type in type_.args
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym: past.Symbol = past.DataSymbol(
            id="out", type=type_.returns, namespace=ct.Namespace.LOCAL, location=loc
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

    def __call__(self, *args, out, offset_provider: dict[str, Dimension], **kwargs) -> None:
        # TODO(tehrengruber): check all offset providers are given
        return self.as_program()(*args, out, offset_provider=offset_provider, **kwargs)


@typing.overload
def field_operator(definition: types.FunctionType) -> FieldOperator:
    ...


@typing.overload
def field_operator(
    *, externals: Optional[dict], backend: Optional[FencilExecutor]
) -> Callable[[types.FunctionType], FieldOperator]:
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

    def field_operator_inner(definition: types.FunctionType) -> FieldOperator:
        return FieldOperator.from_function(definition, externals, backend)

    return field_operator_inner if definition is None else field_operator_inner(definition)
