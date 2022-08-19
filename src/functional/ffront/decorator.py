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

import collections
import dataclasses
import functools
import types
import typing
import warnings
from collections.abc import Callable, Iterable
from typing import Generic, SupportsFloat, SupportsInt, TypeAlias, TypeVar

import numpy as np
from devtools import debug

from eve.extended_typing import Any, Optional
from eve.utils import UIDGenerator
from functional.common import DimensionKind, GridType, GTTypeError
from functional.fencil_processors.processor_interface import (
    FencilExecutor,
    FencilFormatter,
    ensure_processor_kind,
)
from functional.fencil_processors.runners import roundtrip
from functional.ffront import (
    common_types as ct,
    field_operator_ast as foast,
    program_ast as past,
    symbol_makers,
    type_info,
)
from functional.ffront.fbuiltins import BUILTINS, Dimension, FieldOffset
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.gtcallable import GTCallable
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction, ProgramTypeError
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.source_utils import CapturedVars
from functional.iterator import ir as itir
from functional.iterator.embedded import constant_field


Scalar: TypeAlias = SupportsInt | SupportsFloat | np.int32 | np.int64 | np.float32 | np.float64

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


def _deduce_grid_type(
    requested_grid_type: Optional[GridType],
    offsets_and_dimensions: Iterable[FieldOffset | Dimension],
) -> GridType:
    """
    Derive grid type from actually occurring dimensions and check against optional user request.

    Unstructured grid type is consistent with any kind of offset, cartesian
    is easier to optimize for but only allowed in the absence of unstructured
    dimensions and offsets.
    """

    def is_cartesian_offset(o: FieldOffset):
        return len(o.target) == 1 and o.source == o.target[0]

    deduced_grid_type = GridType.CARTESIAN
    for o in offsets_and_dimensions:
        if isinstance(o, FieldOffset) and not is_cartesian_offset(o):
            deduced_grid_type = GridType.UNSTRUCTURED
            break
        if isinstance(o, Dimension) and o.kind == DimensionKind.LOCAL:
            deduced_grid_type = GridType.UNSTRUCTURED
            break

    if requested_grid_type == GridType.CARTESIAN and deduced_grid_type == GridType.UNSTRUCTURED:
        raise GTTypeError(
            "grid_type == GridType.CARTESIAN was requested, but unstructured `FieldOffset` or local `Dimension` was found."
        )

    return deduced_grid_type if requested_grid_type is None else requested_grid_type


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
        captured_vars = CapturedVars.from_function(definition)
        past_node = ProgramParser.apply_to_function(definition)
        return cls(
            past_node=past_node,
            captured_vars=captured_vars,
            externals={} if externals is None else externals,
            backend=backend,
            definition=definition,
            grid_type=grid_type,
        )

    def __post_init__(self):
        # validate contents of captured vars
        for name, value in self._filter_capture_vars_by_type(GTCallable).items():
            if value.__gt_itir__().id != name:
                raise RuntimeError(
                    "Name of the closure reference and the function it holds do not match."
                )

        # validate Symbols of captured vars in PAST
        referenced_var_names: set[str] = set()
        for captured_var in self.past_node.captured_vars:
            if isinstance(captured_var.type, (ct.CallableType, ct.OffsetType, ct.DimensionType)):
                referenced_var_names.add(captured_var.id)
            else:
                raise NotImplementedError("Only function closure vars are allowed currently.")
        defined_var_names = set(self.all_capture_vars.globals) | set(
            self.all_capture_vars.nonlocals
        )
        if undefined := referenced_var_names - defined_var_names:
            raise RuntimeError(f"Reference to undefined symbol(s) `{', '.join(undefined)}`.")

    def with_backend(self, backend: FencilExecutor) -> "Program":
        return Program(
            past_node=self.past_node,
            captured_vars=self.captured_vars,
            externals=self.externals,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    @functools.cached_property
    def all_capture_vars(self) -> CapturedVars:
        return _collect_capture_vars(self.captured_vars)

    @functools.cached_property
    def itir(self) -> itir.FencilDefinition:
        if self.externals:
            raise NotImplementedError("Externals are not supported yet.")

        grid_type = _deduce_grid_type(
            self.grid_type, self._filter_capture_vars_by_type(FieldOffset, Dimension).values()
        )

        gt_callables = self._filter_capture_vars_by_type(GTCallable).values()
        lowered_funcs = [gt_callable.__gt_itir__() for gt_callable in gt_callables]
        return ProgramLowering.apply(
            self.past_node, function_definitions=lowered_funcs, grid_type=grid_type
        )

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs) -> None:
        rewritten_args, size_args, kwargs = self._process_args(args, kwargs)

        if not self.backend:
            warnings.warn(
                UserWarning(
                    f"Field View Program '{self.itir.id}': Using default ({DEFAULT_BACKEND}) backend."
                )
            )
        backend = self.backend or DEFAULT_BACKEND

        ensure_processor_kind(backend, FencilExecutor)
        if "debug" in kwargs:
            debug(self.itir)

        backend(
            self.itir,
            *rewritten_args,
            *size_args,
            **kwargs,
            offset_provider=offset_provider,
            column_axis=self._column_axis,
        )

    def format_itir(
        self, *args, formatter: FencilFormatter, offset_provider: dict[str, Dimension], **kwargs
    ) -> str:
        ensure_processor_kind(formatter, FencilFormatter)
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

    def _validate_args(self, *args, **kwargs) -> None:
        if kwargs:
            raise NotImplementedError("Keyword arguments are not supported yet.")

        arg_types = [symbol_makers.make_symbol_type_from_value(arg) for arg in args]
        kwarg_types = {k: symbol_makers.make_symbol_type_from_value(v) for k, v in kwargs.items()}

        try:
            type_info.accepts_args(
                self.past_node.type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
        except GTTypeError as err:
            raise ProgramTypeError.from_past_node(
                self.past_node, msg=f"Invalid argument types in call to `{self.past_node.id}`!"
            ) from err

    def _process_args(self, args: tuple, kwargs: dict) -> tuple[tuple, tuple, dict[str, Any]]:
        self._validate_args(*args, **kwargs)

        # extract size of all field arguments
        size_args: list[Optional[tuple[int, ...]]] = []
        rewritten_args = list(args)
        for param_idx, param in enumerate(self.past_node.params):
            if isinstance(param.type, ct.ScalarType):
                dtype = type_info.extract_dtype(param.type)
                rewritten_args[param_idx] = constant_field(
                    args[param_idx],
                    dtype=BUILTINS[dtype.kind.name.lower()],
                )
            if not isinstance(param.type, ct.FieldType):
                continue
            has_shape = hasattr(args[param_idx], "shape")
            for dim_idx in range(0, len(param.type.dims)):
                if has_shape:
                    size_args.append(args[param_idx].shape[dim_idx])
                else:
                    size_args.append(None)

        return tuple(rewritten_args), tuple(size_args), kwargs

    def _filter_capture_vars_by_type(self, *types: type) -> dict[str, Any]:
        flat_capture_vars = self.all_capture_vars.globals | self.all_capture_vars.nonlocals
        return {k: v for k, v in flat_capture_vars.items() if isinstance(v, types)}

    @functools.cached_property
    def _column_axis(self):
        # construct mapping from column axis to scan operators defined on
        #  that dimension. only one column axis is allowed, but we can use
        #  this mapping to provide good error messages.
        scanops_per_axis: dict[Dimension, str] = {}
        for name, gt_callable in self._filter_capture_vars_by_type(GTCallable).items():
            if isinstance((type_ := gt_callable.__gt_type__()), ct.ScanOperatorType):
                scanops_per_axis.setdefault(type_.axis, []).append(name)

        if len(scanops_per_axis.values()) == 0:
            return None

        if len(scanops_per_axis.values()) != 1:
            scanops_per_axis_strs = [
                f"- {dim.value}: {', '.join(scanops)}" for dim, scanops in scanops_per_axis.items()
            ]

            raise GTTypeError(
                "Only `ScanOperator`s defined on the same axis "
                + "can be used in a `Program`, but found:\n"
                + "\n".join(scanops_per_axis_strs)
            )

        return iter(scanops_per_axis.keys()).__next__()


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
    backend: Optional[FencilExecutor]  # note: backend is only used if directly called
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        externals: Optional[dict] = None,
        backend: Optional[FencilExecutor] = None,
        *,
        operator_node_cls: type[OperatorNodeT] = foast.FieldOperator,
        operator_attributes: Optional[dict[str, Any]] = None,
    ) -> FieldOperator[OperatorNodeT]:
        operator_attributes = operator_attributes or {}

        captured_vars = CapturedVars.from_function(definition)
        foast_definition_node = FieldOperatorParser.apply_to_function(definition)
        loc = foast_definition_node.location
        operator_attribute_nodes = {
            key: foast.Constant(
                value=value, type=symbol_makers.make_symbol_type_from_value(value), location=loc
            )
            for key, value in operator_attributes.items()
        }
        untyped_foast_node = operator_node_cls(
            id=foast_definition_node.id,
            definition=foast_definition_node,
            location=loc,
            **operator_attribute_nodes,
        )
        foast_node = FieldOperatorTypeDeduction.apply(untyped_foast_node)
        return cls(
            foast_node=foast_node,
            captured_vars=captured_vars,
            externals=externals or {},
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.CallableType:
        type_ = self.foast_node.type
        assert isinstance(type_, ct.CallableType)
        return type_

    def with_backend(self, backend: FencilExecutor) -> FieldOperator:
        return FieldOperator(
            foast_node=self.foast_node,
            captured_vars=self.captured_vars,
            externals=self.externals,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    def __gt_itir__(self) -> itir.FunctionDefinition:
        if hasattr(self, "__cached_itir"):
            return getattr(self, "__cached_itir")  # noqa: B009

        itir_node: itir.FunctionDefinition = FieldOperatorLowering.apply(self.foast_node)

        object.__setattr__(self, "__cached_itir", itir_node)

        return itir_node

    def __gt_captured_vars__(self) -> CapturedVars:
        return self.captured_vars

    def as_program(
        self, arg_types: list[ct.SymbolType], kwarg_types: dict[str, ct.SymbolType]
    ) -> Program:
        # TODO(tehrengruber): implement mechanism to deduce default values
        #  of arg and kwarg types
        # TODO(tehrengruber): check foast operator has no out argument that clashes
        #  with the out argument of the program we generate here.

        name = self.foast_node.id
        loc = self.foast_node.location
        param_sym_uids = UIDGenerator()  # use a new UID generator to allow caching

        type_ = self.__gt_type__()
        params_decl: list[past.Symbol] = [
            past.DataSymbol(
                id=param_sym_uids.sequential_id(prefix="__sym"),
                type=arg_type,
                namespace=ct.Namespace.LOCAL,
                location=loc,
            )
            for arg_type in arg_types
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym: past.Symbol = past.DataSymbol(
            id="out",
            type=type_info.return_type(type_, with_args=arg_types, with_kwargs=kwarg_types),
            namespace=ct.Namespace.LOCAL,
            location=loc,
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
            type=ct.DeferredSymbolType(constraint=ct.ProgramType),
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
        # deduce argument types
        arg_types = []
        for arg in args:
            arg_types.append(symbol_makers.make_symbol_type_from_value(arg))
        kwarg_types = {}
        for name, arg in kwargs.items():
            kwarg_types[name] = symbol_makers.make_symbol_type_from_value(arg)

        return self.as_program(arg_types, kwarg_types)(
            *args, out, offset_provider=offset_provider, **kwargs
        )


@typing.overload
def field_operator(
    definition: types.FunctionType, *, externals: Optional[dict], backend: Optional[FencilExecutor]
) -> FieldOperator[foast.FieldOperator]:
    ...


@typing.overload
def field_operator(
    *, externals: Optional[dict], backend: Optional[FencilExecutor]
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
def scan_operator(
    definition: types.FunctionType,
    *,
    axis: Dimension,
    forward: bool,
    init: Scalar,
    externals: Optional[dict],
    backend: Optional[str],
) -> FieldOperator[foast.ScanOperator]:
    ...


@typing.overload
def scan_operator(
    *,
    axis: Dimension,
    forward: bool,
    init: Scalar,
    externals: Optional[dict],
    backend: Optional[str],
) -> Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]:
    ...


def scan_operator(
    definition: Optional[types.FunctionType] = None,
    *,
    axis: Dimension,
    forward: bool = True,
    init: Scalar = 0.0,
    externals=None,
    backend=None,
) -> FieldOperator[foast.ScanOperator] | Callable[
    [types.FunctionType], FieldOperator[foast.ScanOperator]
]:
    """
    Generate an implementation of the scan operator from a Python function object.

    Arguments:
        definition: Function from scalars to a scalar.

    Keyword Arguments:
        axis: A :ref:`Dimension` to reduce over.
        forward: Boolean specifying the direction.
        init: Initial value for the carry argument of the scan pass.

    Examples:
        >>> import numpy as np
        >>> from functional.iterator.embedded import np_as_located_field
        >>> import functional.iterator.embedded
        >>> functional.iterator.embedded._column_range = 1
        >>> KDim = Dimension("K", kind=DimensionKind.VERTICAL)
        >>> inp = np_as_located_field(KDim)(np.ones((10,)))
        >>> out = np_as_located_field(KDim)(np.zeros((10,)))
        >>> @scan_operator(axis=KDim, forward=True, init=0.)
        ... def scan_operator(carry: float, val: float) -> float:
        ...     return carry+val
        >>> scan_operator(inp, out=out, offset_provider={})  # doctest: +SKIP
        >>> out.array()  # doctest: +SKIP
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    """
    # TODO(tehrengruber): enable doctests again. For unknown / obscure reasons
    #  the above doctest fails when executed using `pytest --doctest-modules`.

    def scan_operator_inner(definition: types.FunctionType) -> FieldOperator:
        return FieldOperator.from_function(
            definition,
            externals,
            backend,
            operator_node_cls=foast.ScanOperator,
            operator_attributes={"axis": axis, "forward": forward, "init": init},
        )

    return scan_operator_inner if definition is None else scan_operator_inner(definition)
