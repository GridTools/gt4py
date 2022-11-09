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
from typing import Generator, Generic, TypeVar

from devtools import debug

from eve.extended_typing import Any, Optional
from eve.utils import UIDGenerator
from functional.common import DimensionKind, GridType, GTTypeError, Scalar
from functional.ffront import (
    common_types as ct,
    field_operator_ast as foast,
    program_ast as past,
    symbol_makers,
    type_info,
)
from functional.ffront.fbuiltins import Dimension, FieldOffset
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from functional.ffront.foast_to_itir import FieldOperatorLowering
from functional.ffront.func_to_foast import FieldOperatorParser
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.gtcallable import GTCallable
from functional.ffront.past_passes.closure_var_type_deduction import (
    ClosureVarTypeDeduction as ProgramClosureVarTypeDeduction,
)
from functional.ffront.past_passes.type_deduction import ProgramTypeDeduction, ProgramTypeError
from functional.ffront.past_to_itir import ProgramLowering
from functional.ffront.source_utils import SourceDefinition, get_closure_vars_from_function
from functional.iterator import ir as itir
from functional.program_processors import processor_interface as ppi
from functional.program_processors.runners import roundtrip


DEFAULT_BACKEND: Callable = roundtrip.executor


def _get_closure_vars_recursively(closure_vars: dict[str, Any]) -> dict[str, Any]:
    all_closure_vars = collections.ChainMap(closure_vars)

    for closure_var in closure_vars.values():
        if isinstance(closure_var, GTCallable):
            # if the closure ref has closure refs by itself, also add them
            if child_closure_vars := closure_var.__gt_closure_vars__():
                all_child_closure_vars = _get_closure_vars_recursively(child_closure_vars)

                collisions: list[str] = []
                for potential_collision in set(closure_vars) & set(all_child_closure_vars):
                    if (
                        closure_vars[potential_collision]
                        != all_child_closure_vars[potential_collision]
                    ):
                        collisions.append(potential_collision)
                if collisions:
                    raise NotImplementedError(
                        f"Using closure vars with same name but different value "
                        f"across functions is not implemented yet. \n"
                        f"Collisions: {'`,  `'.join(collisions)}"
                    )

                all_closure_vars = collections.ChainMap(all_closure_vars, all_child_closure_vars)
    return dict(all_closure_vars)


def _filter_closure_vars_by_type(closure_vars: dict[str, Any], *types: type) -> dict[str, Any]:
    return {name: value for name, value in closure_vars.items() if isinstance(value, types)}


def _canonicalize_args(
    node_params: list[foast.DataSymbol] | list[past.DataSymbol],
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> tuple[tuple, dict]:
    new_args = []
    new_kwargs = {**kwargs}

    for param_i, param in enumerate(node_params):
        if param.id in new_kwargs:
            if param_i < len(args):
                raise ProgramTypeError(f"got multiple values for argument {param.id}.")
            new_args.append(kwargs[param.id])
            new_kwargs.pop(param.id)
        elif param_i < len(args):
            new_args.append(args[param_i])
        else:
            # case when param in function definition but not in function call
            # e.g. function expects 3 parameters, but only 2 were given.
            # Error covered later in `accept_args`.
            pass

    args = tuple(new_args)
    return args, new_kwargs


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


def _field_constituents_shape_and_dims(
    arg, arg_type: ct.FieldType | ct.ScalarType | ct.TupleType
) -> Generator[tuple[tuple[int, ...], list[Dimension]]]:
    if isinstance(arg_type, ct.TupleType):
        for el, el_type in zip(arg, arg_type.types):
            yield from _field_constituents_shape_and_dims(el, el_type)
    elif isinstance(arg_type, ct.FieldType):
        dims = type_info.extract_dims(arg_type)
        if hasattr(arg, "shape"):
            assert len(arg.shape) == len(dims)
            yield (arg.shape, dims)
        else:
            yield (None, dims)
    else:
        raise ValueError("Expected `FieldType` or `TupleType` thereof.")


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
        closure_vars: Mapping of externally defined symbols to their respective values.
            For example, referenced global and nonlocal variables.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
    """

    past_node: past.Program
    closure_vars: dict[str, Any]
    backend: Optional[ppi.ProgramExecutor]
    definition: Optional[types.FunctionType] = None
    grid_type: Optional[GridType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[ppi.ProgramExecutor] = None,
        grid_type: Optional[GridType] = None,
    ) -> Program:
        source_def = SourceDefinition.from_function(definition)
        closure_vars = get_closure_vars_from_function(definition)
        annotations = typing.get_type_hints(definition)
        past_node = ProgramParser.apply(source_def, closure_vars, annotations)
        return cls(
            past_node=past_node,
            closure_vars=closure_vars,
            backend=backend,
            definition=definition,
            grid_type=grid_type,
        )

    def __post_init__(self):
        function_closure_vars = _filter_closure_vars_by_type(self.closure_vars, GTCallable)
        misnamed_functions = [
            f"{name} vs. {func.id}"
            for name, func in function_closure_vars.items()
            if name != func.__gt_itir__().id
        ]
        if misnamed_functions:
            raise RuntimeError(
                f"The following symbols resolve to a function with a mismatching name: {','.join(misnamed_functions)}"
            )

        undefined_symbols = [
            symbol.id
            for symbol in self.past_node.closure_vars
            if symbol.id not in self.closure_vars
        ]
        if undefined_symbols:
            raise RuntimeError(
                f"The following closure variables are undefined: {', '.join(undefined_symbols)}"
            )

    def with_backend(self, backend: ppi.ProgramExecutor) -> "Program":
        return Program(
            past_node=self.past_node,
            closure_vars=self.closure_vars,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    @functools.cached_property
    def _all_closure_vars(self) -> dict[str, Any]:
        return _get_closure_vars_recursively(self.closure_vars)

    @functools.cached_property
    def itir(self) -> itir.FencilDefinition:
        offsets_and_dimensions = _filter_closure_vars_by_type(
            self._all_closure_vars, FieldOffset, Dimension
        )
        grid_type = _deduce_grid_type(self.grid_type, offsets_and_dimensions.values())

        gt_callables = _filter_closure_vars_by_type(self._all_closure_vars, GTCallable).values()
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

        ppi.ensure_processor_kind(backend, ppi.ProgramExecutor)
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
        self,
        *args,
        formatter: ppi.ProgramFormatter,
        offset_provider: dict[str, Dimension],
        **kwargs,
    ) -> str:
        ppi.ensure_processor_kind(formatter, ppi.ProgramFormatter)
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
            raise NotImplementedError("Keyword-only arguments are not supported yet.")

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
        args, kwargs = _canonicalize_args(self.past_node.params, args, kwargs)

        self._validate_args(*args, **kwargs)

        implicit_domain = any(
            isinstance(stmt, past.Call) and "domain" not in stmt.kwargs
            for stmt in self.past_node.body
        )

        # extract size of all field arguments
        size_args: list[Optional[tuple[int, ...]]] = []
        rewritten_args = list(args)
        for param_idx, param in enumerate(self.past_node.params):
            if implicit_domain and isinstance(param.type, (ct.FieldType, ct.TupleType)):
                shapes_and_dims = [*_field_constituents_shape_and_dims(args[param_idx], param.type)]
                shape, dims = shapes_and_dims[0]
                if not all(
                    el_shape == shape and el_dims == dims for (el_shape, el_dims) in shapes_and_dims
                ):
                    raise ValueError(
                        "Constituents of composite arguments (e.g. the elements of a"
                        " tuple) need to have the same shape and dimensions."
                    )
                size_args.extend(shape if shape else [None] * len(dims))
        return tuple(rewritten_args), tuple(size_args), kwargs

    @functools.cached_property
    def _column_axis(self):
        # construct mapping from column axis to scan operators defined on
        #  that dimension. only one column axis is allowed, but we can use
        #  this mapping to provide good error messages.
        scanops_per_axis: dict[Dimension, str] = {}
        for name, gt_callable in _filter_closure_vars_by_type(
            self._all_closure_vars, GTCallable
        ).items():
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
def program(*, backend: Optional[ppi.ProgramExecutor]) -> Callable[[types.FunctionType], Program]:
    ...


def program(
    definition=None,
    *,
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
        return Program.from_function(definition, backend, grid_type)

    return program_inner if definition is None else program_inner(definition)


OperatorNodeT = TypeVar("OperatorNodeT", bound=foast.LocatedNode)


@dataclasses.dataclass(frozen=True)
class FieldOperator(GTCallable, Generic[OperatorNodeT]):
    """
    Construct a field operator object from a FOAST node.

    A call to the resulting object executes the field operator as expressed
    by the FOAST node and with the signature as if it would appear inside
    a program.

    Attributes:
        foast_node: The node representing the field operator.
        closure_vars: Mapping of names referenced in the field operator (i.e.
            globals, nonlocals) to their values.
        backend: The backend used for executing the field operator. Only used
            if the field operator is called directly, otherwise the backend
            specified for the program takes precedence.
        definition: The original Python function object the field operator
            was created from.
    """

    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    backend: Optional[ppi.ProgramExecutor]
    definition: Optional[types.FunctionType] = None

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[ppi.ProgramExecutor] = None,
        *,
        operator_node_cls: type[OperatorNodeT] = foast.FieldOperator,
        operator_attributes: Optional[dict[str, Any]] = None,
    ) -> FieldOperator[OperatorNodeT]:
        operator_attributes = operator_attributes or {}

        source_def = SourceDefinition.from_function(definition)
        closure_vars = get_closure_vars_from_function(definition)
        annotations = typing.get_type_hints(definition)
        foast_definition_node = FieldOperatorParser.apply(source_def, closure_vars, annotations)
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
            closure_vars=closure_vars,
            backend=backend,
            definition=definition,
        )

    def __gt_type__(self) -> ct.CallableType:
        type_ = self.foast_node.type
        assert isinstance(type_, ct.CallableType)
        return type_

    def with_backend(self, backend: ppi.ProgramExecutor) -> FieldOperator:
        return FieldOperator(
            foast_node=self.foast_node,
            closure_vars=self.closure_vars,
            backend=backend,
            definition=self.definition,  # type: ignore[arg-type]  # mypy wrongly deduces definition as method here
        )

    def __gt_itir__(self) -> itir.FunctionDefinition:
        if hasattr(self, "__cached_itir"):
            return getattr(self, "__cached_itir")  # noqa: B009

        itir_node: itir.FunctionDefinition = FieldOperatorLowering.apply(self.foast_node)

        object.__setattr__(self, "__cached_itir", itir_node)

        return itir_node

    def __gt_closure_vars__(self) -> dict[str, Any]:
        return self.closure_vars

    def as_program(
        self, arg_types: list[ct.SymbolType], kwarg_types: dict[str, ct.SymbolType]
    ) -> Program:
        # TODO(tehrengruber): implement mechanism to deduce default values
        #  of arg and kwarg types
        # TODO(tehrengruber): check foast operator has no out argument that clashes
        #  with the out argument of the program we generate here.

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

        if self.foast_node.id in self.closure_vars:
            raise RuntimeError("A closure variable has the same name as the field operator itself.")
        closure_vars = {self.foast_node.id: self}
        closure_symbols = [
            past.Symbol(
                id=self.foast_node.id,
                type=ct.DeferredSymbolType(constraint=None),
                namespace=ct.Namespace.CLOSURE,
                location=loc,
            ),
        ]

        untyped_past_node = past.Program(
            id=f"__field_operator_{self.foast_node.id}",
            type=ct.DeferredSymbolType(constraint=ct.ProgramType),
            params=params_decl + [out_sym],
            body=[
                past.Call(
                    func=past.Name(id=self.foast_node.id, location=loc),
                    args=params_ref,
                    kwargs={"out": out_ref},
                    location=loc,
                )
            ],
            closure_vars=closure_symbols,
            location=loc,
        )
        untyped_past_node = ProgramClosureVarTypeDeduction.apply(untyped_past_node, closure_vars)
        past_node = ProgramTypeDeduction.apply(untyped_past_node)

        return Program(
            past_node=past_node,
            closure_vars=closure_vars,
            backend=self.backend,
        )

    def __call__(
        self,
        *args,
        out,
        offset_provider: dict[str, Dimension],
        **kwargs,
    ) -> None:
        args, kwargs = _canonicalize_args(self.foast_node.definition.params, args, kwargs)
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
    definition: types.FunctionType, *, backend: Optional[ppi.ProgramExecutor]
) -> FieldOperator[foast.FieldOperator]:
    ...


@typing.overload
def field_operator(
    *, backend: Optional[ppi.ProgramExecutor]
) -> Callable[[types.FunctionType], FieldOperator[foast.FieldOperator]]:
    ...


def field_operator(
    definition=None,
    *,
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
        return FieldOperator.from_function(definition, backend)

    return field_operator_inner if definition is None else field_operator_inner(definition)


@typing.overload
def scan_operator(
    definition: types.FunctionType,
    *,
    axis: Dimension,
    forward: bool,
    init: Scalar,
    backend: Optional[str],
) -> FieldOperator[foast.ScanOperator]:
    ...


@typing.overload
def scan_operator(
    *,
    axis: Dimension,
    forward: bool,
    init: Scalar,
    backend: Optional[str],
) -> Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]:
    ...


def scan_operator(
    definition: Optional[types.FunctionType] = None,
    *,
    axis: Dimension,
    forward: bool = True,
    init: Scalar = 0.0,
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
            backend,
            operator_node_cls=foast.ScanOperator,
            operator_attributes={"axis": axis, "forward": forward, "init": init},
        )

    return scan_operator_inner if definition is None else scan_operator_inner(definition)
