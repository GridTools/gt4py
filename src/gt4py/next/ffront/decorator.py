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

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Any, Optional
from gt4py.next import allocators as next_allocators, embedded as next_embedded, errors
from gt4py.next.common import Dimension, DimensionKind, GridType
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import (
    dialect_ast_enums,
    field_operator_ast as foast,
    program_ast as past,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.fbuiltins import FieldOffset
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.ffront.foast_to_itir import FieldOperatorLowering
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.ffront.func_to_past import ProgramParser
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.ffront.past_passes.closure_var_type_deduction import (
    ClosureVarTypeDeduction as ProgramClosureVarTypeDeduction,
)
from gt4py.next.ffront.past_passes.type_deduction import ProgramTypeDeduction
from gt4py.next.ffront.past_to_itir import ProgramLowering
from gt4py.next.ffront.source_utils import SourceDefinition, get_closure_vars_from_function
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils.ir_makers import (
    literal_from_value,
    promote_to_const_iterator,
    ref,
    sym,
)
from gt4py.next.program_processors import processor_interface as ppi
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DEFAULT_BACKEND: Callable = None


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
                        f"Collisions: '{',  '.join(collisions)}'."
                    )

                all_closure_vars = collections.ChainMap(all_closure_vars, all_child_closure_vars)
    return dict(all_closure_vars)


def _filter_closure_vars_by_type(closure_vars: dict[str, Any], *types: type) -> dict[str, Any]:
    return {name: value for name, value in closure_vars.items() if isinstance(value, types)}


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
        raise ValueError(
            "'grid_type == GridType.CARTESIAN' was requested, but unstructured 'FieldOffset' or local 'Dimension' was found."
        )

    return deduced_grid_type if requested_grid_type is None else requested_grid_type


def _field_constituents_shape_and_dims(
    arg, arg_type: ts.FieldType | ts.ScalarType | ts.TupleType
) -> Generator[tuple[tuple[int, ...], list[Dimension]]]:
    if isinstance(arg_type, ts.TupleType):
        for el, el_type in zip(arg, arg_type.types):
            yield from _field_constituents_shape_and_dims(el, el_type)
    elif isinstance(arg_type, ts.FieldType):
        dims = type_info.extract_dims(arg_type)
        if hasattr(arg, "shape"):
            assert len(arg.shape) == len(dims)
            yield (arg.shape, dims)
        else:
            yield (None, dims)
    elif isinstance(arg_type, ts.ScalarType):
        yield (None, [])
    else:
        raise ValueError("Expected 'FieldType' or 'TupleType' thereof.")


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
        grid_type: The grid type (cartesian or unstructured) to be used. If not explicitly given
            it will be deduced from actually occurring dimensions.
    """

    past_node: past.Program
    closure_vars: dict[str, Any]
    definition: Optional[types.FunctionType]
    backend: Optional[ppi.ProgramExecutor]
    grid_type: Optional[GridType]

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[ppi.ProgramExecutor],
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
                f"The following symbols resolve to a function with a mismatching name: {','.join(misnamed_functions)}."
            )

        undefined_symbols = [
            symbol.id
            for symbol in self.past_node.closure_vars
            if symbol.id not in self.closure_vars
        ]
        if undefined_symbols:
            raise RuntimeError(
                f"The following closure variables are undefined: {', '.join(undefined_symbols)}."
            )

    @property
    def __name__(self) -> str:
        return self.definition.__name__

    @functools.cached_property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        if self.backend:
            return self.backend.__gt_allocator__
        else:
            raise RuntimeError(f"Program '{self}' does not have a backend set.")

    def with_backend(self, backend: ppi.ProgramExecutor) -> Program:
        return dataclasses.replace(self, backend=backend)

    def with_grid_type(self, grid_type: GridType) -> Program:
        return dataclasses.replace(self, grid_type=grid_type)

    def with_bound_args(self, **kwargs) -> ProgramWithBoundArgs:
        """
        Bind scalar, i.e. non field, program arguments.

        Example (pseudo-code):

        >>> import gt4py.next as gtx
        >>> @gtx.program  # doctest: +SKIP
        ... def program(condition: bool, out: gtx.Field[[IDim], float]):  # noqa: F821 [undefined-name]
        ...     sample_field_operator(condition, out=out)  # noqa: F821 [undefined-name]

        Create a new program from `program` with the `condition` parameter set to `True`:

        >>> program_with_bound_arg = program.with_bound_args(condition=True)  # doctest: +SKIP

        The resulting program is equivalent to

        >>> @gtx.program  # doctest: +SKIP
        ... def program(condition: bool, out: gtx.Field[[IDim], float]):  # noqa: F821 [undefined-name]
        ...     sample_field_operator(condition=True, out=out)  # noqa: F821 [undefined-name]

        and can be executed without passing `condition`.

        >>> program_with_bound_arg(out, offset_provider={})  # doctest: +SKIP
        """
        for key in kwargs.keys():
            if all(key != param.id for param in self.past_node.params):
                raise TypeError(f"Keyword argument '{key}' is not a valid program parameter.")

        return ProgramWithBoundArgs(
            bound_args=kwargs,
            **{field.name: getattr(self, field.name) for field in dataclasses.fields(self)},
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

        if self.backend is None:
            warnings.warn(
                UserWarning(
                    f"Field View Program '{self.itir.id}': Using Python execution, consider selecting a perfomance backend."
                ),
                stacklevel=2,
            )
            with next_embedded.context.new_context(offset_provider=offset_provider) as ctx:
                ctx.run(self.definition, *rewritten_args, **kwargs)
            return

        ppi.ensure_processor_kind(self.backend, ppi.ProgramExecutor)
        if "debug" in kwargs:
            debug(self.itir)

        self.backend(
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
        arg_types = [type_translation.from_value(arg) for arg in args]
        kwarg_types = {k: type_translation.from_value(v) for k, v in kwargs.items()}

        try:
            type_info.accepts_args(
                self.past_node.type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
        except ValueError as err:
            raise errors.DSLError(
                None, f"Invalid argument types in call to '{self.past_node.id}'.\n{err}"
            ) from err

    def _process_args(self, args: tuple, kwargs: dict) -> tuple[tuple, tuple, dict[str, Any]]:
        self._validate_args(*args, **kwargs)

        args, kwargs = type_info.canonicalize_arguments(self.past_node.type, args, kwargs)

        implicit_domain = any(
            isinstance(stmt, past.Call) and "domain" not in stmt.kwargs
            for stmt in self.past_node.body
        )

        # extract size of all field arguments
        size_args: list[Optional[tuple[int, ...]]] = []
        rewritten_args = list(args)
        for param_idx, param in enumerate(self.past_node.params):
            if implicit_domain and isinstance(param.type, (ts.FieldType, ts.TupleType)):
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
            if isinstance(
                (type_ := gt_callable.__gt_type__()),
                ts_ffront.ScanOperatorType,
            ):
                scanops_per_axis.setdefault(type_.axis, []).append(name)

        if len(scanops_per_axis.values()) == 0:
            return None

        if len(scanops_per_axis.values()) != 1:
            scanops_per_axis_strs = [
                f"- {dim.value}: {', '.join(scanops)}" for dim, scanops in scanops_per_axis.items()
            ]

            raise TypeError(
                "Only 'ScanOperator's defined on the same axis "
                + "can be used in a 'Program', found:\n"
                + "\n".join(scanops_per_axis_strs)
                + "."
            )

        return iter(scanops_per_axis.keys()).__next__()


@dataclasses.dataclass(frozen=True)
class ProgramWithBoundArgs(Program):
    bound_args: dict[str, typing.Union[float, int, bool]] = None

    def _process_args(self, args: tuple, kwargs: dict):
        type_ = self.past_node.type
        new_type = ts_ffront.ProgramType(
            definition=ts.FunctionType(
                kw_only_args={
                    k: v
                    for k, v in type_.definition.kw_only_args.items()
                    if k not in self.bound_args.keys()
                },
                pos_only_args=type_.definition.pos_only_args,
                pos_or_kw_args={
                    k: v
                    for k, v in type_.definition.pos_or_kw_args.items()
                    if k not in self.bound_args.keys()
                },
                returns=type_.definition.returns,
            )
        )

        arg_types = [type_translation.from_value(arg) for arg in args]
        kwarg_types = {k: type_translation.from_value(v) for k, v in kwargs.items()}

        try:
            # This error is also catched using `accepts_args`, but we do it manually here to give
            # a better error message.
            for name in self.bound_args.keys():
                if name in kwargs:
                    raise ValueError(f"Parameter '{name}' already set as a bound argument.")

            type_info.accepts_args(
                new_type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
        except ValueError as err:
            bound_arg_names = ", ".join([f"'{bound_arg}'" for bound_arg in self.bound_args.keys()])
            raise TypeError(
                f"Invalid argument types in call to program '{self.past_node.id}' with "
                f"bound arguments '{bound_arg_names}'."
            ) from err

        full_args = [*args]
        full_kwargs = {**kwargs}
        for index, param in enumerate(self.past_node.params):
            if param.id in self.bound_args.keys():
                if index < len(full_args):
                    full_args.insert(index, self.bound_args[param.id])
                else:
                    full_kwargs[str(param.id)] = self.bound_args[param.id]

        return super()._process_args(tuple(full_args), full_kwargs)

    @functools.cached_property
    def itir(self):
        new_itir = super().itir
        for new_clos in new_itir.closures:
            new_args = [ref(inp.id) for inp in new_clos.inputs]
            for key, value in self.bound_args.items():
                index = next(
                    index
                    for index, closure_input in enumerate(new_clos.inputs)
                    if closure_input.id == key
                )
                new_args[new_args.index(new_clos.inputs[index])] = promote_to_const_iterator(
                    literal_from_value(value)
                )
                new_clos.inputs.pop(index)
            params = [sym(inp.id) for inp in new_clos.inputs]
            expr = itir.FunCall(
                fun=new_clos.stencil,
                args=new_args,
            )
            new_clos.stencil = itir.Lambda(params=params, expr=expr)
        return new_itir


@typing.overload
def program(definition: types.FunctionType) -> Program: ...


@typing.overload
def program(
    *, backend: Optional[ppi.ProgramExecutor]
) -> Callable[[types.FunctionType], Program]: ...


def program(
    definition=None,
    *,
    backend=eve.NOTHING,  # `NOTHING` -> default backend, `None` -> no backend (embedded execution)
    grid_type=None,
) -> Program | Callable[[types.FunctionType], Program]:
    """
    Generate an implementation of a program from a Python function object.

    Examples:
        >>> @program  # noqa: F821 [undefined-name]  # doctest: +SKIP
        ... def program(in_field: Field[[TDim], float64], out_field: Field[[TDim], float64]):  # noqa: F821 [undefined-name]
        ...     field_op(in_field, out=out_field)
        >>> program(in_field, out=out_field)  # noqa: F821 [undefined-name]  # doctest: +SKIP

        >>> # the backend can optionally be passed if already decided
        >>> # not passing it will result in embedded execution by default
        >>> # the above is equivalent to
        >>> @program(backend="roundtrip")  # noqa: F821 [undefined-name]  # doctest: +SKIP
        ... def program(in_field: Field[[TDim], float64], out_field: Field[[TDim], float64]):  # noqa: F821 [undefined-name]
        ...     field_op(in_field, out=out_field)
        >>> program(in_field, out=out_field)  # noqa: F821 [undefined-name]  # doctest: +SKIP
    """

    def program_inner(definition: types.FunctionType) -> Program:
        return Program.from_function(
            definition, DEFAULT_BACKEND if backend is eve.NOTHING else backend, grid_type
        )

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
        grid_type: The grid type (cartesian or unstructured) to be used. If not explicitly given
            it will be deduced from actually occurring dimensions.
    """

    foast_node: OperatorNodeT
    closure_vars: dict[str, Any]
    definition: Optional[types.FunctionType]
    backend: Optional[ppi.ProgramExecutor]
    grid_type: Optional[GridType]
    operator_attributes: Optional[dict[str, Any]] = None
    _program_cache: dict = dataclasses.field(
        init=False, default_factory=dict
    )  # init=False ensure the cache is not copied in calls to replace

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[ppi.ProgramExecutor],
        grid_type: Optional[GridType] = None,
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
            key: foast.Constant(value=value, type=type_translation.from_value(value), location=loc)
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
            definition=definition,
            backend=backend,
            grid_type=grid_type,
            operator_attributes=operator_attributes,
        )

    @property
    def __name__(self) -> str:
        return self.definition.__name__

    def __gt_type__(self) -> ts.CallableType:
        type_ = self.foast_node.type
        assert isinstance(type_, ts.CallableType)
        return type_

    def with_backend(self, backend: ppi.ProgramExecutor) -> FieldOperator:
        return dataclasses.replace(self, backend=backend)

    def with_grid_type(self, grid_type: GridType) -> FieldOperator:
        return dataclasses.replace(self, grid_type=grid_type)

    def __gt_itir__(self) -> itir.FunctionDefinition:
        if hasattr(self, "__cached_itir"):
            return getattr(self, "__cached_itir")

        itir_node: itir.FunctionDefinition = FieldOperatorLowering.apply(self.foast_node)

        object.__setattr__(self, "__cached_itir", itir_node)

        return itir_node

    def __gt_closure_vars__(self) -> dict[str, Any]:
        return self.closure_vars

    def as_program(
        self, arg_types: list[ts.TypeSpec], kwarg_types: dict[str, ts.TypeSpec]
    ) -> Program:
        # TODO(tehrengruber): implement mechanism to deduce default values
        #  of arg and kwarg types
        # TODO(tehrengruber): check foast operator has no out argument that clashes
        #  with the out argument of the program we generate here.
        hash_ = eve_utils.content_hash((
            tuple(arg_types),
            tuple((name, arg) for name, arg in kwarg_types.items()),
        ))
        try:
            return self._program_cache[hash_]
        except KeyError:
            pass

        loc = self.foast_node.location
        param_sym_uids = eve_utils.UIDGenerator()  # use a new UID generator to allow caching

        type_ = self.__gt_type__()
        params_decl: list[past.Symbol] = [
            past.DataSymbol(
                id=param_sym_uids.sequential_id(prefix="__sym"),
                type=arg_type,
                namespace=dialect_ast_enums.Namespace.LOCAL,
                location=loc,
            )
            for arg_type in arg_types
        ]
        params_ref = [past.Name(id=pdecl.id, location=loc) for pdecl in params_decl]
        out_sym: past.Symbol = past.DataSymbol(
            id="out",
            type=type_info.return_type(type_, with_args=arg_types, with_kwargs=kwarg_types),
            namespace=dialect_ast_enums.Namespace.LOCAL,
            location=loc,
        )
        out_ref = past.Name(id="out", location=loc)

        if self.foast_node.id in self.closure_vars:
            raise RuntimeError("A closure variable has the same name as the field operator itself.")
        closure_vars = {self.foast_node.id: self}
        closure_symbols = [
            past.Symbol(
                id=self.foast_node.id,
                type=ts.DeferredType(constraint=None),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=loc,
            ),
        ]

        untyped_past_node = past.Program(
            id=f"__field_operator_{self.foast_node.id}",
            type=ts.DeferredType(constraint=ts_ffront.ProgramType),
            params=[*params_decl, out_sym],
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

        self._program_cache[hash_] = Program(
            past_node=past_node,
            closure_vars=closure_vars,
            definition=None,
            backend=self.backend,
            grid_type=self.grid_type,
        )
        return self._program_cache[hash_]

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> None:
        if not next_embedded.context.within_context() and self.backend is not None:
            # non embedded execution
            if "offset_provider" not in kwargs:
                raise errors.MissingArgumentError(None, "offset_provider", True)
            offset_provider = kwargs.pop("offset_provider")

            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            out = kwargs.pop("out")
            args, kwargs = type_info.canonicalize_arguments(self.foast_node.type, args, kwargs)
            # TODO(tehrengruber): check all offset providers are given
            # deduce argument types
            arg_types = []
            for arg in args:
                arg_types.append(type_translation.from_value(arg))
            kwarg_types = {}
            for name, arg in kwargs.items():
                kwarg_types[name] = type_translation.from_value(arg)

            return self.as_program(arg_types, kwarg_types)(
                *args, out, offset_provider=offset_provider, **kwargs
            )
        else:
            if self.operator_attributes is not None and any(
                has_scan_op_attribute := [
                    attribute in self.operator_attributes
                    for attribute in ["init", "axis", "forward"]
                ]
            ):
                assert all(has_scan_op_attribute)
                forward = self.operator_attributes["forward"]
                init = self.operator_attributes["init"]
                axis = self.operator_attributes["axis"]
                op = embedded_operators.ScanOperator(self.definition, forward, init, axis)
            else:
                op = embedded_operators.EmbeddedOperator(self.definition)
            return embedded_operators.field_operator_call(op, args, kwargs)


@typing.overload
def field_operator(
    definition: types.FunctionType, *, backend: Optional[ppi.ProgramExecutor]
) -> FieldOperator[foast.FieldOperator]: ...


@typing.overload
def field_operator(
    *, backend: Optional[ppi.ProgramExecutor]
) -> Callable[[types.FunctionType], FieldOperator[foast.FieldOperator]]: ...


def field_operator(definition=None, *, backend=eve.NOTHING, grid_type=None):
    """
    Generate an implementation of the field operator from a Python function object.

    Examples:
        >>> @field_operator  # doctest: +SKIP
        ... def field_op(in_field: Field[[TDim], float64]) -> Field[[TDim], float64]:  # noqa: F821 [undefined-name]
        ...     ...
        >>> field_op(in_field, out=out_field)  # noqa: F821 [undefined-name]  # doctest: +SKIP

        >>> # the backend can optionally be passed if already decided
        >>> # not passing it will result in embedded execution by default
        >>> @field_operator(backend="roundtrip")  # doctest: +SKIP
        ... def field_op(in_field: Field[[TDim], float64]) -> Field[[TDim], float64]:  # noqa: F821 [undefined-name]
        ...     ...
    """

    def field_operator_inner(definition: types.FunctionType) -> FieldOperator[foast.FieldOperator]:
        return FieldOperator.from_function(
            definition, DEFAULT_BACKEND if backend is eve.NOTHING else backend, grid_type
        )

    return field_operator_inner if definition is None else field_operator_inner(definition)


@typing.overload
def scan_operator(
    definition: types.FunctionType,
    *,
    axis: Dimension,
    forward: bool,
    init: core_defs.Scalar,
    backend: Optional[str],
    grid_type: GridType,
) -> FieldOperator[foast.ScanOperator]: ...


@typing.overload
def scan_operator(
    *,
    axis: Dimension,
    forward: bool,
    init: core_defs.Scalar,
    backend: Optional[str],
    grid_type: GridType,
) -> Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]: ...


def scan_operator(
    definition: Optional[types.FunctionType] = None,
    *,
    axis: Dimension,
    forward: bool = True,
    init: core_defs.Scalar = 0.0,
    backend=eve.NOTHING,
    grid_type: GridType = None,
) -> (
    FieldOperator[foast.ScanOperator]
    | Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]
):
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
        >>> import gt4py.next as gtx
        >>> from gt4py.next.iterator import embedded
        >>> embedded._column_range = 1  # implementation detail
        >>> KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
        >>> inp = gtx.as_field([KDim], np.ones((10,)))
        >>> out = gtx.as_field([KDim], np.zeros((10,)))
        >>> @gtx.scan_operator(axis=KDim, forward=True, init=0.0)
        ... def scan_operator(carry: float, val: float) -> float:
        ...     return carry + val
        >>> scan_operator(inp, out=out, offset_provider={})  # doctest: +SKIP
        >>> out.array()  # doctest: +SKIP
        array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    """
    # TODO(tehrengruber): enable doctests again. For unknown / obscure reasons
    #  the above doctest fails when executed using `pytest --doctest-modules`.

    def scan_operator_inner(definition: types.FunctionType) -> FieldOperator:
        return FieldOperator.from_function(
            definition,
            DEFAULT_BACKEND if backend is eve.NOTHING else backend,
            grid_type,
            operator_node_cls=foast.ScanOperator,
            operator_attributes={"axis": axis, "forward": forward, "init": init},
        )

    return scan_operator_inner if definition is None else scan_operator_inner(definition)
