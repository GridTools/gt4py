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

import dataclasses
import functools
import types
import typing
import warnings
from collections.abc import Callable
from typing import Any, Generic, Optional, TypeVar

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import (
    allocators as next_allocators,
    backend as next_backend,
    embedded as next_embedded,
    errors,
)
from gt4py.next.common import Connectivity, Dimension, GridType
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import (
    field_operator_ast as foast,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
    transform_utils,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.gtcallable import GTCallable
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
        closure_vars: Mapping of externally defined symbols to their respective values.
            For example, referenced global and nonlocal variables.
        backend: The backend to be used for code generation.
        definition: The Python function object corresponding to the PAST node.
        grid_type: The grid type (cartesian or unstructured) to be used. If not explicitly given
            it will be deduced from actually occurring dimensions.
        connectivities: A dictionary holding static/compile-time information about the offset providers.
            For now, it is used for ahead of time compilation in DaCe orchestrated programs,
            i.e. DaCe programs that call GT4Py Programs -SDFGConvertible interface-.
    """

    definition_stage: ffront_stages.ProgramDefinition
    backend: Optional[next_backend.Backend]
    connectivities: Optional[dict[str, Connectivity]]

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[next_backend],
        grid_type: Optional[GridType] = None,
        connectivities: Optional[dict[str, Connectivity]] = None,
    ) -> Program:
        program_def = ffront_stages.ProgramDefinition(definition=definition, grid_type=grid_type)
        return cls(definition_stage=program_def, backend=backend, connectivities=connectivities)

    # needed in testing
    @property
    def definition(self):
        return self.definition_stage.definition

    @functools.cached_property
    def past_stage(self):
        # backwards compatibility for backends that do not support the full toolchain
        if self.backend is not None and self.backend.transforms_prog is not None:
            return self.backend.transforms_prog.func_to_past(self.definition_stage)
        return next_backend.DEFAULT_PROG_TRANSFORMS.func_to_past(self.definition_stage)

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self):
        if self.backend is not None and self.backend.transforms_prog is not None:
            self.backend.transforms_prog.past_lint(self.past_stage)
        return next_backend.DEFAULT_PROG_TRANSFORMS.past_lint(self.past_stage)

    @property
    def __name__(self) -> str:
        return self.definition_stage.definition.__name__

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

    def with_connectivities(self, connectivities: dict[str, Connectivity]) -> Program:
        return dataclasses.replace(self, connectivities=connectivities)

    def with_grid_type(self, grid_type: GridType) -> Program:
        return dataclasses.replace(
            self, definition_stage=dataclasses.replace(self.definition_stage, grid_type=grid_type)
        )

    def with_bound_args(self, **kwargs: Any) -> ProgramWithBoundArgs:
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
            if all(key != param.id for param in self.past_stage.past_node.params):
                raise TypeError(f"Keyword argument '{key}' is not a valid program parameter.")

        return ProgramWithBoundArgs(
            bound_args=kwargs,
            **{field.name: getattr(self, field.name) for field in dataclasses.fields(self)},
        )

    @functools.cached_property
    def _all_closure_vars(self) -> dict[str, Any]:
        return transform_utils._get_closure_vars_recursively(self.past_stage.closure_vars)

    @functools.cached_property
    def itir(self) -> itir.FencilDefinition:
        no_args_past = ffront_stages.PastClosure(
            past_node=self.past_stage.past_node,
            closure_vars=self.past_stage.closure_vars,
            grid_type=self.definition_stage.grid_type,
            args=[],
            kwargs={},
        )
        if self.backend is not None and self.backend.transforms_prog is not None:
            return self.backend.transforms_prog.past_to_itir(no_args_past).program
        return past_to_itir.PastToItirFactory()(no_args_past).program

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs: Any) -> None:
        if self.backend is None:
            warnings.warn(
                UserWarning(
                    f"Field View Program '{self.itir.id}': Using Python execution, consider selecting a perfomance backend."
                ),
                stacklevel=2,
            )
            with next_embedded.context.new_context(offset_provider=offset_provider) as ctx:
                # TODO(ricoh): check if rewriting still needed
                rewritten_args, size_args, kwargs = past_process_args._process_args(
                    self.past_stage.past_node, args, kwargs
                )
                ctx.run(self.definition_stage.definition, *rewritten_args, **kwargs)
            return

        ppi.ensure_processor_kind(self.backend.executor, ppi.ProgramExecutor)

        self.backend(
            self.definition_stage, *args, **(kwargs | {"offset_provider": offset_provider})
        )


try:
    from gt4py.next.program_processors.runners.dace_iterator import Program
except ImportError:
    pass


@dataclasses.dataclass(frozen=True)
class ProgramFromPast(Program):
    """
    This version of program has no DSL definition associated with it.

    PAST nodes can be built programmatically from field operators or from scratch.
    This wrapper provides the appropriate toolchain entry points.
    """

    past_stage: ffront_stages.PastProgramDefinition

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs):
        if self.backend is None:
            raise NotImplementedError(
                "Programs created from a PAST node (without a function definition) can not be executed in embedded mode"
            )

        ppi.ensure_processor_kind(self.backend.executor, ppi.ProgramExecutor)
        self.backend(self.past_stage, *args, **(kwargs | {"offset_provider": offset_provider}))

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self):
        if self.backend is not None and self.backend.transforms_prog is not None:
            self.backend.transforms_prog.past_lint(self.past_stage)
        return next_backend.DEFAULT_PROG_TRANSFORMS.past_lint(self.past_stage)


@dataclasses.dataclass(frozen=True)
class ProgramWithBoundArgs(Program):
    bound_args: dict[str, typing.Union[float, int, bool]] = None

    def __call__(self, *args, offset_provider: dict[str, Dimension], **kwargs):
        type_ = self.past_stage.past_node.type
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
                new_type, with_args=arg_types, with_kwargs=kwarg_types, raise_exception=True
            )
        except ValueError as err:
            bound_arg_names = ", ".join([f"'{bound_arg}'" for bound_arg in self.bound_args.keys()])
            raise TypeError(
                f"Invalid argument types in call to program '{self.past_stage.past_node.id}' with "
                f"bound arguments '{bound_arg_names}'."
            ) from err

        full_args = [*args]
        full_kwargs = {**kwargs}
        for index, param in enumerate(self.past_stage.past_node.params):
            if param.id in self.bound_args.keys():
                if index < len(full_args):
                    full_args.insert(index, self.bound_args[param.id])
                else:
                    full_kwargs[str(param.id)] = self.bound_args[param.id]

        return super().__call__(*tuple(full_args), offset_provider=offset_provider, **full_kwargs)

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
            expr = itir.FunCall(fun=new_clos.stencil, args=new_args)
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
    # `NOTHING` -> default backend, `None` -> no backend (embedded execution)
    backend=eve.NOTHING,
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

    definition_stage: ffront_stages.FieldOperatorDefinition
    backend: Optional[ppi.ProgramExecutor]
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
        return cls(
            definition_stage=ffront_stages.FieldOperatorDefinition(
                definition=definition,
                grid_type=grid_type,
                node_class=operator_node_cls,
                attributes=operator_attributes or {},
            ),
            backend=backend,
        )

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self):
        """This ensures that DSL linting occurs at decoration time."""
        _ = self.foast_stage

    @functools.cached_property
    def foast_stage(self) -> ffront_stages.FoastOperatorDefinition:
        if self.backend is not None and self.backend.transforms_fop is not None:
            return self.backend.transforms_fop.func_to_foast(self.definition_stage)
        return next_backend.DEFAULT_FIELDOP_TRANSFORMS.func_to_foast(self.definition_stage)

    @property
    def __name__(self) -> str:
        return self.definition_stage.definition.__name__

    @property
    def definition(self) -> str:
        return self.definition_stage.definition

    def __gt_type__(self) -> ts.CallableType:
        type_ = self.foast_stage.foast_node.type
        assert isinstance(type_, ts.CallableType)
        return type_

    def with_backend(self, backend: ppi.ProgramExecutor) -> FieldOperator:
        return dataclasses.replace(self, backend=backend)

    def with_grid_type(self, grid_type: GridType) -> FieldOperator:
        return dataclasses.replace(
            self, definition_stage=dataclasses.replace(self.definition_stage, grid_type=grid_type)
        )

    def __gt_itir__(self) -> itir.FunctionDefinition:
        if self.backend is not None and self.backend.transforms_fop is not None:
            return self.backend.transforms_fop.foast_to_itir(self.foast_stage)
        return next_backend.DEFAULT_FIELDOP_TRANSFORMS.foast_to_itir(self.foast_stage)

    def __gt_closure_vars__(self) -> dict[str, Any]:
        return self.foast_stage.closure_vars

    def as_program(
        self, arg_types: list[ts.TypeSpec], kwarg_types: dict[str, ts.TypeSpec]
    ) -> Program:
        foast_with_types = (
            ffront_stages.FoastWithTypes(
                foast_op_def=self.foast_stage,
                arg_types=tuple(arg_types),
                kwarg_types=kwarg_types,
                closure_vars={self.foast_stage.foast_node.id: self},
            ),
        )
        past_stage = None
        if self.backend is not None and self.backend.transforms_fop is not None:
            past_stage = self.backend.transforms_fop.foast_to_past_closure.foast_to_past(
                foast_with_types
            )
        else:
            past_stage = (
                next_backend.DEFAULT_FIELDOP_TRANSFORMS.foast_to_past_closure.foast_to_past(
                    ffront_stages.FoastWithTypes(
                        foast_op_def=self.foast_stage,
                        arg_types=tuple(arg_types),
                        kwarg_types=kwarg_types,
                        closure_vars={self.foast_stage.foast_node.id: self},
                    ),
                )
            )
        return ProgramFromPast(definition_stage=None, past_stage=past_stage, backend=self.backend)

    def __call__(self, *args, **kwargs) -> None:
        if not next_embedded.context.within_valid_context() and self.backend is not None:
            # non embedded execution
            if "offset_provider" not in kwargs:
                raise errors.MissingArgumentError(None, "offset_provider", True)
            offset_provider = kwargs.pop("offset_provider")

            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            out = kwargs.pop("out")
            args, kwargs = type_info.canonicalize_arguments(
                self.foast_stage.foast_node.type, args, kwargs
            )
            return self.backend(
                self.definition_stage,
                *args,
                out=out,
                offset_provider=offset_provider,
                from_fieldop=self,
                **kwargs,
            )
        else:
            attributes = (
                self.definition_stage.attributes
                if self.definition_stage
                else self.foast_stage.attributes
            )
            if attributes is not None and any(
                has_scan_op_attribute := [
                    attribute in attributes for attribute in ["init", "axis", "forward"]
                ]
            ):
                assert all(has_scan_op_attribute)
                forward = attributes["forward"]
                init = attributes["init"]
                axis = attributes["axis"]
                op = embedded_operators.ScanOperator(
                    self.definition_stage.definition, forward, init, axis
                )
            else:
                op = embedded_operators.EmbeddedOperator(self.definition_stage.definition)
            return embedded_operators.field_operator_call(op, args, kwargs)


@dataclasses.dataclass(frozen=True)
class FieldOperatorFromFoast(FieldOperator):
    """
    This version of the field operator does not have a DSL definition.

    FieldOperator AST nodes can be programmatically built, which may be
    particularly useful in testing and debugging.
    This class provides the appropriate toolchain entry points.
    """

    foast_stage: ffront_stages.FoastOperatorDefinition

    def __call__(self, *args, **kwargs) -> None:
        return self.backend(self.foast_stage, *args, from_fieldop=self, **kwargs)


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


@ffront_stages.add_content_to_fingerprint.register
def add_fieldop_to_fingerprint(obj: FieldOperator, hasher: xtyping.HashlibAlgorithm) -> None:
    ffront_stages.add_content_to_fingerprint(obj.definition_stage, hasher)
    ffront_stages.add_content_to_fingerprint(obj.backend, hasher)


@ffront_stages.add_content_to_fingerprint.register
def add_foast_fieldop_to_fingerprint(
    obj: FieldOperatorFromFoast, hasher: xtyping.HashlibAlgorithm
) -> None:
    ffront_stages.add_content_to_fingerprint(obj.foast_stage, hasher)
    ffront_stages.add_content_to_fingerprint(obj.backend, hasher)


@ffront_stages.add_content_to_fingerprint.register
def add_program_to_fingerprint(obj: Program, hasher: xtyping.HashlibAlgorithm) -> None:
    ffront_stages.add_content_to_fingerprint(obj.definition_stage, hasher)
    ffront_stages.add_content_to_fingerprint(obj.backend, hasher)


@ffront_stages.add_content_to_fingerprint.register
def add_past_program_to_fingerprint(obj: ProgramFromPast, hasher: xtyping.HashlibAlgorithm) -> None:
    ffront_stages.add_content_to_fingerprint(obj.past_stage, hasher)
    ffront_stages.add_content_to_fingerprint(obj.backend, hasher)
