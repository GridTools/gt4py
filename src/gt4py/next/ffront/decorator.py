# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(tehrengruber): This file contains to many different components. Split
#  into components for each dialect.


from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import types
import typing
import warnings
from collections.abc import Callable
from typing import Any, Generic, Optional, Sequence, TypeAlias

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import Self, Unpack, override
from gt4py.next import (
    backend as next_backend,
    common,
    custom_layout_allocators as next_allocators,
    embedded as next_embedded,
    errors,
    utils,
)
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import (
    field_operator_ast as foast,
    foast_to_gtir,
    past_process_args,
    stages as ffront_stages,
    transform_utils,
    type_info as ffront_type_info,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.instrumentation import hook_machinery, metrics
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, compiled_program, options, toolchain
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DEFAULT_BACKEND: next_backend.Backend | None = None


ProgramCallMetricsCollector = metrics.make_collector(
    level=metrics.MINIMAL, metric_name=metrics.TOTAL_METRIC
)


@hook_machinery.context_hook
def program_call_context(
    program: Program,
    args: tuple[Any, ...],
    offset_provider: common.OffsetProvider,
    enable_jit: bool,
    kwargs: dict[str, Any],
) -> contextlib.AbstractContextManager:
    """Hook called at the beginning and end of a program call."""
    return ProgramCallMetricsCollector()


@hook_machinery.context_hook
def embedded_program_call_context(
    program: Program,
    args: tuple[Any, ...],
    offset_provider: common.OffsetProvider,
    kwargs: dict[str, Any],
) -> contextlib.AbstractContextManager:
    """Hook called at the beginning and end of an embedded program call."""
    return metrics.metrics_context(f"{program.__name__}<'<embedded>')>")


@dataclasses.dataclass(frozen=True)
class _CompilableGTEntryPointMixin(Generic[ffront_stages.DSLDefinitionT]):
    """
    Mixing used by program and program-like objects.

    Contains functionality and configuration options common to all kinds of program-likes.
    """

    definition_stage: ffront_stages.DSLDefinitionT
    backend: Optional[next_backend.Backend]
    compilation_options: options.CompilationOptions

    @abc.abstractmethod
    def __gt_type__(self) -> ts.CallableType: ...

    def with_backend(self, backend: next_backend.Backend) -> Self:
        return dataclasses.replace(self, backend=backend)

    def with_compilation_options(
        self, **compilation_options: Unpack[options.CompilationOptionsArgs]
    ) -> Self:
        return dataclasses.replace(
            self,
            compilation_options=dataclasses.replace(
                self.compilation_options, **compilation_options
            ),
        )

    @functools.cached_property
    def _compiled_programs(self) -> compiled_program.CompiledProgramsPool:
        # This cached property initializer is only called when JITting the first
        # program variant of the pool. If the program is compiled by directly
        # calling `compile()`, the pool is initialized with the options passed
        # to `compile()` instead of re-using the existing compilations options.
        return self._make_compiled_programs_pool(
            static_params=self.compilation_options.static_params or (),
            static_domains=self.compilation_options.static_domains,
        )

    def _make_compiled_programs_pool(
        self, static_params: Sequence[str], static_domains: bool
    ) -> compiled_program.CompiledProgramsPool:
        if self.backend is None or self.backend == eve.NOTHING:
            raise RuntimeError("Cannot compile a program without backend.")

        program_type = ffront_type_info.type_in_program_context(self.__gt_type__())
        assert isinstance(program_type, ts_ffront.ProgramType)

        argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]] = {}

        if static_params:
            argument_descriptor_mapping[arguments.StaticArg] = static_params

        if static_domains:
            argument_descriptor_mapping[arguments.FieldDomainDescriptor] = (
                _field_domain_descriptor_mapping_from_func_type(program_type.definition)
            )

        return compiled_program.CompiledProgramsPool(
            backend=self.backend,
            definition_stage=self.definition_stage,
            program_type=program_type,
            argument_descriptor_mapping=argument_descriptor_mapping,
        )

    def compile(
        self,
        offset_provider: common.OffsetProviderType
        | common.OffsetProvider
        | list[common.OffsetProviderType | common.OffsetProvider]
        | None = None,
        **static_args: list[xtyping.MaybeNestedInTuple[core_defs.Scalar]],
    ) -> Self:
        """
        Compiles the program or operator for the given combination of static arguments and offset
        provider type.

        Note: Unlike `with_...` methods, this method does not return a new instance of the program,
        but adds the compiled variants to the current program instance.
        """
        # TODO(havogt): we should reconsider if we want to return a new program on `compile` (and
        #  rename to `with_static_args` or similar) once we have a better understanding of the
        #  use-cases.
        # check if pool has already been initialized. since this is also a cached property go via
        #  the dict directly. Note that we don't need to check any args, since the pool checks
        #  this on compile anyway.
        if "_compiled_programs" not in self.__dict__:
            self.__dict__["_compiled_programs"] = self._make_compiled_programs_pool(
                static_params=tuple(static_args.keys()),
                static_domains=self.compilation_options.static_domains,
            )

        if self.compilation_options.connectivities is None and offset_provider is None:
            raise ValueError(
                "Cannot compile a program without connectivities / OffsetProviderType."
            )
        if not all(isinstance(v, list) for v in static_args.values()):
            raise TypeError(
                "Please provide the static arguments as lists."
            )  # To avoid confusion with tuple args

        offset_provider = (
            self.compilation_options.connectivities if offset_provider is None else offset_provider
        )
        if not isinstance(offset_provider, list):
            offset_provider = [offset_provider]  # type: ignore[list-item] # cleanup offset_provider vs offset_provider_type

        assert all(
            common.is_offset_provider(op) or common.is_offset_provider_type(op)
            for op in offset_provider
        )

        self._compiled_programs.compile(offset_providers=offset_provider, **static_args)
        return self


def _field_domain_descriptor_mapping_from_func_type(func_type: ts.FunctionType) -> list[str]:
    static_domain_args = []
    param_types = func_type.pos_or_kw_args | func_type.kw_only_args
    for name, type_ in param_types.items():
        for el_type_, path in type_info.primitive_constituents(type_, with_path_arg=True):
            if isinstance(el_type_, ts.FieldType):
                path_as_expr = "".join(f"[{idx}]" for idx in path)
                static_domain_args.append(f"{name}{path_as_expr}")
    return static_domain_args


# TODO(tehrengruber): Decide if and how programs can call other programs. As a
#  result Program could become a GTCallable.
@dataclasses.dataclass(frozen=True)
class Program(_CompilableGTEntryPointMixin[ffront_stages.DSLProgramDef]):
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

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: next_backend.Backend | None,
        grid_type: common.GridType | None = None,
        **compilation_options: Unpack[options.CompilationOptionsArgs],
    ) -> Program:
        program_def = ffront_stages.DSLProgramDef(definition=definition, grid_type=grid_type)
        return cls(
            definition_stage=program_def,
            backend=backend,
            compilation_options=options.CompilationOptions(**compilation_options),
        )

    def __gt_type__(self) -> ts_ffront.ProgramType:
        assert isinstance(self.past_stage.past_node.type, ts_ffront.ProgramType)
        return self.past_stage.past_node.type

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self) -> None:
        no_args_past = toolchain.ConcreteArtifact(
            self.past_stage, arguments.CompileTimeArgs.empty()
        )
        _ = self._frontend_transforms.past_lint(no_args_past).data

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

    @property
    def definition(self) -> types.FunctionType:
        return self.definition_stage.definition

    @functools.cached_property
    def past_stage(self) -> ffront_stages.PASTProgramDef:
        # backwards compatibility for backends that do not support the full toolchain
        no_args_def = toolchain.ConcreteArtifact(
            self.definition_stage, arguments.CompileTimeArgs.empty()
        )
        return self._frontend_transforms.func_to_past(no_args_def).data

    @property
    def _frontend_transforms(self) -> next_backend.Transforms:
        if self.backend is None:
            return next_backend.DEFAULT_TRANSFORMS
        # TODO(tehrengruber): This class relies heavily on `self.backend.transforms` being
        #  a `next_backend.Transforms`, but the backend type annotation does not reflect that.
        assert isinstance(self.backend.transforms, next_backend.Transforms)
        return self.backend.transforms

    @functools.cached_property
    def _all_closure_vars(self) -> dict[str, Any]:
        return transform_utils._get_closure_vars_recursively(self.past_stage.closure_vars)

    @functools.cached_property
    def gtir(self) -> itir.Program:
        no_args_past = toolchain.ConcreteArtifact(
            data=ffront_stages.PASTProgramDef(
                past_node=self.past_stage.past_node,
                closure_vars=self.past_stage.closure_vars,
                grid_type=self.definition_stage.grid_type,
            ),
            args=arguments.CompileTimeArgs.empty(),
        )
        return self._frontend_transforms.past_to_itir(no_args_past).data

    def with_grid_type(self, grid_type: common.GridType) -> Program:
        return dataclasses.replace(
            self, definition_stage=dataclasses.replace(self.definition_stage, grid_type=grid_type)
        )

    def with_static_params(self, *static_params: str | None) -> Program:
        if not static_params or (static_params == (None,)):
            _static_params: tuple[str, ...] = ()
        else:
            assert all(p is not None for p in static_params)
            _static_params = typing.cast(tuple[str, ...], static_params)
        return dataclasses.replace(
            self,
            compilation_options=dataclasses.replace(
                self.compilation_options, static_params=_static_params
            ),
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
            **{
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.init
            },
        )

    def __call__(
        self,
        *args: Any,
        offset_provider: common.OffsetProvider | None = None,
        enable_jit: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if offset_provider is None:
            offset_provider = {}
        enable_jit = self.compilation_options.enable_jit if enable_jit is None else enable_jit

        with program_call_context(
            program=self,
            args=args,
            offset_provider=offset_provider,
            enable_jit=enable_jit,
            kwargs=kwargs,
        ):
            if __debug__:
                # TODO: remove or make dependency on self.past_stage optional
                past_process_args._validate_args(
                    self.past_stage.past_node,
                    arg_types=[type_translation.from_value(arg) for arg in args],
                    kwarg_types={k: type_translation.from_value(v) for k, v in kwargs.items()},
                )

            if self.backend is not None:
                self._compiled_programs(
                    *args, **kwargs, offset_provider=offset_provider, enable_jit=enable_jit
                )
            else:
                # Embedded execution.
                warnings.warn(
                    UserWarning(
                        f"Field View Program '{self.definition_stage.definition.__name__}': Using Python execution, consider selecting a performance backend."
                    ),
                    stacklevel=2,
                )

                with next_embedded.context.update(offset_provider=offset_provider):
                    with embedded_program_call_context(self, args, offset_provider, kwargs):
                        self.definition_stage.definition(*args, **kwargs)


try:
    from gt4py.next.program_processors.runners.dace.program import (  # type: ignore[assignment]
        Program,
    )
except ImportError:
    pass


@dataclasses.dataclass(frozen=True)
class ProgramWithBoundArgs(Program):
    bound_args: dict[str, float | int | bool] = dataclasses.field(default_factory=dict)

    @override
    def __call__(
        self, *args: Any, offset_provider: common.OffsetProvider | None = None, **kwargs: Any
    ) -> None:
        if offset_provider is None:
            offset_provider = {}
        type_ = self.past_stage.past_node.type
        assert isinstance(type_, ts_ffront.ProgramType)
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

    @override
    def compile(
        self,
        offset_provider: common.OffsetProviderType
        | common.OffsetProvider
        | list[common.OffsetProviderType | common.OffsetProvider]
        | None = None,
        **static_args: list[xtyping.MaybeNestedInTuple[core_defs.Scalar]],
    ) -> Self:
        raise NotImplementedError("Compilation of programs with bound arguments is not implemented")


@typing.overload
def program(definition: types.FunctionType) -> Program: ...


@typing.overload
def program(
    *,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
    **compilation_options: Unpack[options.CompilationOptionsArgs],
) -> Callable[[types.FunctionType], Program]: ...


def program(
    definition: types.FunctionType | None = None,
    *,
    # `NOTHING` -> default backend, `None` -> no backend (embedded execution)
    backend: next_backend.Backend | eve.NothingType | None = eve.NOTHING,
    grid_type: common.GridType | None = None,
    **compilation_options: Unpack[options.CompilationOptionsArgs],
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
        program = Program.from_function(
            definition,
            backend=typing.cast(
                next_backend.Backend | None, DEFAULT_BACKEND if backend is eve.NOTHING else backend
            ),
            grid_type=grid_type,
            **compilation_options,
        )
        return program

    return program_inner if definition is None else program_inner(definition)


@dataclasses.dataclass(frozen=True)
class FieldOperator(_CompilableGTEntryPointMixin[ffront_stages.DSLFieldOperatorDef], GTCallable):
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

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[next_backend.Backend],
        grid_type: Optional[common.GridType] = None,
        *,
        operator_node_cls: type[foast.OperatorNode] = foast.FieldOperator,
        operator_attributes: Optional[dict[str, Any]] = None,
        **compilation_options: Unpack[options.CompilationOptionsArgs],
    ) -> FieldOperator:
        return cls(
            definition_stage=ffront_stages.DSLFieldOperatorDef(
                definition=definition,
                grid_type=grid_type,
                node_class=operator_node_cls,
                attributes=operator_attributes or {},
            ),
            backend=backend,
            compilation_options=options.CompilationOptions(**compilation_options),
        )

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self) -> None:
        """This ensures that DSL linting occurs at decoration time."""
        _ = self.foast_stage

    @functools.cached_property
    def foast_stage(self) -> ffront_stages.FOASTOperatorDef:
        return self._frontend_transforms.func_to_foast(
            toolchain.ConcreteArtifact(
                data=self.definition_stage, args=arguments.CompileTimeArgs.empty()
            )
        ).data

    @property
    def __name__(self) -> str:
        return self.definition_stage.definition.__name__

    @property
    def definition(self) -> types.FunctionType:
        return self.definition_stage.definition

    @property
    def _frontend_transforms(self) -> next_backend.Transforms:
        if self.backend is None:
            return next_backend.DEFAULT_TRANSFORMS
        # TODO(tehrengruber): This class relies heavily on `self.backend.transforms` being
        #  a `next_backend.Transforms`, but the backend type annotation does not reflect that.
        assert isinstance(self.backend.transforms, next_backend.Transforms)
        return self.backend.transforms

    def __gt_type__(self) -> ts.CallableType:
        type_ = self.foast_stage.foast_node.type
        assert isinstance(type_, ts.CallableType)
        return type_

    def with_grid_type(self, grid_type: common.GridType) -> FieldOperator:
        return dataclasses.replace(
            self, definition_stage=dataclasses.replace(self.definition_stage, grid_type=grid_type)
        )

    # TODO(tehrengruber): We can not use transforms from `self.backend` since this can be
    #  a different backend than the one of the program that calls this field operator. Just use
    #  the hard-coded lowering until this is cleaned up.
    def __gt_itir__(self) -> itir.FunctionDefinition:
        return foast_to_gtir.foast_to_gtir(self.foast_stage)

    # FIXME[#1582](tehrengruber): remove after refactoring to GTIR
    def __gt_gtir__(self) -> itir.FunctionDefinition:
        return foast_to_gtir.foast_to_gtir(self.foast_stage)

    def __gt_closure_vars__(self) -> dict[str, Any]:
        return self.foast_stage.closure_vars

    def __call__(self, *args: Any, enable_jit: bool | None = None, **kwargs: Any) -> Any:
        if not next_embedded.context.within_valid_context() and self.backend is not None:
            # non embedded execution
            offset_provider = {**kwargs.pop("offset_provider", {})}
            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            out = kwargs.pop("out")
            if "domain" in kwargs:
                domain = utils.tree_map(common.domain)(kwargs.pop("domain"))
                if not isinstance(domain, tuple):
                    domain = utils.tree_map(lambda _: domain)(out)
                out = utils.tree_map(lambda f, dom: f[dom])(out, domain)

            return self._compiled_programs(
                *args,
                **kwargs,
                out=out,
                offset_provider=offset_provider,
                enable_jit=self.compilation_options.enable_jit
                if enable_jit is None
                else enable_jit,
            )
        else:
            if not next_embedded.context.within_valid_context():
                # field_operator as program
                kwargs["offset_provider"] = {**kwargs.pop("offset_provider", {})}
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
                op: embedded_operators.EmbeddedOperator = embedded_operators.ScanOperator(
                    self.definition_stage.definition, forward, init, axis
                )
            else:
                op = embedded_operators.EmbeddedOperator(self.definition_stage.definition)
            return embedded_operators.field_operator_call(op, args, kwargs)


GTEntryPoint: TypeAlias = Program | FieldOperator


# TODO(tehrengruber): This class does not follow the Liskov-Substitution principle as it doesn't
#  have a field operator definition. Currently implementation is merely a hack to keep the only
#  test relying on this working. Revisit.
@dataclasses.dataclass(frozen=True)
class FieldOperatorFromFoast(FieldOperator):
    """
    This version of the field operator does not have a DSL definition.

    FieldOperator AST nodes can be programmatically built, which may be
    particularly useful in testing and debugging.
    This class provides the appropriate toolchain entry points.
    """

    foast_stage: ffront_stages.FOASTOperatorDef

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.backend is not None
        compiled_fo = self.backend.compile(
            self.foast_stage, arguments.CompileTimeArgs.from_concrete(*args, **kwargs)
        )
        return compiled_fo(*args, **kwargs)


@typing.overload
def field_operator(
    definition: types.FunctionType,
    *,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
) -> FieldOperator: ...


@typing.overload
def field_operator(
    *, backend: next_backend.Backend | eve.NothingType | None, grid_type: common.GridType | None
) -> Callable[[types.FunctionType], FieldOperator]: ...


def field_operator(
    definition: types.FunctionType | None = None,
    *,
    backend: next_backend.Backend | eve.NothingType | None = eve.NOTHING,
    grid_type: common.GridType | None = None,
    **compilation_options: Unpack[options.CompilationOptionsArgs],
) -> FieldOperator | Callable[[types.FunctionType], FieldOperator]:
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

    def field_operator_inner(definition: types.FunctionType) -> FieldOperator:
        return FieldOperator.from_function(
            definition,
            typing.cast(
                next_backend.Backend | None, DEFAULT_BACKEND if backend is eve.NOTHING else backend
            ),
            grid_type,
            **compilation_options,
        )

    return field_operator_inner if definition is None else field_operator_inner(definition)


@typing.overload
def scan_operator(
    definition: types.FunctionType,
    *,
    axis: common.Dimension,
    forward: bool,
    init: core_defs.Scalar,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
) -> FieldOperator: ...


@typing.overload
def scan_operator(
    *,
    axis: common.Dimension,
    forward: bool,
    init: core_defs.Scalar,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
) -> Callable[[types.FunctionType], FieldOperator]: ...


def scan_operator(
    definition: Optional[types.FunctionType] = None,
    *,
    axis: common.Dimension,
    forward: bool = True,
    init: core_defs.Scalar = 0.0,
    backend: next_backend.Backend | None | eve.NothingType = eve.NOTHING,
    grid_type: common.GridType | None = None,
) -> FieldOperator | Callable[[types.FunctionType], FieldOperator]:
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
            typing.cast(
                next_backend.Backend | None, DEFAULT_BACKEND if backend is eve.NOTHING else backend
            ),
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
