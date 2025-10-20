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

import dataclasses
import functools
import time
import types
import typing
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Generic, Optional, TypeVar

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.eve.extended_typing import Self, override
from gt4py.next import (
    allocators as next_allocators,
    backend as next_backend,
    common,
    config,
    embedded as next_embedded,
    errors,
    metrics,
    utils,
)
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.ffront import (
    field_operator_ast as foast,
    foast_to_gtir,
    past_process_args,
    signature,
    stages as ffront_stages,
    transform_utils,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import arguments, compiled_program, stages, toolchain
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


DEFAULT_BACKEND: next_backend.Backend | None = None


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
    connectivities: Optional[
        common.OffsetProvider
    ]  # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    enable_jit: bool | None
    static_params: (
        Sequence[str] | None
    )  # if the user requests static params, they will be used later to initialize CompiledPrograms

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: next_backend.Backend | None,
        grid_type: common.GridType | None = None,
        enable_jit: bool | None = None,
        static_params: Sequence[str] | None = None,
        connectivities: Optional[
            common.OffsetProvider
        ] = None,  # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    ) -> Program:
        program_def = ffront_stages.ProgramDefinition(definition=definition, grid_type=grid_type)
        return cls(
            definition_stage=program_def,
            backend=backend,
            connectivities=connectivities,
            enable_jit=enable_jit,
            static_params=static_params,
        )

    # needed in testing
    @property
    def definition(self) -> types.FunctionType:
        return self.definition_stage.definition

    @functools.cached_property
    def past_stage(self) -> ffront_stages.PRG:
        # backwards compatibility for backends that do not support the full toolchain
        no_args_def = toolchain.CompilableProgram(
            self.definition_stage, arguments.CompileTimeArgs.empty()
        )
        return self._frontend_transforms.func_to_past(no_args_def).data

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self) -> None:
        no_args_past = toolchain.CompilableProgram(
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
    def _frontend_transforms(self) -> next_backend.Transforms:
        if self.backend is None:
            return next_backend.DEFAULT_TRANSFORMS
        # TODO(tehrengruber): This class relies heavily on `self.backend.transforms` being
        #  a `next_backend.Transforms`, but the backend type annotation does not reflect that.
        assert isinstance(self.backend.transforms, next_backend.Transforms)
        return self.backend.transforms

    def with_backend(self, backend: next_backend.Backend) -> Program:
        return dataclasses.replace(self, backend=backend)

    def with_connectivities(
        self,
        connectivities: common.OffsetProvider,  # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime informatio
    ) -> Program:
        return dataclasses.replace(self, connectivities=connectivities)

    def with_grid_type(self, grid_type: common.GridType) -> Program:
        return dataclasses.replace(
            self, definition_stage=dataclasses.replace(self.definition_stage, grid_type=grid_type)
        )

    def with_static_params(self, *static_params: str | None) -> Program:
        if not static_params or (static_params == (None,)):
            _static_params = None
        else:
            assert all(p is not None for p in static_params)
            _static_params = typing.cast(tuple[str], static_params)
        return dataclasses.replace(
            self,
            static_params=_static_params,
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

    @functools.cached_property
    def _all_closure_vars(self) -> dict[str, Any]:
        return transform_utils._get_closure_vars_recursively(self.past_stage.closure_vars)

    @functools.cached_property
    def gtir(self) -> itir.Program:
        no_args_past = toolchain.CompilableProgram(
            data=ffront_stages.PastProgramDefinition(
                past_node=self.past_stage.past_node,
                closure_vars=self.past_stage.closure_vars,
                grid_type=self.definition_stage.grid_type,
            ),
            args=arguments.CompileTimeArgs.empty(),
        )
        return self._frontend_transforms.past_to_itir(no_args_past).data

    @functools.cached_property
    def _compiled_programs(self) -> compiled_program.CompiledProgramsPool:
        if self.backend is None or self.backend == eve.NOTHING:
            raise RuntimeError("Cannot compile a program without backend.")

        if self.static_params is None:
            object.__setattr__(self, "static_params", ())

        argument_descriptor_mapping = {
            arguments.StaticArg: self.static_params,
        }

        program_type = self.past_stage.past_node.type
        assert isinstance(program_type, ts_ffront.ProgramType)
        return compiled_program.CompiledProgramsPool(
            backend=self.backend,
            definition_stage=self.definition_stage,
            program_type=program_type,
            argument_descriptor_mapping=argument_descriptor_mapping,  # type: ignore[arg-type]  # covariant `type[T]` not possible
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
        if enable_jit is None:
            enable_jit = (
                self.enable_jit if self.enable_jit is not None else config.ENABLE_JIT_DEFAULT
            )

        with metrics.collect() as metrics_source:
            if collect_info_metrics := (config.COLLECT_METRICS_LEVEL >= metrics.INFO):
                start = time.time()

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
                # Metrics source key needs to be setup here, since embedded programs
                # don't have variants and thus there's no other place we could do this.
                if config.COLLECT_METRICS_LEVEL:
                    assert metrics_source is not None
                    metrics_source.key = (
                        f"{self.__name__}<{getattr(self.backend, 'name', '<embedded>')}>"
                    )
                warnings.warn(
                    UserWarning(
                        f"Field View Program '{self.definition_stage.definition.__name__}': Using Python execution, consider selecting a performance backend."
                    ),
                    stacklevel=2,
                )
                with next_embedded.context.update(offset_provider=offset_provider):
                    self.definition_stage.definition(*args, **kwargs)

            if collect_info_metrics:
                assert metrics_source is not None
                metrics_source.metrics[metrics.TOTAL_METRIC].add_sample(time.time() - start)

    def compile(
        self,
        offset_provider: common.OffsetProviderType
        | common.OffsetProvider
        | list[common.OffsetProviderType | common.OffsetProvider]
        | None = None,
        enable_jit: bool | None = None,
        **static_args: list[xtyping.MaybeNestedInTuple[core_defs.Scalar]],
    ) -> Self:
        """
        Compiles the program for the given combination of static arguments and offset provider type.

        Note: Unlike `with_...` methods, this method does not return a new instance of the program,
        but adds the compiled variants to the current program instance.
        """
        # TODO(havogt): we should reconsider if we want to return a new program on `compile` (and
        # rename to `with_static_args` or similar) once we have a better understanding of the
        # use-cases.

        if enable_jit is not None:
            object.__setattr__(self, "enable_jit", enable_jit)
        if self.static_params is None:
            object.__setattr__(self, "static_params", tuple(static_args.keys()))
        if self.connectivities is None and offset_provider is None:
            raise ValueError(
                "Cannot compile a program without connectivities / OffsetProviderType."
            )
        if not all(isinstance(v, list) for v in static_args.values()):
            raise TypeError(
                "Please provide the static arguments as lists."
            )  # To avoid confusion with tuple args

        offset_provider = self.connectivities if offset_provider is None else offset_provider
        if not isinstance(offset_provider, list):
            offset_provider = [offset_provider]  # type: ignore[list-item] # cleanup offset_provider vs offset_provider_type

        assert all(
            common.is_offset_provider(op) or common.is_offset_provider_type(op)
            for op in offset_provider
        )

        self._compiled_programs.compile(offset_providers=offset_provider, **static_args)
        return self

    def freeze(self) -> FrozenProgram:
        if self.backend is None:
            raise ValueError("Can not freeze a program without backend (embedded execution).")
        return FrozenProgram(
            self.definition_stage if self.definition_stage else self.past_stage,
            backend=self.backend,
        )


@dataclasses.dataclass(frozen=True)
class FrozenProgram:
    """
    Simplified program instance, which skips the whole toolchain after the first execution.

    Does not work in embedded execution.
    """

    program: ffront_stages.DSL_PRG | ffront_stages.PRG
    backend: next_backend.Backend
    _compiled_program: Optional[stages.CompiledProgram] = dataclasses.field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        if self.backend is None:
            raise ValueError("Can not JIT-compile programs without backend (embedded execution).")

    @property
    def definition(self) -> types.FunctionType:
        # `PastProgramDefinition` doesn't have `definition`
        assert isinstance(self.program, ffront_stages.ProgramDefinition)
        return self.program.definition

    def with_backend(self, backend: next_backend.Backend) -> FrozenProgram:
        return self.__class__(program=self.program, backend=backend)

    def with_grid_type(self, grid_type: common.GridType) -> FrozenProgram:
        return self.__class__(
            program=dataclasses.replace(self.program, grid_type=grid_type), backend=self.backend
        )

    def jit(
        self, *args: Any, offset_provider: common.OffsetProvider, **kwargs: Any
    ) -> stages.CompiledProgram:
        return self.backend.jit(self.program, *args, offset_provider=offset_provider, **kwargs)

    def __call__(self, *args: Any, offset_provider: common.OffsetProvider, **kwargs: Any) -> None:
        args, kwargs = signature.convert_to_positional(self.program, *args, **kwargs)

        if not self._compiled_program:
            super().__setattr__(
                "_compiled_program", self.jit(*args, offset_provider=offset_provider, **kwargs)
            )
        self._compiled_program(*args, offset_provider=offset_provider, **kwargs)  # type: ignore[misc] # _compiled_program is not None


try:
    from gt4py.next.program_processors.runners.dace.program import (  # type: ignore[assignment]
        Program,
    )
except ImportError:
    pass


# TODO(tehrengruber): This class does not follow the Liskov-Substitution principle as it doesn't
#  have a program definition. Revisit.
@dataclasses.dataclass(frozen=True)
class ProgramFromPast(Program):
    """
    This version of program has no DSL definition associated with it.

    PAST nodes can be built programmatically from field operators or from scratch.
    This wrapper provides the appropriate toolchain entry points.
    """

    past_stage: ffront_stages.PastProgramDefinition

    @override
    def __call__(
        self, *args: Any, offset_provider: Optional[common.OffsetProvider] = None, **kwargs: Any
    ) -> None:
        if self.backend is None:
            raise NotImplementedError(
                "Programs created from a PAST node (without a function definition) can not be executed in embedded mode"
            )

        if offset_provider is None:
            offset_provider = {}
        # TODO(ricoh): add test that does the equivalent of IDim + 1 in a ProgramFromPast
        self.backend(
            self.past_stage,
            *args,
            **(kwargs | {"offset_provider": {**offset_provider}}),
        )

    # TODO(ricoh): linting should become optional, up to the backend.
    def __post_init__(self) -> None:
        self._frontend_transforms.past_lint(self.past_stage)  # type: ignore[arg-type] # ignored because the class has more TODO than code


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
        enable_jit: bool | None = None,
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
    enable_jit: bool | None,
    static_params: Sequence[str] | None,
    frozen: bool,
) -> Callable[[types.FunctionType], Program]: ...


def program(
    definition: types.FunctionType | None = None,
    *,
    # `NOTHING` -> default backend, `None` -> no backend (embedded execution)
    backend: next_backend.Backend | eve.NothingType | None = eve.NOTHING,
    grid_type: common.GridType | None = None,
    enable_jit: bool | None = None,  # only relevant if static_params are set
    static_params: Sequence[str] | None = None,
    frozen: bool = False,
) -> Program | FrozenProgram | Callable[[types.FunctionType], Program | FrozenProgram]:
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
            enable_jit=enable_jit,
            static_params=static_params,
        )
        if frozen:
            return program.freeze()  # type: ignore[return-value] # TODO(havogt): Should `FrozenProgram` be a `Program`?
        return program

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
    backend: Optional[next_backend.Backend]
    _program_cache: dict = dataclasses.field(
        init=False, default_factory=dict
    )  # init=False ensure the cache is not copied in calls to replace

    @classmethod
    def from_function(
        cls,
        definition: types.FunctionType,
        backend: Optional[next_backend.Backend],
        grid_type: Optional[common.GridType] = None,
        *,
        operator_node_cls: type[OperatorNodeT] = foast.FieldOperator,  # type: ignore[assignment] # TODO(ricoh): understand why mypy complains
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
    def __post_init__(self) -> None:
        """This ensures that DSL linting occurs at decoration time."""
        _ = self.foast_stage

    @functools.cached_property
    def foast_stage(self) -> ffront_stages.FoastOperatorDefinition:
        return self._frontend_transforms.func_to_foast(
            toolchain.CompilableProgram(
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

    def with_backend(self, backend: next_backend.Backend) -> FieldOperator:
        return dataclasses.replace(self, backend=backend)

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

    def as_program(self, compiletime_args: arguments.CompileTimeArgs) -> Program:
        foast_with_types = (
            toolchain.CompilableProgram(
                data=self.foast_stage,
                args=compiletime_args,
            ),
        )

        past_stage = self._frontend_transforms.field_view_op_to_prog.foast_to_past(  # type: ignore[attr-defined] # TODO(havogt): needs more work
            foast_with_types
        ).data
        return ProgramFromPast(
            definition_stage=None,  # type: ignore[arg-type] # ProgramFromPast needs to be fixed
            past_stage=past_stage,
            backend=self.backend,
            connectivities=None,
            enable_jit=False,  # TODO(havogt): revisit ProgramFromPast
            static_params=None,  # TODO(havogt): revisit ProgramFromPast
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not next_embedded.context.within_valid_context() and self.backend is not None:
            # non embedded execution
            offset_provider = {**kwargs.pop("offset_provider", {})}
            if "out" not in kwargs:
                raise errors.MissingArgumentError(None, "out", True)
            out = kwargs.pop("out")
            if "domain" in kwargs:
                domain = common.normalize_domains(kwargs.pop("domain"))
                if not isinstance(domain, tuple):
                    domain = utils.tree_map(lambda _: domain)(out)
                out = utils.tree_map(lambda f, dom: f[dom])(out, domain)

            args, kwargs = type_info.canonicalize_arguments(
                self.foast_stage.foast_node.type, args, kwargs
            )
            return self.backend(
                self.definition_stage,
                *args,
                out=out,
                offset_provider=offset_provider,
                **kwargs,
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


@dataclasses.dataclass(frozen=True)
class FieldOperatorFromFoast(FieldOperator):
    """
    This version of the field operator does not have a DSL definition.

    FieldOperator AST nodes can be programmatically built, which may be
    particularly useful in testing and debugging.
    This class provides the appropriate toolchain entry points.
    """

    foast_stage: ffront_stages.FoastOperatorDefinition

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        assert self.backend is not None
        return self.backend(self.foast_stage, *args, **kwargs)


@typing.overload
def field_operator(
    definition: types.FunctionType,
    *,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
) -> FieldOperator[foast.FieldOperator]: ...


@typing.overload
def field_operator(
    *, backend: next_backend.Backend | eve.NothingType | None, grid_type: common.GridType | None
) -> Callable[[types.FunctionType], FieldOperator[foast.FieldOperator]]: ...


def field_operator(
    definition: types.FunctionType | None = None,
    *,
    backend: next_backend.Backend | eve.NothingType | None = eve.NOTHING,
    grid_type: common.GridType | None = None,
) -> (
    FieldOperator[foast.FieldOperator]
    | Callable[[types.FunctionType], FieldOperator[foast.FieldOperator]]
):
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
            definition,
            typing.cast(
                next_backend.Backend | None, DEFAULT_BACKEND if backend is eve.NOTHING else backend
            ),
            grid_type,
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
) -> FieldOperator[foast.ScanOperator]: ...


@typing.overload
def scan_operator(
    *,
    axis: common.Dimension,
    forward: bool,
    init: core_defs.Scalar,
    backend: next_backend.Backend | eve.NothingType | None,
    grid_type: common.GridType | None,
) -> Callable[[types.FunctionType], FieldOperator[foast.ScanOperator]]: ...


def scan_operator(
    definition: Optional[types.FunctionType] = None,
    *,
    axis: common.Dimension,
    forward: bool = True,
    init: core_defs.Scalar = 0.0,
    backend: next_backend.Backend | None | eve.NothingType = eve.NOTHING,
    grid_type: common.GridType | None = None,
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


@ffront_stages.add_content_to_fingerprint.register
def add_past_program_to_fingerprint(obj: ProgramFromPast, hasher: xtyping.HashlibAlgorithm) -> None:
    ffront_stages.add_content_to_fingerprint(obj.past_stage, hasher)
    ffront_stages.add_content_to_fingerprint(obj.backend, hasher)
