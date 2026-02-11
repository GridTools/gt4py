# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import functools
import itertools
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping, utils as eve_utils
from gt4py.next import backend as gtx_backend, common, config, errors, utils as gtx_utils
from gt4py.next.ffront import (
    stages as ffront_stages,
    type_info as ffront_type_info,
    type_specifications as ts_ffront,
    type_translation,
)
from gt4py.next.instrumentation import hook_machinery, metrics
from gt4py.next.otf import arguments, stages
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.utils import tree_map


T = TypeVar("T")

ScalarOrTupleOfScalars: TypeAlias = xtyping.MaybeNestedInTuple[core_defs.Scalar]

#: Content of the key: (*hashable_arg_descriptors, id(offset_provider), concrete_instantation_if_generic)
CompiledProgramsKey: TypeAlias = tuple[tuple[Hashable, ...], int, None | str]

ArgStaticDescriptorsByType: TypeAlias = dict[
    type[arguments.ArgStaticDescriptor], dict[str, arguments.ArgStaticDescriptor]
]


def _make_pool_root(
    program_definition: ffront_stages.DSLDefinition, backend: gtx_backend.Backend
) -> tuple[str, str]:
    return (program_definition.definition.__name__, backend.name)


@functools.cache
def _metrics_prefix_from_pool_root(root: tuple[str, str]) -> str:
    """Generate a metrics prefix from a compiled programs pool root."""
    return f"{root[0]}<{root[1]}>"


@hook_machinery.event_hook
def compile_variant_hook(
    program_definition: ffront_stages.DSLDefinition,
    backend: gtx_backend.Backend,
    offset_provider: common.OffsetProviderType | common.OffsetProvider,
    argument_descriptors: ArgStaticDescriptorsByType,
    key: CompiledProgramsKey,
) -> None:
    """Callback hook invoked before compiling a program variant."""

    if metrics.is_any_level_enabled():
        # Create a new metrics entity for this compiled program variant and
        # attach relevant metadata to it.
        source_key = f"{_metrics_prefix_from_pool_root(_make_pool_root(program_definition, backend))}[{hash(key)}]"
        assert source_key not in metrics.sources, (
            "The key for the program variant being compiled is already set!!"
        )

        metrics.sources[source_key].metadata |= dict(
            name=program_definition.definition.__name__,
            backend=backend.name,
            compiled_program_pool_key=hash(key),
            **{
                f"{eve_utils.CaseStyleConverter.convert(key.__name__, 'pascal', 'snake')}s": value
                for key, value in argument_descriptors.items()
            },
        )


@hook_machinery.context_hook
def compiled_program_call_context(
    compiled_program: stages.CompiledProgram,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    offset_provider: common.OffsetProvider,
    root: tuple[str, str],
    key: CompiledProgramsKey,
) -> contextlib.AbstractContextManager:
    """
    Hook called at the beginning and end of a compiled program call.

    Args:
        compiled_program: The compiled program being called.
        args: The arguments with which the program is called.
        kwargs: The keyword arguments with which the program is called.
        offset_provider: The offset provider passed to the program.
        root: The root of the compiled programs pool this program belongs to, i.e. a tuple of
            (program name, backend name).
        key: The key of the compiled program in the compiled programs pool.

    """
    # We set the metrics key for the compiled program call at enter and leave it
    # set at exit, since it is needed in the exit of the outer `program_call` context,
    # which will take care of unsetting it. This is because the compiled program call
    # is part of a program call, but we want the metrics to be associated with a
    # specific compiled program variant, not just the generic outer program.
    return metrics.metrics_setter_at_enter(f"{_metrics_prefix_from_pool_root(root)}[{hash(key)}]")


# TODO(havogt): We would like this to be a ProcessPoolExecutor, which requires (to decide what) to pickle.
_async_compilation_pool: concurrent.futures.Executor | None = None


def _init_async_compilation_pool() -> None:
    global _async_compilation_pool
    if _async_compilation_pool is None and config.BUILD_JOBS > 0:
        _async_compilation_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.BUILD_JOBS
        )


_init_async_compilation_pool()


def wait_for_compilation() -> None:
    """
    Waits for all ongoing compilations to finish.

    This is useful to ensure that all compiled programs are ready before
    proceeding with further operations. E.g. when the first call is included in timings.
    """
    global _async_compilation_pool
    if _async_compilation_pool is not None:
        _async_compilation_pool.shutdown(wait=True)
        _async_compilation_pool = None
        _init_async_compilation_pool()


def _make_tuple_expr(el_exprs: list[str]) -> str:
    return "".join((f"{el},") for el in el_exprs)


def _make_param_context_from_func_type(
    func_type: ts.FunctionType,
    type_map: Callable[[ts.TypeSpec], T] = lambda x: x,  # type: ignore[assignment, return-value]  # mypy not smart enough to narrow type for default
) -> dict[str, xtyping.MaybeNestedInTuple[T]]:
    """
    Create a context to evaluate expressions in from a function type.

    >>> int32_t, int64_t = (
    ...     ts.ScalarType(kind=ts.ScalarKind.INT32),
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ... )
    >>> type_ = ts.FunctionType(
    ...     pos_only_args=[],
    ...     pos_or_kw_args={"inp1": ts.TupleType(types=[int32_t, int64_t])},
    ...     kw_only_args={"inp2": int64_t},
    ...     returns=int64_t,
    ... )
    >>> context = _make_param_context_from_func_type(type_)
    >>> assert context == {"inp1": (int32_t, int64_t), "inp2": int64_t}
    """
    params = func_type.pos_or_kw_args | func_type.kw_only_args
    return {
        param: type_info.apply_to_primitive_constituents(
            type_map, type_, tuple_constructor=lambda *els: tuple(els)
        )
        for param, type_ in params.items()
    }


def _get_type_of_param_expr(program_type: ts_ffront.ProgramType, expr: str) -> ts.TypeSpec:
    structured_type_ = eval(expr, _make_param_context_from_func_type(program_type.definition))
    type_ = tree_map(
        lambda v: v, result_collection_constructor=lambda _, elts: ts.TupleType(types=list(elts))
    )(structured_type_)
    assert isinstance(type_, ts.TypeSpec)
    return type_


def _make_argument_descriptors(
    program_type: ts_ffront.ProgramType,
    argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]],
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> ArgStaticDescriptorsByType:
    """Given a set of runtime arguments construct all argument descriptors from them."""
    func_type = program_type.definition
    params = list(func_type.pos_or_kw_args.keys()) + list(func_type.kw_only_args.keys())
    descriptors: ArgStaticDescriptorsByType = {}
    for descriptor_cls, exprs in argument_descriptor_mapping.items():
        descriptors[descriptor_cls] = {}
        for expr in exprs:
            argument = eval(f"""lambda {",".join(params)}: {expr}""")(*args, **kwargs)
            descriptors[descriptor_cls][expr] = descriptor_cls.from_value(argument)
    _validate_argument_descriptors(program_type, descriptors)
    return descriptors


def _convert_to_argument_descriptor_context(
    func_type: ts.FunctionType, argument_descriptors: ArgStaticDescriptorsByType
) -> arguments.ArgStaticDescriptorsContextsByType:
    """
    Given argument descriptors, i.e., a mapping from an expr to a descriptor, transform them into a
    context of argument descriptors in which we can evaluate expressions.

    >>> int32_t, int64_t = (
    ...     ts.ScalarType(kind=ts.ScalarKind.INT32),
    ...     ts.ScalarType(kind=ts.ScalarKind.INT64),
    ... )
    >>> type_ = ts.FunctionType(
    ...     pos_only_args=[],
    ...     pos_or_kw_args={"inp1": ts.TupleType(types=[int32_t, int64_t])},
    ...     kw_only_args={"inp2": int64_t},
    ...     returns=int64_t,
    ... )
    >>> argument_descriptors = {arguments.StaticArg: {"inp1[1]": arguments.StaticArg(value=1)}}
    >>> contexts = _convert_to_argument_descriptor_context(type_, argument_descriptors)
    >>> contexts[arguments.StaticArg]
    {'inp1': (None, StaticArg(value=1)), 'inp2': None}
    """
    descriptor_contexts: arguments.ArgStaticDescriptorsContextsByType = {}
    for descriptor_cls, descriptor_expr_mapping in argument_descriptors.items():
        context: arguments.ArgStaticDescriptorsContext = _make_param_context_from_func_type(
            func_type, lambda x: None
        )
        # convert tuples to list such that we can alter the context easily
        context = {
            k: gtx_utils.tree_map(
                lambda v: v,
                collection_type=tuple,
                result_collection_constructor=lambda _, elts: list(elts),
            )(v)
            for k, v in context.items()
        }
        assert "__descriptor" not in context
        for expr, descriptor in descriptor_expr_mapping.items():
            # note: we don't need to handle any errors here since the `expr` has been validated
            #  in `_validate_argument_descriptor_mapping`
            exec(
                f"{expr} = __descriptor",
                {"__descriptor": descriptor},
                context,
            )
        # convert lists back to tuples
        context = {
            k: gtx_utils.tree_map(
                lambda v: v,
                collection_type=list,
                result_collection_constructor=lambda _, elts: tuple(elts),
            )(v)
            for k, v in context.items()
        }
        descriptor_contexts[descriptor_cls] = context  # type: ignore[index]  # Hard to understand, it looks like a mypy bug

    return descriptor_contexts


def _validate_argument_descriptors(
    program_type: ts_ffront.ProgramType, all_descriptors: ArgStaticDescriptorsByType
) -> None:
    for descriptors in all_descriptors.values():
        for expr, descriptor in descriptors.items():
            param_type = _get_type_of_param_expr(program_type, expr)
            descriptor.validate(expr, param_type)


@dataclasses.dataclass
class CompiledProgramsPool(Generic[ffront_stages.DSLDefinitionT]):
    """
    A pool of compiled programs for a given program and backend.

    If 'argument_descriptor_mapping' is populated the pool will create a program for each
    argument that has an argument descriptor. E.g., if a param is marked static we create
    a new program for each value of that parameter. See :ref:`arguments.ArgumentDescriptor` for
    more information on argument descriptors.

    If `enable_jit` is True in the call to the pool, it will compile a program
    with static information as described in `argument_descriptor_mapping`, otherwise it
    will error. In the latter case, the pool needs to be filled with call(s)
    to `compile` before it can be used.
    """

    backend: gtx_backend.Backend
    definition_stage: ffront_stages.DSLDefinitionT
    # Note: This type can be incomplete, i.e. contain DeferredType, whenever the operator is a
    #  scan operator. In the future it could also be the type of a generic program.
    program_type: ts_ffront.ProgramType
    #: mapping from an argument descriptor type to a list of parameters or expression thereof
    #: e.g. `{arguments.StaticArg: ["static_int_param"]}`
    #: Note: The list is not ordered.
    argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]] | None

    # store for the compiled programs
    compiled_programs: dict[CompiledProgramsKey, stages.CompiledProgram] = dataclasses.field(
        default_factory=dict, init=False
    )

    # store for the async compilation jobs
    _compilation_jobs: dict[
        CompiledProgramsKey, concurrent.futures.Future[stages.CompiledProgram]
    ] = dataclasses.field(default_factory=dict, init=False)

    @functools.cached_property
    def root(self) -> tuple[str, str]:
        return _make_pool_root(self.definition_stage, self.backend)

    def __post_init__(self) -> None:
        # TODO(havogt): We currently don't support pos_only or kw_only args at the program level.
        # This check makes sure we don't miss updating this code if we add support for them in the future.
        assert not self.program_type.definition.kw_only_args
        assert not self.program_type.definition.pos_only_args
        self._validate_argument_descriptor_mapping()

        # Force initialization of all cached properties here to minimize first-time call overhead
        self._primitive_values_extractor  # noqa: B018

    def __call__(
        self, *args: Any, offset_provider: common.OffsetProvider, enable_jit: bool, **kwargs: Any
    ) -> None:
        """
        Calls a program with the given arguments and offset provider.

        If the program is not in cache, it will jit compile with static arguments
        (defined by 'static_params') in case `enable_jit` is True. Otherwise,
        it is an error.
        """
        canonical_args, canonical_kwargs = self._args_canonicalizer(*args, **kwargs)
        if (extractor := self._primitive_values_extractor) is not None:
            args, kwargs = extractor(*canonical_args, **canonical_kwargs)
        else:
            args, kwargs = canonical_args, canonical_kwargs
        static_args_values = self._argument_descriptor_cache_key_from_args(*args, **kwargs)

        if self._is_generic:
            # In case the program or operator is generic, i.e. callable for arguments of varying
            # type, add the argument types to the cache key as the argument types are used during
            # compilation. In case the program is not generic we can avoid the potentially
            # expensive type deduction for all arguments and not include it in the key.
            warnings.warn(
                "Calling generic programs / direct calls to scan operators are not optimized. "
                "Consider calling a specialized version instead.",
                stacklevel=2,
            )
            arg_specialization_key = eve_utils.content_hash(
                (
                    tuple(type_translation.from_value(arg) for arg in canonical_args),
                    {k: type_translation.from_value(v) for k, v in canonical_kwargs.items()},
                )
            )
        else:
            arg_specialization_key = None

        key = (
            static_args_values,
            common.hash_offset_provider_items_by_id(offset_provider),
            arg_specialization_key,
        )

        try:
            compiled_program = self.compiled_programs[key]

        except KeyError as e:
            if self._finish_compilation_job(key):
                compiled_program = self.compiled_programs[key]
            elif enable_jit:
                assert self.argument_descriptor_mapping is not None
                self._compile_variant(
                    argument_descriptors=_make_argument_descriptors(
                        self.program_type, self.argument_descriptor_mapping, args, kwargs
                    ),
                    # note: it is important to use the args before named collections are extracted
                    #  as otherwise the implicit program generation from an operator fails
                    arg_specialization_info=(
                        tuple(type_translation.from_value(arg) for arg in canonical_args),
                        {k: type_translation.from_value(v) for k, v in canonical_kwargs.items()},
                    ),
                    offset_provider=offset_provider,
                    call_key=key,
                )
                return self(
                    *canonical_args,
                    offset_provider=offset_provider,
                    enable_jit=False,
                    **canonical_kwargs,
                )  # passing `enable_jit=False` because a cache miss should be a hard-error in this call`

            else:
                raise RuntimeError("No program compiled for this set of static arguments.") from e

        with compiled_program_call_context(
            compiled_program, args, kwargs, offset_provider, self.root, key
        ):
            compiled_program(*args, **kwargs, offset_provider=offset_provider)

    @functools.cached_property
    def _primitive_values_extractor(self) -> Callable | None:
        return arguments.make_primitive_value_args_extractor(self.program_type.definition)

    @functools.cached_property
    def _is_generic(self) -> bool:
        """
        Is the operator or program generic in the sense that it can be called for different
        argument types.

        Right now this is only the case for scan operators.
        """
        # TODO(tehrengruber): This concept does not exist elsewhere and is not properly reflected
        #  in the type system. For now we just use `DeferredType` to communicate between
        #  here and `type_info.type_in_program_context`.
        return any(
            isinstance(t, ts.DeferredType)
            for t in itertools.chain(
                self.program_type.definition.pos_only_args,
                self.program_type.definition.pos_or_kw_args.values(),
                self.program_type.definition.kw_only_args.values(),
            )
        )

    @functools.cached_property
    def _args_canonicalizer(self) -> Callable[..., tuple[tuple, dict[str, Any]]]:
        return ffront_type_info.make_args_canonicalizer(
            self.program_type, name=self.definition_stage.definition.__name__
        )

    @functools.cached_property
    def _argument_descriptor_cache_key_from_args(
        self,
    ) -> Callable[..., tuple[Hashable, ...]]:
        """
        Given the entire set of runtime arguments compute the cache key used to retrieve the
        instance of the compiled program which is compiled for the argument descriptors from
        the given set of arguments.

        This is part of the performance critical path that is called on every program call,
        hence we code generate a single lambda expression here.
        """
        func_type = self.program_type.definition
        params = list(func_type.pos_or_kw_args.keys()) + list(func_type.kw_only_args.keys())
        elements: list[str] = []
        for descriptor_cls, arg_exprs in self.argument_descriptor_mapping.items():  # type: ignore[union-attr]  # can never be `None` at this point
            for arg_expr in arg_exprs:
                attr_extractor = descriptor_cls.attribute_extractor_exprs(arg_expr)
                elements.extend(attr_extractor.values())
        return eval(f"""lambda {",".join(params)}: ({_make_tuple_expr(elements)})""")

    def _argument_descriptor_cache_key_from_descriptors(
        self,
        argument_descriptor_contexts: arguments.ArgStaticDescriptorsContextsByType,
    ) -> tuple:
        """
        Given a set of argument descriptors deduce the cache key used to retrieve the instance
        of the compiled program which is compiled for the given argument descriptors.

        This function is not performance critical as it is only called once when compiling a
        variant.
        """
        elements = []
        for descriptor_cls, arg_exprs in self.argument_descriptor_mapping.items():  # type: ignore[union-attr]  # can never be `None` at this point
            for arg_expr in arg_exprs:
                attr_extractor = descriptor_cls.attribute_extractor_exprs(arg_expr)
                attrs = attr_extractor.keys()
                for attr in attrs:
                    elements.append(
                        getattr(
                            eval(f"{arg_expr}", {}, argument_descriptor_contexts[descriptor_cls]),
                            attr,
                        )
                    )
        return tuple(elements)

    def _initialize_argument_descriptor_mapping(
        self, argument_descriptors: ArgStaticDescriptorsByType
    ) -> None:
        if self.argument_descriptor_mapping is None:
            self.argument_descriptor_mapping = {
                descr_cls: list(descriptor_expr_mapping.keys())
                for descr_cls, descriptor_expr_mapping in argument_descriptors.items()
            }
            self._validate_argument_descriptor_mapping()
        else:
            for descr_cls, descriptor_expr_mapping in argument_descriptors.items():
                if (expected := set(self.argument_descriptor_mapping[descr_cls])) != (
                    got := set(descriptor_expr_mapping.keys())
                ):
                    raise ValueError(
                        f"Argument descriptor {descr_cls.__name__} must be the same for all compiled programs, got {list(got)} expected {list(expected)}."
                    )

    def _validate_argument_descriptor_mapping(self) -> None:
        if self.argument_descriptor_mapping is None:
            return
        context = _make_param_context_from_func_type(self.program_type.definition, lambda x: None)
        for descr_cls, exprs in self.argument_descriptor_mapping.items():
            for expr in exprs:
                try:
                    # TODO(tehrengruber): Re-evaluate the way we validate here when we add support
                    #  for containers.
                    if any(
                        v is not None for v in gtx_utils.flatten_nested_tuple(eval(expr, context))
                    ):
                        raise ValueError()
                except (ValueError, KeyError, NameError):
                    raise errors.DSLTypeError(  # noqa: B904 # we don't care about the original exception
                        message=f"Invalid parameter expression '{expr}' for '{descr_cls.__name__}'. "
                        f"Must be the name of a parameter or an access to one of its elements.",
                        location=None,
                    )

    def _is_existing_key(self, key: CompiledProgramsKey) -> bool:
        return key in self.compiled_programs or key in self._compilation_jobs

    def _finish_compilation_job(self, key: CompiledProgramsKey) -> bool:
        if key not in self._compilation_jobs:
            return False

        compiled_program_future = self._compilation_jobs.pop(key)
        assert isinstance(compiled_program_future, concurrent.futures.Future)
        assert key not in self.compiled_programs
        self.compiled_programs[key] = compiled_program_future.result()
        return True

    def _compile_variant(
        self,
        argument_descriptors: ArgStaticDescriptorsByType,
        offset_provider: common.OffsetProviderType | common.OffsetProvider,
        #: tuple consisting of the types of the positional and keyword arguments.
        arg_specialization_info: tuple[tuple[ts.TypeSpec, ...], dict[str, ts.TypeSpec]]
        | None = None,
        # argument used only to validate key computed in a call / dispatch agrees with the
        # key computed here
        call_key: CompiledProgramsKey | None = None,
    ) -> None:
        if not common.is_offset_provider(offset_provider):
            if common.is_offset_provider_type(offset_provider):
                raise ValueError(
                    "Variant compilation of programs with 'OffsetProviderType' is not yet supported."
                )
            else:
                raise ValueError(f"Invalid 'offset_provider': {offset_provider}")

        self._initialize_argument_descriptor_mapping(argument_descriptors)
        _validate_argument_descriptors(self.program_type, argument_descriptors)

        argument_descriptor_contexts = _convert_to_argument_descriptor_context(
            self.program_type.definition, argument_descriptors
        )
        key = (
            self._argument_descriptor_cache_key_from_descriptors(argument_descriptor_contexts),
            common.hash_offset_provider_items_by_id(offset_provider),
            eve_utils.content_hash(arg_specialization_info) if self._is_generic else None,
        )
        assert call_key is None or call_key == key

        if self._is_existing_key(key):
            raise ValueError(f"Program with key {key} already exists.")

        if arg_specialization_info:
            arg_types, kwarg_types = arg_specialization_info
        else:
            if self._is_generic:
                raise ValueError(
                    "Can not precompile generic program or scan operator without argument types."
                )
            arg_types = (
                *self.program_type.definition.pos_only_args,
                *self.program_type.definition.pos_or_kw_args.values(),
            )
            kwarg_types = self.program_type.definition.kw_only_args

        compile_time_args = arguments.CompileTimeArgs(
            offset_provider=offset_provider,
            column_axis=None,  # TODO(havogt): column_axis seems to a unused, even for programs with scans
            args=arg_types,
            kwargs=kwarg_types,
            argument_descriptor_contexts=argument_descriptor_contexts,
        )
        compile_call = functools.partial(
            self.backend.compile, self.definition_stage, compile_time_args=compile_time_args
        )
        compile_variant_hook(
            self.definition_stage,
            self.backend,
            offset_provider=offset_provider,
            argument_descriptors=argument_descriptors,
            key=key,
        )

        if _async_compilation_pool is None:
            self.compiled_programs[key] = compile_call()
        else:
            self._compilation_jobs[key] = _async_compilation_pool.submit(compile_call)

    # TODO(tehrengruber): Rework the interface to allow precompilation with compile time
    #  domains and of scans.
    def compile(
        self,
        offset_providers: list[common.OffsetProvider | common.OffsetProviderType],
        **static_args: list[ScalarOrTupleOfScalars],
    ) -> None:
        """
        Compiles the program for all combinations of static arguments and the given 'OffsetProviderType'.

        Note: In case you want to compile for specific combinations of static arguments (instead
        of the combinatoral), you can call compile multiples times.

        Examples:
            pool.compile(static_arg0=[0,1], static_arg1=[2,3], ...)
                will compile for (0,2), (0,3), (1,2), (1,3)
            pool.compile(static_arg0=[0], static_arg1=[2]).compile(static_arg=[1], static_arg1=[3])
                will compile for (0,2), (1,3)
        """
        for offset_provider in offset_providers:  # not included in product for better type checking
            for static_values in itertools.product(*static_args.values()):
                self._compile_variant(
                    argument_descriptors={
                        arguments.StaticArg: dict(
                            zip(
                                static_args.keys(),
                                [arguments.StaticArg(value=v) for v in static_values],
                                strict=True,
                            )
                        ),
                    },
                    offset_provider=offset_provider,
                )
