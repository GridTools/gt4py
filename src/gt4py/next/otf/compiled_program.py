# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import itertools
from typing import Any, Callable, Sequence, TypeAlias, TypeVar

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing, utils as eve_utils
from gt4py.next import backend as gtx_backend, common, config, errors, utils as gtx_utils
from gt4py.next.ffront import stages as ffront_stages, type_specifications as ts_ffront
from gt4py.next.otf import arguments, stages
from gt4py.next.type_system import type_info, type_specifications as ts
from gt4py.next.utils import tree_map


T = TypeVar("T")

# TODO(havogt): We would like this to be a ProcessPoolExecutor, which requires (to decide what) to pickle.
_async_compilation_pool: concurrent.futures.Executor | None = None


def _init_async_compilation_pool() -> None:
    global _async_compilation_pool
    if _async_compilation_pool is None and config.BUILD_JOBS > 0:
        _async_compilation_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.BUILD_JOBS
        )


_init_async_compilation_pool()

ScalarOrTupleOfScalars: TypeAlias = extended_typing.MaybeNestedInTuple[core_defs.Scalar]
CompiledProgramsKey: TypeAlias = tuple[
    tuple[ScalarOrTupleOfScalars, ...], common.OffsetProviderType
]
ArgumentDescriptors: TypeAlias = dict[
    type[arguments.ArgStaticDescriptor], dict[str, arguments.ArgStaticDescriptor]
]
ArgumentDescriptorContext: TypeAlias = dict[
    str, extended_typing.MaybeNestedInTuple[arguments.ArgStaticDescriptor | None]
]
ArgumentDescriptorContexts: TypeAlias = dict[
    type[arguments.ArgStaticDescriptor],
    ArgumentDescriptorContext,
]


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


def _hash_compiled_program_unsafe(cp_key: CompiledProgramsKey) -> int:
    values, offset_provider = cp_key
    assert common.is_offset_provider_type(offset_provider)
    return hash((values, id(offset_provider)))


def _make_tuple_expr(el_exprs: list[str]) -> str:
    return "".join((f"{el},") for el in el_exprs)


def _make_param_context_from_func_type(
    func_type: ts.FunctionType,
    type_map: Callable[[ts.TypeSpec], T] = lambda x: x,  # type: ignore[assignment, return-value]  # mypy not smart enough to narrow type for default
) -> dict[str, extended_typing.MaybeNestedInTuple[T]]:
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
        lambda v: v, result_collection_constructor=lambda elts: ts.TupleType(types=list(elts))
    )(structured_type_)
    assert isinstance(type_, ts.TypeSpec)
    return type_


def _make_argument_descriptors(
    program_type: ts_ffront.ProgramType,
    argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]],
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> ArgumentDescriptors:
    """Given a set of runtime arguments construct all argument descriptors from them."""
    func_type = program_type.definition
    params = list(func_type.pos_or_kw_args.keys()) + list(func_type.kw_only_args.keys())
    descriptors: ArgumentDescriptors = {}
    for descriptor_cls, exprs in argument_descriptor_mapping.items():
        descriptors[descriptor_cls] = {}
        for expr in exprs:
            argument = eval(f"""lambda {",".join(params)}: {expr}""")(*args, **kwargs)
            descriptors[descriptor_cls][expr] = descriptor_cls.from_value(argument)
    _validate_argument_descriptors(program_type, descriptors)
    return descriptors


def _convert_to_argument_descriptor_context(
    func_type: ts.FunctionType, argument_descriptors: ArgumentDescriptors
) -> ArgumentDescriptorContexts:
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
    descriptor_contexts: ArgumentDescriptorContexts = {}
    for descriptor_cls, descriptor_expr_mapping in argument_descriptors.items():
        context: ArgumentDescriptorContext = _make_param_context_from_func_type(
            func_type, lambda x: None
        )
        # convert tuples to list such that we can alter the context easily
        context = {
            k: gtx_utils.tree_map(
                lambda v: v, collection_type=tuple, result_collection_constructor=list
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
                lambda v: v, collection_type=list, result_collection_constructor=tuple
            )(v)
            for k, v in context.items()
        }
        descriptor_contexts[descriptor_cls] = context

    return descriptor_contexts


def _validate_argument_descriptors(
    program_type: ts_ffront.ProgramType,
    all_descriptors: ArgumentDescriptors,
) -> None:
    for descriptors in all_descriptors.values():
        for expr, descriptor in descriptors.items():
            param_type = _get_type_of_param_expr(program_type, expr)
            descriptor.validate(expr, param_type)


@dataclasses.dataclass
class CompiledProgramsPool:
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
    definition_stage: ffront_stages.ProgramDefinition
    program_type: ts_ffront.ProgramType
    #: mapping from an argument descriptor type to a list of parameters or expression thereof
    #: e.g. `{arguments.StaticArg: ["static_int_param"]}`
    #: Note: The list is not ordered.
    argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]] | None

    _compiled_programs: eve_utils.CustomMapping = dataclasses.field(
        default_factory=lambda: eve_utils.CustomMapping(_hash_compiled_program_unsafe),
        init=False,
    )

    _offset_provider_type_cache: eve_utils.CustomMapping = dataclasses.field(
        default_factory=lambda: eve_utils.CustomMapping(common.hash_offset_provider_unsafe),
        init=False,
    )  # cache the offset provider type in order to avoid recomputing it at each program call

    def __post_init__(self) -> None:
        # TODO(havogt): We currently don't support pos_only or kw_only args at the program level.
        # This check makes sure we don't miss updating this code if we add support for them in the future.
        assert not self.program_type.definition.kw_only_args
        assert not self.program_type.definition.pos_only_args
        self._validate_argument_descriptor_mapping()

    def __call__(
        self, *args: Any, offset_provider: common.OffsetProvider, enable_jit: bool, **kwargs: Any
    ) -> None:
        """
        Calls a program with the given arguments and offset provider.

        If the program is not in cache, it will jit compile with static arguments
        (defined by 'static_params') in case `enable_jit` is True. Otherwise,
        it is an error.
        """
        args, kwargs = type_info.canonicalize_arguments(self.program_type, args, kwargs)
        static_args_values = self._argument_descriptor_cache_key_from_args(*args, **kwargs)
        # TODO(tehrengruber): Dispatching over offset provider type is wrong, especially when we
        #  use compile time domains.
        key = (static_args_values, self._offset_provider_to_type_unsafe(offset_provider))
        try:
            self._compiled_programs[key](*args, **kwargs, offset_provider=offset_provider)
        except TypeError:  # 'Future' object is not callable
            # ... otherwise we resolve the future and call again
            program = self._resolve_future(key)
            program(*args, **kwargs, offset_provider=offset_provider)
        except KeyError as e:
            if enable_jit:
                assert self.argument_descriptor_mapping is not None
                self._compile_variant(
                    argument_descriptors=_make_argument_descriptors(
                        self.program_type, self.argument_descriptor_mapping, args, kwargs
                    ),
                    offset_provider=offset_provider,
                )
                return self(
                    *args, offset_provider=offset_provider, enable_jit=False, **kwargs
                )  # passing `enable_jit=False` because a cache miss should be a hard-error in this call`
            raise RuntimeError("No program compiled for this set of static arguments.") from e

    @functools.cached_property
    def _argument_descriptor_cache_key_from_args(self) -> Callable:
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
        argument_descriptor_contexts: ArgumentDescriptorContexts,
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
        self, argument_descriptors: ArgumentDescriptors
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

    def _compile_variant(
        self,
        argument_descriptors: ArgumentDescriptors,
        offset_provider: common.OffsetProviderType | common.OffsetProvider,
    ) -> None:
        self._initialize_argument_descriptor_mapping(argument_descriptors)
        _validate_argument_descriptors(self.program_type, argument_descriptors)

        argument_descriptor_contexts = _convert_to_argument_descriptor_context(
            self.program_type.definition, argument_descriptors
        )
        key = (
            self._argument_descriptor_cache_key_from_descriptors(argument_descriptor_contexts),
            self._offset_provider_to_type_unsafe(offset_provider),
        )
        if key in self._compiled_programs:
            raise ValueError(f"Program with key {key} already exists.")

        compile_time_args = arguments.CompileTimeArgs(
            offset_provider=offset_provider,  # type:ignore[arg-type] # TODO(havogt): resolve OffsetProviderType vs OffsetProvider
            column_axis=None,  # TODO(havogt): column_axis seems to a unused, even for programs with scans
            args=tuple(self.program_type.definition.pos_only_args)
            + tuple(self.program_type.definition.pos_or_kw_args.values()),
            kwargs=self.program_type.definition.kw_only_args,
            argument_descriptor_contexts=argument_descriptor_contexts,
        )
        compile_call = functools.partial(
            self.backend.compile, self.definition_stage, compile_time_args=compile_time_args
        )
        if _async_compilation_pool is None:
            # synchronous compilation
            self._compiled_programs[key] = compile_call()
        else:
            self._compiled_programs[key] = _async_compilation_pool.submit(compile_call)

    def _offset_provider_to_type_unsafe(
        self,
        offset_provider: common.OffsetProvider | common.OffsetProviderType,
    ) -> common.OffsetProviderType:
        try:
            op_type = self._offset_provider_type_cache[offset_provider]
        except KeyError:
            op_type = (
                offset_provider
                if common.is_offset_provider_type(offset_provider)
                else common.offset_provider_to_type(offset_provider)
            )
            self._offset_provider_type_cache[offset_provider] = op_type
        return op_type

    # TODO(tehrengruber): Rework the interface to allow precompilation with compile time
    #  domains.
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

    def _resolve_future(self, key: CompiledProgramsKey) -> stages.CompiledProgram:
        program = self._compiled_programs[key]
        assert isinstance(program, concurrent.futures.Future)
        result = program.result()
        self._compiled_programs[key] = result
        return result
