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
from typing import Any, Callable, DefaultDict, Sequence, TypeAlias, TypeVar

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing, utils as eve_utils
from gt4py.next import backend as gtx_backend, common, config, errors
from gt4py.next.ffront import stages as ffront_stages, type_specifications as ts_ffront
from gt4py.next.otf import arguments, stages
from gt4py.next.type_system import type_info, type_specifications as ts


T = TypeVar("T")

# TODO(havogt): We would like this to be a ProcessPoolExecutor, which requires (to decide what) to pickle.
_async_compilation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.BUILD_JOBS)

ScalarOrTupleOfScalars: TypeAlias = extended_typing.MaybeNestedInTuple[core_defs.Scalar]
CompiledProgramsKey: TypeAlias = tuple[
    tuple[ScalarOrTupleOfScalars, ...], common.OffsetProviderType
]
ArgumentDescriptors: TypeAlias = dict[
    type[arguments.ArgumentDescriptor], dict[str, arguments.ArgumentDescriptor]
]


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
    params = func_type.pos_or_kw_args | func_type.kw_only_args
    return {
        param: type_info.apply_to_primitive_constituents(
            type_map, type_, tuple_constructor=lambda *els: tuple(els)
        )
        for param, type_ in params.items()
    }


def _get_type_of_param_expr(program_type: ts_ffront.ProgramType, expr: str) -> ts.TypeSpec:
    type_ = eval(expr, _make_param_context_from_func_type(program_type.definition))
    assert isinstance(type_, ts.TypeSpec)
    return type_


@dataclasses.dataclass
class CompiledProgramsPool:
    """
    A pool of compiled programs for a given program and backend.

    If 'static_params' is set (or static arguments are passed to 'compile'),
    the pool will create a program for each argument that is marked static
    and each 'OffsetProviderType'.

    If `enable_jit` is True in the call to the pool, it will compile a program
    with static arguments corresponding to the 'static_params', otherwise it
    will error. In the latter case, the pool needs to be filled with call(s)
    to 'compile' before it can be used.
    """

    backend: gtx_backend.Backend
    definition_stage: ffront_stages.ProgramDefinition
    program_type: ts_ffront.ProgramType
    #: mapping from an argument descriptor type to a list of parameters or expression thereof
    #: e.g. `{arguments.StaticArg: ["static_int_param"]}`
    #: Note: The list is not ordered.
    argument_descriptor_mapping: dict[type[arguments.ArgumentDescriptor], Sequence[str]] | None

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
        # TODO: dispatching over offset provider type is wrong. especially when we use compile time domains. test?
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
                    argument_descriptors=self._make_argument_descriptors(*args, **kwargs),
                    offset_provider=offset_provider,
                )
                return self(
                    *args, offset_provider=offset_provider, enable_jit=False, **kwargs
                )  # passing `enable_jit=False` because a cache miss should be a hard-error in this call`
            raise RuntimeError("No program compiled for this set of static arguments.") from e

    @functools.cached_property
    def _argument_descriptor_cache_key_from_args(self) -> Callable:
        func_type = self.program_type.definition
        params = list(func_type.pos_or_kw_args.keys()) + list(func_type.kw_only_args.keys())
        elements: list[str] = []
        for descriptor_cls, arg_exprs in self.argument_descriptor_mapping.items():  # type: ignore[union-attr]  # can never be `None` at this point
            for arg_expr in arg_exprs:
                attr_extractor = descriptor_cls.attribute_extractor(arg_expr)
                elements.extend(attr_extractor.values())
        return eval(f"""lambda {",".join(params)}: ({_make_tuple_expr(elements)})""")

    def _argument_descriptor_cache_key_from_structured_descriptors(
        self,
        argument_descriptors: dict[
            type[arguments.ArgumentDescriptor],
            dict[str, extended_typing.MaybeNestedInTuple[arguments.ArgumentDescriptor | None]],
        ],
    ) -> tuple:
        elements = []
        for descriptor_cls, arg_exprs in self.argument_descriptor_mapping.items():  # type: ignore[union-attr]  # can never be `None` at this point
            for arg_expr in arg_exprs:
                attr_extractor = descriptor_cls.attribute_extractor(arg_expr)
                attrs = attr_extractor.keys()
                for attr in attrs:
                    elements.append(
                        getattr(eval(f"{arg_expr}", argument_descriptors[descriptor_cls]), attr)
                    )
        return tuple(elements)

    @functools.cached_property
    def _descriptor_attr_retrievers(
        self,
    ) -> dict[type[arguments.ArgumentDescriptor], dict[str, Callable]]:
        """
        For each argument expression build a lambda function that constructs (the attributes of)
        its argument descriptor
        """

        def make_dict_expr(exprs: dict[str, str]) -> str:
            return "{" + ",".join((f"'{k}': {v}" for k, v in exprs.items())) + "}"

        func_type = self.program_type.definition
        params = list(func_type.pos_or_kw_args.keys()) + list(func_type.kw_only_args.keys())
        retrievers: dict[type[arguments.ArgumentDescriptor], dict[str, Callable]] = DefaultDict(
            dict
        )
        for descriptor_cls, arg_exprs in self.argument_descriptor_mapping.items():  # type: ignore[union-attr]  # can never be `None` at this point
            for arg_expr in arg_exprs:
                attr_exprs = descriptor_cls.attribute_extractor(arg_expr)
                retrievers[descriptor_cls][arg_expr] = eval(
                    f"""lambda {",".join(params)}: {make_dict_expr(attr_exprs)}"""
                )

        return retrievers

    def _make_argument_descriptors(self, *args: Any, **kwargs: Any) -> ArgumentDescriptors:
        descriptors: ArgumentDescriptors = {}
        for descriptor_cls, attr_retrievers in self._descriptor_attr_retrievers.items():
            descriptors[descriptor_cls] = {}
            for expr, attr_retriever in attr_retrievers.items():
                descriptor = descriptor_cls(**attr_retriever(*args, **kwargs))
                descriptors[descriptor_cls][expr] = descriptor
        self._validate_argument_descriptors(descriptors)
        return descriptors

    def _validate_argument_descriptors(
        self,
        all_descriptors: ArgumentDescriptors,
    ) -> None:
        for descriptors in all_descriptors.values():
            for expr, descriptor in descriptors.items():
                param_type = _get_type_of_param_expr(self.program_type, expr)
                descriptor.validate(expr, param_type)

    def _validate_argument_descriptor_mapping(self) -> None:
        if self.argument_descriptor_mapping is None:
            return
        context = _make_param_context_from_func_type(self.program_type.definition, lambda x: None)
        for descr_cls, exprs in self.argument_descriptor_mapping.items():
            for expr in exprs:
                try:
                    if eval(expr, context) is not None:
                        raise ValueError()
                except (ValueError, KeyError):
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
                        f"Argument descriptor {descr_cls.__name__} must be the same for all compiled programs. Got {list(got)}, expected {list(expected)}."
                    )

        self._validate_argument_descriptors(argument_descriptors)

        structured_descriptors = {}
        for descriptor_cls, descriptor_expr_mapping in argument_descriptors.items():
            structured_descriptors[descriptor_cls] = _make_param_context_from_func_type(
                self.program_type.definition, lambda x: None
            )
            assert "__descriptor" not in structured_descriptors[descriptor_cls]
            for expr, descriptor in descriptor_expr_mapping.items():
                # note: we don't need to handle any errors here since the `expr` has been validated
                #  in `_validate_argument_descriptor_mapping`
                exec(
                    f"{expr} = __descriptor",
                    {"__descriptor": descriptor},
                    structured_descriptors[descriptor_cls],
                )

        key = (
            self._argument_descriptor_cache_key_from_structured_descriptors(structured_descriptors),  # type: ignore[arg-type] # mypy not smart enough
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
            argument_descriptors=argument_descriptors,
        )
        self._compiled_programs[key] = _async_compilation_pool.submit(
            self.backend.compile, self.definition_stage, compile_time_args=compile_time_args
        )

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
