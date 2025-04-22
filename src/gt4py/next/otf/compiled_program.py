# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures
import dataclasses
import functools
import itertools
from collections.abc import Sequence
from typing import Any, TypeAlias

from gt4py._core import definitions as core_defs
from gt4py.next import backend as gtx_backend, common, config
from gt4py.next.ffront import stages as ffront_stages, type_specifications as ts_ffront
from gt4py.next.otf import arguments, stages
from gt4py.next.type_system import type_info


# TODO(havogt): We would like this to be a ProcessPoolExecutor, which requires (to decide what) to pickle.
_async_compilation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.BUILD_JOBS)

ScalarOrTupleOfScalars: TypeAlias = (
    core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...]
)  # or nested


@dataclasses.dataclass  # not frozen for performance
class _CompiledProgramsKey:
    values: tuple[ScalarOrTupleOfScalars, ...]
    offset_provider_type: common.OffsetProviderType

    def __hash__(self) -> int:
        return hash((self.values, tuple(self.offset_provider_type.items())))


class CompiledPrograms:
    backend: gtx_backend.Backend
    definition_stage: ffront_stages.ProgramDefinition
    program_type: ts_ffront.ProgramType
    _static_arg_indices: tuple[int, ...] | None = None

    _compiled_programs: dict[
        _CompiledProgramsKey,
        stages.CompiledProgram | concurrent.futures.Future[stages.CompiledProgram],
    ]

    def __init__(
        self,
        backend: gtx_backend.Backend,
        definition_stage: ffront_stages.ProgramDefinition,
        program_type: ts_ffront.ProgramType,
        static_params: Sequence[str] | None = None,
    ) -> None:
        self.backend = backend
        self.definition_stage = definition_stage
        self.program_type = program_type

        # TODO(havogt): We currently don't support pos_only or kw_only args at the program level.
        # This check makes sure we don't miss updating this code if we add support for them in the future.
        assert not self.program_type.definition.kw_only_args
        assert not self.program_type.definition.pos_only_args

        if static_params is not None:
            self._init_static_arg_indices(static_params)
        self._compiled_programs = {}

    def __call__(
        self, *args: Any, offset_provider: common.OffsetProvider, enable_jit: bool, **kwargs: Any
    ) -> None:
        assert self._static_arg_indices is not None
        args, kwargs = type_info.canonicalize_arguments(self.program_type, args, kwargs)
        offset_provider_type = _offset_provider_to_type_unsafe(offset_provider)
        key = _CompiledProgramsKey(
            tuple(args[i] for i in self._static_arg_indices),
            offset_provider_type,
        )
        try:
            self._compiled_programs[key](*args, **kwargs, offset_provider=offset_provider)  # type: ignore[operator] # for performance: try to call first...
        except TypeError:  # 'Future' object is not callable
            # ... otherwise we resolve the future and call again
            program = self._resolve_future(key)
            program(*args, **kwargs, offset_provider=offset_provider)
        except KeyError as e:
            if enable_jit:
                static_values = tuple(args[i] for i in self._static_arg_indices)
                self._compile_variant(
                    static_values=static_values, offset_provider_type=offset_provider
                )
                return self(
                    *args, offset_provider=offset_provider, enable_jit=False, **kwargs
                )  # passing `enable_jit=False` as a cache miss should be a hard-error in this call`
            raise RuntimeError("No program compiled for this set of static arguments.") from e

    def _init_static_arg_indices(self, static_params: Sequence[str]) -> None:
        """
        Initializes the static argument indices for this program.
        """
        arg_types_dict = self.program_type.definition.pos_or_kw_args
        static_args_indices = tuple(
            sorted(list(arg_types_dict.keys()).index(k) for k in static_params)
        )

        if self._static_arg_indices is None:
            self._static_arg_indices = static_args_indices
        elif self._static_arg_indices != static_args_indices:
            raise ValueError("Static arguments must be the same for all compiled programs.")

    def _compile_variant(
        self,
        static_values: tuple[ScalarOrTupleOfScalars, ...],
        offset_provider_type: common.OffsetProviderType | common.OffsetProvider,
    ) -> None:
        assert self._static_arg_indices is not None
        index_to_value = dict(zip(self._static_arg_indices, static_values))
        args = tuple(
            type_
            if i not in index_to_value
            else arguments.StaticArg(value=index_to_value[i], type_=type_)
            for i, type_ in enumerate(self.program_type.definition.pos_or_kw_args.values())
        )
        compile_time_args = arguments.CompileTimeArgs(
            offset_provider=offset_provider_type,  # type:ignore[arg-type] # TODO(havogt): resolve OffsetProviderType vs OffsetProvider
            column_axis=None,  # TODO(havogt): column_axis seems to a unused, even for programs with scans
            args=args,
            kwargs={},
        )
        key = _CompiledProgramsKey(
            static_values,
            offset_provider_type
            if common.is_offset_provider_type(offset_provider_type)
            else common.offset_provider_to_type(offset_provider_type),
        )
        if key in self._compiled_programs:
            raise ValueError(f"Program with key {key} already exists.")
        self._compiled_programs[key] = _async_compilation_pool.submit(
            self.backend.compile, self.definition_stage, compile_time_args=compile_time_args
        )

    def compile(
        self,
        offset_provider_type: common.OffsetProvider | common.OffsetProviderType,
        **static_args: list[core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...]],
    ) -> None:
        self._init_static_arg_indices(static_params=tuple(static_args.keys()))

        for static_values in itertools.product(*static_args.values()):
            self._compile_variant(tuple(static_values), offset_provider_type)

    def _resolve_future(self, key: _CompiledProgramsKey) -> stages.CompiledProgram:
        program = self._compiled_programs[key]
        assert isinstance(program, concurrent.futures.Future)
        result = program.result()
        self._compiled_programs[key] = result
        return result

    @functools.cached_property
    def static_params(self) -> tuple[str, ...]:
        """
        Returns the names of the static parameters for this program.
        """
        names = tuple(self.program_type.definition.pos_or_kw_args.keys())
        assert self._static_arg_indices is not None
        return tuple(names[i] for i in self._static_arg_indices)


def _offset_provider_to_type_unsafe(
    offset_provider: common.OffsetProvider,
) -> common.OffsetProviderType:
    @functools.lru_cache(maxsize=128)
    def _offset_provider_to_type_unsafe_impl(_: int) -> common.OffsetProviderType:
        return common.offset_provider_to_type(offset_provider)

    return _offset_provider_to_type_unsafe_impl(id(offset_provider))
