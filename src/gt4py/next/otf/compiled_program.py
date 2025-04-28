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
from collections.abc import Sequence
from typing import Any, TypeAlias

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing, utils as eve_utils
from gt4py.next import backend as gtx_backend, common, config
from gt4py.next.ffront import stages as ffront_stages, type_specifications as ts_ffront
from gt4py.next.otf import arguments, stages
from gt4py.next.type_system import type_info


# TODO(havogt): We would like this to be a ProcessPoolExecutor, which requires (to decide what) to pickle.
_async_compilation_pool = concurrent.futures.ThreadPoolExecutor(max_workers=config.BUILD_JOBS)

ScalarOrTupleOfScalars: TypeAlias = extended_typing.TOrNestedTuple[core_defs.Scalar]


@dataclasses.dataclass  # not frozen for performance
class _CompiledProgramsKey:
    values: tuple[ScalarOrTupleOfScalars, ...]  # in order of static_params
    offset_provider_type: common.OffsetProviderType

    def __hash__(self) -> int:
        assert common.is_offset_provider_type(self.offset_provider_type)
        return hash((self.values, frozenset(self.offset_provider_type.items())))


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
    static_params: Sequence[str] | None = None  # not ordered

    # TODO(havogt): This dict could be replaced by a `functools.cache`d method
    # and appropriate hashing of the arguments.
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

        self.static_params = static_params
        self._compiled_programs = {}

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
                assert self.static_params is not None
                static_args = {name: args[self._param_index(name)] for name in self.static_params}
                self._compile_variant(static_args=static_args, offset_provider_type=offset_provider)
                return self(
                    *args, offset_provider=offset_provider, enable_jit=False, **kwargs
                )  # passing `enable_jit=False` because a cache miss should be a hard-error in this call`
            raise RuntimeError("No program compiled for this set of static arguments.") from e

    def _param_index(self, name: str) -> int:
        return list(self.program_type.definition.pos_or_kw_args.keys()).index(name)

    @functools.cached_property
    def _static_arg_indices(self) -> tuple[int, ...]:
        if self.static_params is None:
            # this could also be done in `__call__` but would be an extra if in the fast path
            self.static_params = ()
        return tuple(self._param_index(p) for p in self.static_params)

    def _compile_variant(
        self,
        static_args: dict[str, ScalarOrTupleOfScalars],
        offset_provider_type: common.OffsetProviderType | common.OffsetProvider,
    ) -> None:
        assert self.static_params is not None
        args = tuple(
            arguments.StaticArg(value=static_args[name], type_=type_)
            if name in self.static_params
            else type_
            for name, type_ in self.program_type.definition.pos_or_kw_args.items()
        )
        compile_time_args = arguments.CompileTimeArgs(
            offset_provider=offset_provider_type,  # type:ignore[arg-type] # TODO(havogt): resolve OffsetProviderType vs OffsetProvider
            column_axis=None,  # TODO(havogt): column_axis seems to a unused, even for programs with scans
            args=args,
            kwargs={},
        )
        key = _CompiledProgramsKey(
            tuple(static_args[p] for p in self.static_params),
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
        if self.static_params is not None:
            if not sorted(self.static_params) == sorted(static_args.keys()):
                raise ValueError(
                    f"Static arguments must be the same for all compiled programs. Got {list(static_args.keys())}, expected {self.static_params}."
                )
        else:
            self.static_params = tuple(static_args.keys())

        for static_values in itertools.product(*static_args.values()):
            self._compile_variant(
                dict(zip(static_args.keys(), static_values, strict=True)), offset_provider_type
            )

    def _resolve_future(self, key: _CompiledProgramsKey) -> stages.CompiledProgram:
        program = self._compiled_programs[key]
        assert isinstance(program, concurrent.futures.Future)
        result = program.result()
        self._compiled_programs[key] = result
        return result


@functools.lru_cache(maxsize=128)
def _offset_provider_to_type_unsafe_impl(
    hashable_offset_provider: eve_utils.HashableBy[common.OffsetProvider],
) -> common.OffsetProviderType:
    return common.offset_provider_to_type(hashable_offset_provider.value)


def _offset_provider_to_type_unsafe(
    offset_provider: common.OffsetProvider,
) -> common.OffsetProviderType:
    return _offset_provider_to_type_unsafe_impl(
        eve_utils.hashable_by_id(offset_provider),
    )
