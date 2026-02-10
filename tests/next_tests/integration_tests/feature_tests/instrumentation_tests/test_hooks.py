# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import pytest

import gt4py.next as gtx
from gt4py.next import common, Dims, gtfn_cpu, typing as gtx_typing
from gt4py.next.instrumentation import hooks

try:
    from gt4py.next.program_processors.runners import dace as dace_backends

    BACKENDS = [None, gtfn_cpu, dace_backends.run_dace_cpu_cached]
except ImportError:
    BACKENDS = [None, gtfn_cpu]


IDim = gtx.Dimension("IDim")


@gtx.field_operator
def fop(cond: bool, a: gtx.Field[gtx.Dims[IDim], float], b: gtx.Field[gtx.Dims[IDim], float]):
    return a if cond else b


@gtx.program
def prog(
    cond: bool,
    a: gtx.Field[gtx.Dims[IDim], gtx.float64],
    b: gtx.Field[gtx.Dims[IDim], gtx.float64],
    out: gtx.Field[gtx.Dims[IDim], gtx.float64],
):
    fop(cond, a, b, out=out)


callback_results = []
embedded_callback_results = []
compiled_callback_results = []


@contextlib.contextmanager
def custom_program_callback(
    program: gtx_typing.Program,
    args: tuple[Any, ...],
    offset_provider: common.OffsetProvider,
    enable_jit: bool,
    kwargs: dict[str, Any],
) -> contextlib.AbstractContextManager:
    callback_results.append(("enter-program", None))

    yield

    callback_results.append(
        (
            "custom_program_callback",
            {
                "program": program.__name__,
                "args": args,
                "offset_provider": offset_provider.keys(),
                "enable_jit": enable_jit,
                "kwargs": kwargs.keys(),
            },
        )
    )


@contextlib.contextmanager
def custom_embedded_program_callback(
    program: gtx_typing.Program,
    args: tuple[Any, ...],
    offset_provider: common.OffsetProvider,
    kwargs: dict[str, Any],
) -> contextlib.AbstractContextManager:
    embedded_callback_results.append(("enter-embedded-program", None))

    yield

    embedded_callback_results.append(
        (
            "custom_embedded_program_callback",
            {
                "program": program.__name__,
                "args": args,
                "offset_provider": offset_provider.keys(),
                "kwargs": kwargs.keys(),
            },
        )
    )


@contextlib.contextmanager
def custom_compiled_program_callback(
    compiled_program: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    offset_provider: common.OffsetProvider,
    root: tuple[str, str],
    key: gtx_typing.CompiledProgramsKey,
) -> contextlib.AbstractContextManager:
    compiled_callback_results.append(("enter-compiled-program", None))

    yield

    compiled_callback_results.append(
        (
            "custom_compiled_program_callback",
            {
                "program": compiled_program,
                "args": args,
                "kwargs": kwargs,
                "offset_provider": offset_provider.keys(),
                "root": root,
                "key": key,
            },
        )
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: getattr(b, "name", str(b)))
def test_program_call_hooks(backend: gtx_typing.Backend):
    size = 10
    a_field = gtx.full([(IDim, size)], 1, dtype=gtx.float64)
    b_field = gtx.full([(IDim, size)], 1, dtype=gtx.float64)
    out_field = gtx.empty([(IDim, size)], dtype=gtx.float64)

    test_program = prog.with_backend(backend)

    # Run the program without hooks
    callback_results.clear()
    embedded_callback_results.clear()
    test_program(True, a_field, b_field, out=out_field)

    # Callbacks should not have been called
    assert callback_results == []
    callback_results.clear()
    assert embedded_callback_results == []
    embedded_callback_results.clear()
    assert compiled_callback_results == []
    compiled_callback_results.clear()

    # Add hooks and run the program again
    hooks.program_call_context.register(custom_program_callback)
    hooks.embedded_program_call_context.register(custom_embedded_program_callback)
    hooks.compiled_program_call_context.register(custom_compiled_program_callback)
    test_program(True, a_field, b_field, out=out_field)

    # Check that the callbacks were called
    assert len(callback_results) == 2
    assert callback_results[0] == ("enter-program", None)

    hook_name, hook_call_info = callback_results[1]
    assert hook_name == "custom_program_callback"
    assert hook_call_info["program"] == test_program.__name__

    if backend is None:
        # The embedded program call hook should have also been called
        # with the embedded backend
        assert len(embedded_callback_results) == 2
        assert embedded_callback_results[0] == ("enter-embedded-program", None)

        hook_name, hook_call_info = embedded_callback_results[1]
        assert hook_name == "custom_embedded_program_callback"
        assert hook_call_info["program"] == prog.__name__

        assert len(compiled_callback_results) == 0

    else:
        # The compiled program call hook should have also been called
        # with the compiled backends
        assert len(compiled_callback_results) == 2
        assert compiled_callback_results[0] == ("enter-compiled-program", None)

        hook_name, hook_call_info = compiled_callback_results[1]
        assert hook_name == "custom_compiled_program_callback"
        assert (
            hook_call_info["program"]
            == test_program._compiled_programs.compiled_programs[hook_call_info["key"]]
        )

        assert len(embedded_callback_results) == 0

    callback_results.clear()
    embedded_callback_results.clear()
    compiled_callback_results.clear()

    # Remove hooks and call the program again
    hooks.program_call_context.remove(custom_program_callback)
    hooks.embedded_program_call_context.remove(custom_embedded_program_callback)
    hooks.compiled_program_call_context.remove(custom_compiled_program_callback)
    test_program(True, a_field, b_field, out=out_field)

    # Callbacks should not have been called
    assert callback_results == []
    callback_results.clear()
    assert embedded_callback_results == []
    embedded_callback_results.clear()
    assert compiled_callback_results == []
    compiled_callback_results.clear()


@pytest.mark.parametrize(
    "backend", [b for b in BACKENDS if b is not None], ids=lambda b: getattr(b, "name", str(b))
)
def test_compile_variant_hook(backend: gtx_typing.Backend):
    def custom_compile_variant_hook(
        program_definition: gtx_typing.DSLDefinition,
        backend: gtx_typing.Backend,
        offset_provider: common.OffsetProviderType | common.OffsetProvider,
        argument_descriptors: dict[type, dict[str, Any]],
        key: gtx_typing.CompiledProgramsKey,
    ) -> None:
        callback_results.append(
            (
                "custom_compile_variant_hook",
                {
                    "program_definition": program_definition,
                    "backend": backend.name,
                    "argument_descriptors": {
                        k.__name__: [*v.keys()] for k, v in argument_descriptors.items()
                    },
                    "key": key,
                },
            )
        )

    callback_results.clear()
    hooks.compile_variant_hook.register(custom_compile_variant_hook)
    testee = prog.with_backend(backend).compile(cond=[True], offset_provider={})
    hooks.compile_variant_hook.remove(custom_compile_variant_hook)

    assert len(callback_results) == 1, f"{callback_results=}"
    hook_name, hook_call_info = callback_results[0]
    assert hook_name == "custom_compile_variant_hook"
    assert hook_call_info["program_definition"] == prog.definition_stage
    assert hook_call_info["backend"] == backend.name
    assert hook_call_info["argument_descriptors"] == {"StaticArg": ["cond"]}
