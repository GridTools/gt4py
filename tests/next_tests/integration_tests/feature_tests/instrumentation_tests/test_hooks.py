# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
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


@contextlib.contextmanager
def custom_program_callback(
    program: gtx_typing.Program,
    args: tuple[Any, ...],
    offset_provider: common.OffsetProvider,
    enable_jit: bool,
    kwargs: dict[str, Any],
) -> contextlib.AbstractContextManager:
    callback_results.append(("enter", None))

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
    embedded_callback_results.append(("enter", None))

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

    # Add hooks and run the program again
    hooks.program_call_context.register(custom_program_callback)
    hooks.embedded_program_call_context.register(custom_embedded_program_callback)
    test_program(True, a_field, b_field, out=out_field)

    # Check that the callbacks were called
    assert len(callback_results) == 2
    assert callback_results[0] == ("enter", None)

    hook_name, hook_call_info = callback_results[1]
    assert hook_name == "custom_program_callback"
    assert hook_call_info["program"] == test_program.__name__

    # The embedded program call hook should have also been called
    # with the embedded backend
    if backend is None:
        assert len(embedded_callback_results) == 2
        assert embedded_callback_results[0] == ("enter", None)

        hook_name, hook_call_info = embedded_callback_results[1]
        assert hook_name == "custom_embedded_program_callback"
        assert hook_call_info["program"] == prog.__name__
    else:
        assert len(embedded_callback_results) == 0

    callback_results.clear()
    embedded_callback_results.clear()

    # Remove hooks and call the program again
    hooks.program_call_context.remove(custom_program_callback)
    hooks.embedded_program_call_context.remove(custom_embedded_program_callback)
    test_program(True, a_field, b_field, out=out_field)

    # Callbacks should not have been called
    assert callback_results == []
    callback_results.clear()
    assert embedded_callback_results == []
    embedded_callback_results.clear()


@pytest.mark.parametrize(
    "backend", [b for b in BACKENDS if b is not None], ids=lambda b: getattr(b, "name", str(b))
)
def test_compile_variant_hook(backend: gtx_typing.Backend):
    @hooks.compile_variant_hook.register
    def custom_compile_variant_hook(
        program_definition,  #: gtx_typing.DSLDefinitionForm,
        backend: gtx_typing.Backend,
        offset_provider: common.OffsetProviderType | common.OffsetProvider,
        argument_descriptors,  #: common.ArgumentDescriptors,
        key,
    ) -> None:
        callback_results.append(
            (
                "custom_compile_variant_hook",
                {
                    "program_definition": program_definition,
                    "backend": backend.name,
                    "argument_descriptors": list(argument_descriptors.keys()),
                    "key": key,
                },
            )
        )

    callback_results.clear()
    testee = prog.with_backend(backend).compile(cond=[True], offset_provider={})

    assert len(callback_results) == 1
    hook_name, hook_call_info = callback_results[0]
    assert hook_name == "custom_compile_variant_hook"
    assert hook_call_info["program_definition"] == prog.definition_stage.name
    assert hook_call_info["backend"] == backend.name
    assert set(hook_call_info["argument_descriptors"]) == {"cond", "a", "b", "out"}
