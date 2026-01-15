# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
from collections.abc import Hashable
from typing import Any, Final

import pytest

from gt4py.next import Dims, gtfn_cpu, broadcast, typing as gtx_typing
import gt4py.next as gtx
from gt4py.next.instrumentation import hooks
from gt4py.next import common, backend as gtx_backend
from gt4py.next.typing import Program

try:
    from gt4py.next.program_processors.runners import dace as dace_backends

    BACKENDS = [None, gtfn_cpu, dace_backends.run_dace_cpu_cached]
except ImportError:
    BACKENDS = [None, gtfn_cpu]


callback_results = []
embedded_callback_results = []


@contextlib.contextmanager
def custom_program_callback(
    program: Program,
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
    program: Program,
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


# @hooks.compile_variant_hook.register
# def compile_variant_hook(
#     key: tuple[tuple[Hashable, ...], int],
#     backend: gtx_backend.Backend,
#     program_definition: "ffront_stages.ProgramDefinition",
#     compile_time_args: "arguments.CompileTimeArgs",
# ) -> None:
#     """Callback hook invoked before compiling a program variant."""
#     ...


Cell = gtx.Dimension("Cell")
IDim = gtx.Dimension("IDim")


@gtx.field_operator
def identity_fop(
    in_field: gtx.Field[Dims[IDim], gtx.float64],
) -> gtx.Field[Dims[IDim], gtx.float64]:
    return in_field


@gtx.program
def copy_program(
    in_field: gtx.Field[Dims[IDim], gtx.float64], out: gtx.Field[Dims[IDim], gtx.float64]
):
    identity_fop(in_field, out=out)


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda b: getattr(b, "name", str(b)))
def test_program_call_hooks(backend: gtx_typing.Backend):
    size = 10
    in_field = gtx.full([(IDim, size)], 1, dtype=gtx.float64)
    out_field = gtx.empty([(IDim, size)], dtype=gtx.float64)

    test_program = copy_program.with_backend(backend)

    # Run the program without hooks
    callback_results.clear()
    embedded_callback_results.clear()
    test_program(in_field, out=out_field)

    # Callbacks should not have been called
    assert callback_results == []
    callback_results.clear()
    assert embedded_callback_results == []
    embedded_callback_results.clear()

    # Add hooks and run the program again
    hooks.program_call_hook.register(custom_program_callback)
    hooks.embedded_program_call_hook.register(custom_embedded_program_callback)
    test_program(in_field, out=out_field)

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
        assert hook_call_info["program"] == copy_program.__name__
    else:
        assert len(embedded_callback_results) == 0

    callback_results.clear()
    embedded_callback_results.clear()

    # Remove hooks and call the program again
    hooks.program_call_hook.remove(custom_program_callback)
    hooks.embedded_program_call_hook.remove(custom_embedded_program_callback)
    test_program(in_field, out=out_field)

    # Callbacks should not have been called
    assert callback_results == []
    callback_results.clear()
    assert embedded_callback_results == []
    embedded_callback_results.clear()
