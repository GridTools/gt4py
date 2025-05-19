# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the translation stage of the dace backend workflow."""

import pytest

dace = pytest.importorskip("dace")

from gt4py._core import definitions as core_defs
from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.workflow import (
    translation as dace_translation_stage,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests import ffront_test_utils


FloatType = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
FieldType = ts.FieldType(dims=[ffront_test_utils.IDim], dtype=FloatType)
IntType = ts.ScalarType(kind=ts.ScalarKind.INT32)


def _is_present_async_sdfg_init_code(sdfg: dace.SDFG) -> bool:
    async_sdfg_init_code = f"__set_stream_{sdfg.name}(__state, cudaStreamDefault);"
    return (
        len(
            [
                line
                for line in sdfg.init_code["frame"].code.splitlines()
                if line == async_sdfg_init_code
            ]
        )
        >= 1
    )


def _is_present_async_sdfg_global_code(sdfg: dace.SDFG) -> bool:
    async_sdfg_global_code = f"""\
DACE_EXPORTED bool __dace_gpu_set_stream({sdfg.name}_state_t *__state, int streamid, gpuStream_t stream);
DACE_EXPORTED void __set_stream_{sdfg.name}({sdfg.name}_state_t *__state, gpuStream_t stream) {{
for (int i = 0; i < __state->gpu_context->num_streams; i++)
    __dace_gpu_set_stream(__state, i, stream);
}}\
"""
    return async_sdfg_global_code in sdfg.global_code["frame"].code


def _check_sdfg_with_async_call(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert st.nosync == True

    assert _is_present_async_sdfg_global_code(sdfg)
    assert _is_present_async_sdfg_init_code(sdfg)


def _check_sdfg_without_async_call(sdfg: dace.SDFG) -> None:
    assert len(sdfg.states()) == 1
    st = sdfg.states()[0]
    assert st.nosync == False

    assert not _is_present_async_sdfg_global_code(sdfg)
    assert not _is_present_async_sdfg_init_code(sdfg)


@pytest.mark.parametrize(
    "make_async_sdfg_call",
    [False, True],
)
def test_generate_sdfg_async_call(make_async_sdfg_call):
    """Verify that the flag `async_sdfg_call` takes effect on the SDFG generation."""
    program_name = "field_ir_{}_async_call".format("with" if make_async_sdfg_call else "without")
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={ffront_test_utils.IDim: (0, "N")})

    program = itir.Program(
        id=program_name,
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=FieldType),
            itir.Sym(id="y", type=FieldType),
            itir.Sym(id="N", type=IntType),
        ],
        body=[
            itir.SetAt(
                expr=im.op_as_fieldop("plus", domain)("x", 1.0),
                domain=domain,
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = dace_translation_stage.DaCeTranslator(
        device_type=core_defs.CUPY_DEVICE_TYPE,
        auto_optimize=False,
        async_sdfg_call=make_async_sdfg_call,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    if make_async_sdfg_call:
        _check_sdfg_with_async_call(sdfg)
    else:
        _check_sdfg_without_async_call(sdfg)


def test_generate_sdfg_async_call_no_map():
    """Verify that the flag `async_sdfg_call=True` has no effect on an SDFG that does not contain any GPU map."""
    program_name = "scalar_ir_with_async_call"
    domain = im.domain(gtx_common.GridType.CARTESIAN, ranges={ffront_test_utils.IDim: (0, "N")})

    program = itir.Program(
        id=program_name,
        declarations=[],
        function_definitions=[],
        params=[
            itir.Sym(id="x", type=FieldType),
            itir.Sym(id="y", type=FieldType),
            itir.Sym(id="N", type=IntType),
        ],
        body=[
            itir.SetAt(
                expr=itir.SymRef(id="x"),
                domain=domain,
                target=itir.SymRef(id="y"),
            ),
        ],
    )

    sdfg = dace_translation_stage.DaCeTranslator(
        device_type=core_defs.CUPY_DEVICE_TYPE,
        auto_optimize=False,
        async_sdfg_call=True,
    ).generate_sdfg(program, offset_provider={}, column_axis=None)

    _check_sdfg_without_async_call(sdfg)
