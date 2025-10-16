# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the bindings stage of the dace backend workflow."""

import pytest

dace = pytest.importorskip("dace")

from gt4py.next.otf import languages, stages
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners.dace import utils as gtx_dace_utils
from gt4py.next.program_processors.runners.dace.workflow import bindings as dace_bindings_stage
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests.feature_tests.ffront_tests import ffront_test_utils


_bind_func_name = "dummy"


_bind_header = """\
import ctypes
from gt4py.next import common as gtx_common, field_utils


def _get_stride(ndarray, dim_index):
    return ndarray.strides[dim_index] // ndarray.itemsize


"""


_binding_source = (
    _bind_header
    + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    arg0, arg1, arg2, arg3, arg4 = args
    last_call_args[4] = ctypes.c_double(arg0)
    last_call_args[0].value = arg1.data_ptr()
    last_call_args[7] = ctypes.c_int(_get_stride(arg1.ndarray, 0))
    arg2_0, arg2_1 = arg2
    last_call_args[1].value = arg2_0.data_ptr()
    last_call_args[8] = ctypes.c_int(_get_stride(arg2_0.ndarray, 0))
    last_call_args[9] = ctypes.c_int(_get_stride(arg2_0.ndarray, 1))
    arg2_1_0, arg2_1_1 = arg2_1
    last_call_args[2].value = arg2_1_0.data_ptr()
    last_call_args[10] = ctypes.c_int(_get_stride(arg2_1_0.ndarray, 0))
    last_call_args[11] = ctypes.c_int(_get_stride(arg2_1_0.ndarray, 1))
    last_call_args[5] = ctypes.c_double(arg2_1_1)
    last_call_args[6] = ctypes.c_double(arg3)
    last_call_args[3].value = arg4.data_ptr()
    last_call_args[12] = ctypes.c_int(arg4.domain.ranges[0].start)
    last_call_args[13] = ctypes.c_int(arg4.domain.ranges[0].stop)
    last_call_args[14] = ctypes.c_int(arg4.domain.ranges[1].start)
    last_call_args[15] = ctypes.c_int(arg4.domain.ranges[1].stop)
    last_call_args[16] = ctypes.c_int(_get_stride(arg4.ndarray, 0))
    last_call_args[17] = ctypes.c_int(_get_stride(arg4.ndarray, 1))\
"""
)


_binding_source_with_static_domain = (
    _bind_header
    + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    arg0, arg1, arg2, arg3, arg4 = args
    last_call_args[4] = ctypes.c_double(arg0)
    last_call_args[0].value = arg1.data_ptr()
    last_call_args[7] = ctypes.c_int(_get_stride(arg1.ndarray, 0))
    arg2_0, arg2_1 = arg2
    last_call_args[1].value = arg2_0.data_ptr()
    last_call_args[8] = ctypes.c_int(_get_stride(arg2_0.ndarray, 0))
    last_call_args[9] = ctypes.c_int(_get_stride(arg2_0.ndarray, 1))
    arg2_1_0, arg2_1_1 = arg2_1
    last_call_args[2].value = arg2_1_0.data_ptr()
    last_call_args[10] = ctypes.c_int(_get_stride(arg2_1_0.ndarray, 0))
    last_call_args[11] = ctypes.c_int(_get_stride(arg2_1_0.ndarray, 1))
    last_call_args[5] = ctypes.c_double(arg2_1_1)
    last_call_args[6] = ctypes.c_double(arg3)
    last_call_args[3].value = arg4.data_ptr()
    last_call_args[12] = ctypes.c_int(_get_stride(arg4.ndarray, 0))
    last_call_args[13] = ctypes.c_int(_get_stride(arg4.ndarray, 1))\
"""
)


# The difference between the two bindings versions is that one uses static domain
# for the field arguments, therefore the range symbols are not present in the SDFG.
assert _binding_source_with_static_domain != _binding_source


def _language_settings() -> languages.LanguageSettings:
    return languages.LanguageSettings(formatter_key="", formatter_style="", file_extension="sdfg")


def _make_sdfg(sdfg_name: str, use_static_domain: bool) -> dace.SDFG:
    sdfg = dace.SDFG(sdfg_name)

    M, N = (20, 10)
    A, _ = sdfg.add_scalar("A", dace.float64)

    B_dim0_rstart, B_dim0_rstop = (
        (0, M) if use_static_domain else (dace.symbol(f"__B_0_range_{i}") for i in (0, 1))
    )
    # set 'B_dim1' size to constant value to test the case of constant size in one dimension
    B_shape = (B_dim0_rstop - B_dim0_rstart, 10)
    # set 'B_dim1' stride to constant value
    B_stride0, _ = (dace.symbol(f"__B_stride_{i}") for i in (0, 1))
    B, _ = sdfg.add_array("B", B_shape, dace.int32, strides=(B_stride0, 1))

    C_0_dim0_rstart, C_0_dim0_rstop = (
        (0, M)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("C_0", ffront_test_utils.IDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("C_0", ffront_test_utils.IDim)),
        )
    )
    C_0_dim1_rstart, C_0_dim1_rstop = (
        (0, N)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("C_0", ffront_test_utils.JDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("C_0", ffront_test_utils.JDim)),
        )
    )
    C_0_shape = (C_0_dim0_rstop - C_0_dim0_rstart, C_0_dim1_rstop - C_0_dim1_rstart)
    C_0_strides = tuple(dace.symbol(f"__C_0_stride_{i}") for i in (0, 1))
    C_0, _ = sdfg.add_array("C_0", C_0_shape, dace.int32, strides=C_0_strides)

    C_1_0_dim0_rstart, C_1_0_dim0_rstop = (
        (0, M)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("C_1_0", ffront_test_utils.IDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("C_1_0", ffront_test_utils.IDim)),
        )
    )
    C_1_0_dim1_rstart, C_1_0_dim1_rstop = (
        (0, N)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("C_1_0", ffront_test_utils.JDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("C_1_0", ffront_test_utils.JDim)),
        )
    )
    C_1_0_shape = (C_1_0_dim0_rstop - C_1_0_dim0_rstart, C_1_0_dim1_rstop - C_1_0_dim1_rstart)
    C_1_0_strides = tuple(dace.symbol(f"__C_1_0_stride_{i}") for i in (0, 1))
    C_1_0, _ = sdfg.add_array("C_1_0", C_1_0_shape, dace.int32, strides=C_1_0_strides)

    C_1_1, _ = sdfg.add_scalar("C_1_1", dace.float64)

    D, _ = sdfg.add_scalar("D", dace.float64)

    E_dim0_rstart, E_dim0_rstop = (
        (0, M)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("E", ffront_test_utils.IDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("E", ffront_test_utils.IDim)),
        )
    )
    E_dim1_rstart, E_dim1_rstop = (
        (0, N)
        if use_static_domain
        else (
            dace.symbol(gtx_dace_utils.range_start_symbol("E", ffront_test_utils.JDim)),
            dace.symbol(gtx_dace_utils.range_stop_symbol("E", ffront_test_utils.JDim)),
        )
    )
    E_shape = (E_dim0_rstop - E_dim0_rstart, E_dim1_rstop - E_dim1_rstart)
    E_strides = tuple(dace.symbol(f"__E_stride_{i}") for i in (0, 1))
    E, _ = sdfg.add_array("E", E_shape, dace.int32, strides=E_strides)

    st = sdfg.add_state()
    st.add_mapped_tasklet(
        "compute",
        code="result = arg0 + arg1 + arg2 + arg3 + arg4 + arg5",
        map_ranges=dict(i=f"{E_dim0_rstart}:{E_dim0_rstop}", j=f"{E_dim1_rstart}:{E_dim1_rstop}"),
        inputs={
            "arg0": dace.Memlet(data=A, subset="0"),
            "arg1": dace.Memlet(data=B, subset=f"i-{E_dim0_rstart},j-{E_dim1_rstart}"),
            "arg2": dace.Memlet(data=C_0, subset=f"i-{E_dim0_rstart},j-{E_dim1_rstart}"),
            "arg3": dace.Memlet(data=C_1_0, subset=f"i-{E_dim0_rstart},j-{E_dim1_rstart}"),
            "arg4": dace.Memlet(data=C_1_1, subset="0"),
            "arg5": dace.Memlet(data=D, subset="0"),
        },
        outputs={
            "result": dace.Memlet(data=E, subset=f"i-{E_dim0_rstart},j-{E_dim1_rstart}"),
        },
        external_edges=True,
    )

    sdfg.validate()
    return sdfg


@pytest.mark.parametrize(
    "static_domain_config",
    [(False, _binding_source), (True, _binding_source_with_static_domain)],
)
def test_bind_sdfg(static_domain_config):
    use_static_domain, binding_source_ref = static_domain_config
    program_name = "sdfg_bindings{}".format("_static_domain" if use_static_domain else "")

    FloatType = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    FieldType = ts.FieldType(dims=[ffront_test_utils.IDim, ffront_test_utils.JDim], dtype=FloatType)
    TupleType = ts.TupleType(types=[FieldType, ts.TupleType(types=[FieldType, FloatType])])

    sdfg = _make_sdfg(program_name, use_static_domain)

    program_parameters = (
        interface.Parameter("A", FloatType),
        interface.Parameter("B", FieldType),
        interface.Parameter("C", TupleType),
        interface.Parameter("D", FloatType),
        interface.Parameter("E", FieldType),
    )

    program_source: stages.ProgramSource[dace_bindings_stage.SrcL, languages.LanguageSettings] = (
        stages.ProgramSource(
            entry_point=interface.Function(program_name, tuple(program_parameters)),
            source_code=sdfg.to_json(),
            library_deps=tuple(),
            language=languages.SDFG,
            language_settings=_language_settings(),
        )
    )

    compilable_program_source = dace_bindings_stage.bind_sdfg(program_source, _bind_func_name)

    assert compilable_program_source.program_source == program_source
    assert len(compilable_program_source.library_deps) == 0

    # ignore assert statements
    binding_source_pruned = "\n".join(
        line
        for line in compilable_program_source.binding_source.source_code.splitlines()
        if not line.lstrip().startswith("assert")
    )

    assert binding_source_pruned == binding_source_ref
