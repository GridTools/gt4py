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


_binding_source_not_persistent = (
    _bind_header
    + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    last_call_args[4] = ctypes.c_double(args[0])
    last_call_args[0].value = args[1].data_ptr()
    last_call_args[7] = ctypes.c_int(_get_stride(args[1].ndarray, 0))
    last_call_args[1].value = args[2][0].data_ptr()
    last_call_args[8] = ctypes.c_int(_get_stride(args[2][0].ndarray, 0))
    last_call_args[9] = ctypes.c_int(_get_stride(args[2][0].ndarray, 1))
    last_call_args[2].value = args[2][1][0].data_ptr()
    last_call_args[10] = ctypes.c_int(_get_stride(args[2][1][0].ndarray, 0))
    last_call_args[11] = ctypes.c_int(_get_stride(args[2][1][0].ndarray, 1))
    last_call_args[5] = ctypes.c_double(args[2][1][1])
    last_call_args[6] = ctypes.c_double(args[3])
    last_call_args[3].value = args[4].data_ptr()
    last_call_args[12] = ctypes.c_int(args[4].domain.ranges[0].start)
    last_call_args[13] = ctypes.c_int(args[4].domain.ranges[0].stop)
    last_call_args[14] = ctypes.c_int(args[4].domain.ranges[1].start)
    last_call_args[15] = ctypes.c_int(args[4].domain.ranges[1].stop)
    last_call_args[16] = ctypes.c_int(_get_stride(args[4].ndarray, 0))
    last_call_args[17] = ctypes.c_int(_get_stride(args[4].ndarray, 1))\
"""
)


_binding_source_persistent = (
    _bind_header
    + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    last_call_args[4] = ctypes.c_double(args[0])
    last_call_args[0].value = args[1].data_ptr()
    last_call_args[1].value = args[2][0].data_ptr()
    last_call_args[2].value = args[2][1][0].data_ptr()
    last_call_args[5] = ctypes.c_double(args[2][1][1])
    last_call_args[6] = ctypes.c_double(args[3])
    last_call_args[3].value = args[4].data_ptr()\
"""
)


# The difference between the two bindings versions is that the shape and strides
# of array 'E' are not updated when 'make_persistent=True'. Therefore, the lines
# for updating 'last_call_args[12-15]' are missing in this binding code.
assert _binding_source_persistent != _binding_source_not_persistent


def _language_settings() -> languages.LanguageSettings:
    return languages.LanguageSettings(formatter_key="", formatter_style="", file_extension="sdfg")


def _make_sdfg(sdfg_name: str) -> dace.SDFG:
    sdfg = dace.SDFG(sdfg_name)

    A, _ = sdfg.add_scalar("A", dace.float64)

    B_dim0_rstart, B_dim0_rstop = (dace.symbol(f"__B_0_range_{i}") for i in (0, 1))
    # set 'B_dim1' size to constant value to test the case of constant size in one dimension
    B_shape = (B_dim0_rstop - B_dim0_rstart, 10)
    # set 'B_dim1' stride to constant value
    B_stride0, _ = (dace.symbol(f"__B_stride_{i}") for i in (0, 1))
    B, _ = sdfg.add_array("B", B_shape, dace.int32, strides=(B_stride0, 1))

    C_0_dim0_rstart, C_0_dim0_rstop = (dace.symbol(f"__C_0_0_range_{i}") for i in (0, 1))
    C_0_dim1_rstart, C_0_dim1_rstop = (dace.symbol(f"__C_0_1_range_{i}") for i in (0, 1))
    C_0_shape = (C_0_dim0_rstop - C_0_dim0_rstart, C_0_dim1_rstop - C_0_dim1_rstart)
    C_0_strides = tuple(dace.symbol(f"__C_0_stride_{i}") for i in (0, 1))
    C_0, _ = sdfg.add_array("C_0", C_0_shape, dace.int32, strides=C_0_strides)

    C_1_0_dim0_rstart, C_1_0_dim0_rstop = (dace.symbol(f"__C_1_0_0_range_{i}") for i in (0, 1))
    C_1_0_dim1_rstart, C_1_0_dim1_rstop = (dace.symbol(f"__C_1_0_1_range_{i}") for i in (0, 1))
    C_1_0_shape = (C_1_0_dim0_rstop - C_1_0_dim0_rstart, C_1_0_dim1_rstop - C_1_0_dim1_rstart)
    C_1_0_strides = tuple(dace.symbol(f"__C_1_0_stride_{i}") for i in (0, 1))
    C_1_0, _ = sdfg.add_array("C_1_0", C_1_0_shape, dace.int32, strides=C_1_0_strides)

    C_1_1, _ = sdfg.add_scalar("C_1_1", dace.float64)

    D, _ = sdfg.add_scalar("D", dace.float64)

    E_dim0_rstart, E_dim0_rstop = (dace.symbol(f"__E_0_range_{i}") for i in (0, 1))
    E_dim1_rstart, E_dim1_rstop = (dace.symbol(f"__E_1_range_{i}") for i in (0, 1))
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
    "persistent_config",
    [(False, _binding_source_not_persistent), (True, _binding_source_persistent)],
)
def test_bind_sdfg(persistent_config):
    make_persistent, binding_source_ref = persistent_config
    program_name = "sdfg_bindings{}".format("_persistent" if make_persistent else "")

    FloatType = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    FieldType = ts.FieldType(dims=[ffront_test_utils.IDim, ffront_test_utils.JDim], dtype=FloatType)
    TupleType = ts.TupleType(types=[FieldType, ts.TupleType(types=[FieldType, FloatType])])

    sdfg = _make_sdfg(program_name)

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
            implicit_domain=False,
        )
    )

    compilable_program_source = dace_bindings_stage.bind_sdfg(
        program_source, _bind_func_name, make_persistent
    )

    assert compilable_program_source.program_source == program_source
    assert len(compilable_program_source.library_deps) == 0

    # ignore assert statements
    binding_source_pruned = "\n".join(
        line
        for line in compilable_program_source.binding_source.source_code.splitlines()
        if not line.lstrip().startswith("assert")
    )

    assert binding_source_pruned == binding_source_ref
