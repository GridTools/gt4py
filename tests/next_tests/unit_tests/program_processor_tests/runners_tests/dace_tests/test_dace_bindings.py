# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the bindings stage of the dace backend workflow."""

import functools
import numpy as np
import pytest

dace = pytest.importorskip("dace")

from gt4py import next as gtx
from gt4py.next import common as gtx_common, int32
from gt4py.next.otf import languages, stages
from gt4py.next.program_processors.runners import dace as dace_runner
from gt4py.next.program_processors.runners.dace import workflow as dace_workflow
from gt4py.next import neighbor_sum
from next_tests.integration_tests.cases import E2V, E2VDim, V2E, V2EDim

from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests import ffront_test_utils
from next_tests.unit_tests.test_common import IDim, JDim, KDim


_bind_func_name = "update_sdfg_args"


_bind_header = """\
import ctypes
from gt4py.next import common as gtx_common, field_utils


"""


def _binding_source_cartesian(use_metrics: bool) -> str:
    # In this SDFG 'sdfg_call_args[2]' is used to collect the stencil compute time.
    # Note that the position of 'gt_compute_time' in the SDFG arguments list is
    # defined by dace, based on alphabetical order from index 0 ('a', 'b', 'gt_compute_time').
    metrics_arg_index = 2
    idx = [21, 22, 0, 3, 5, 7, 4, 6, 8, 1, 9, 11, 13, 10, 12, 14, 23, 2, 15, 17, 19, 16, 18, 20]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, sdfg_call_args, offset_provider):
    (
        args_0,
        args_1,
        args_2,
        args_3,
        args_4,
        args_5,
    ) = args
    (
        args_0_0,
        args_0_1,
    ) = args_0
    sdfg_call_args[{idx[0]}] = ctypes.c_int(args_0_0)
    (
        args_0_1_0,
        args_0_1_1,
        args_0_1_2,
    ) = args_0_1
    sdfg_call_args[{idx[1]}] = ctypes.c_int(args_0_1_0)
    sdfg_call_args[{idx[2]}].value = args_0_1_1.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[3]}] = ctypes.c_int(args_0_1_1.domain.ranges[0].start)
    sdfg_call_args[{idx[4]}] = ctypes.c_int(args_0_1_1.domain.ranges[1].start)
    sdfg_call_args[{idx[5]}] = ctypes.c_int(args_0_1_1.domain.ranges[2].start)
    sdfg_call_args[{idx[6]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[7]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[8]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[2])
    (
        args_1_0,
        args_1_1,
    ) = args_1
    (args_1_0_0,) = args_1_0
    sdfg_call_args[{idx[9]}].value = args_1_0_0.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[10]}] = ctypes.c_int(args_1_0_0.domain.ranges[0].start)
    sdfg_call_args[{idx[11]}] = ctypes.c_int(args_1_0_0.domain.ranges[1].start)
    sdfg_call_args[{idx[12]}] = ctypes.c_int(args_1_0_0.domain.ranges[2].start)
    sdfg_call_args[{idx[13]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[14]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[15]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[2])
    sdfg_call_args[{idx[16]}] = ctypes.c_int(args_1_1)
    sdfg_call_args[{idx[17]}].value = args_5.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[18]}] = ctypes.c_int(args_5.domain.ranges[0].start)
    sdfg_call_args[{idx[19]}] = ctypes.c_int(args_5.domain.ranges[1].start)
    sdfg_call_args[{idx[20]}] = ctypes.c_int(args_5.domain.ranges[2].start)
    sdfg_call_args[{idx[21]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[22]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[23]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[2])\
"""
    )


def _binding_source_cartesian_with_zero_origin(use_metrics: bool) -> str:
    # In this SDFG 'sdfg_call_args[2]' is used to collect the stencil compute time.
    # Note that the position of 'gt_compute_time' in the SDFG arguments list is
    # defined by dace, based on alphabetical order from index 0 ('a', 'b', 'gt_compute_time').
    metrics_arg_index = 2
    idx = [12, 13, 0, 3, 4, 5, 1, 6, 7, 8, 14, 2, 9, 10, 11]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, sdfg_call_args, offset_provider):
    (
        args_0,
        args_1,
        args_2,
        args_3,
        args_4,
        args_5,
    ) = args
    (
        args_0_0,
        args_0_1,
    ) = args_0
    sdfg_call_args[{idx[0]}] = ctypes.c_int(args_0_0)
    (
        args_0_1_0,
        args_0_1_1,
        args_0_1_2,
    ) = args_0_1
    sdfg_call_args[{idx[1]}] = ctypes.c_int(args_0_1_0)
    sdfg_call_args[{idx[2]}].value = args_0_1_1.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[3]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[4]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[5]}] = ctypes.c_int(args_0_1_1.__gt_buffer_info__.elem_strides[2])
    (
        args_1_0,
        args_1_1,
    ) = args_1
    (args_1_0_0,) = args_1_0
    sdfg_call_args[{idx[6]}].value = args_1_0_0.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[7]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[8]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[9]}] = ctypes.c_int(args_1_0_0.__gt_buffer_info__.elem_strides[2])
    sdfg_call_args[{idx[10]}] = ctypes.c_int(args_1_1)
    sdfg_call_args[{idx[11]}].value = args_5.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[12]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[13]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[1])
    sdfg_call_args[{idx[14]}] = ctypes.c_int(args_5.__gt_buffer_info__.elem_strides[2])\
"""
    )


def _binding_source_unstructured(use_metrics: bool) -> str:
    metrics_arg_index = 2
    idx = [0, 4, 1, 5, 6, 7, 2, 9, 8, 3, 11, 10]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, sdfg_call_args, offset_provider):
    (
        args_0,
        args_1,
    ) = args
    sdfg_call_args[{idx[0]}].value = args_0.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[1]}] = ctypes.c_int(args_0.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[2]}].value = args_1.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[3]}] = ctypes.c_int(args_1.domain.ranges[0].start)
    sdfg_call_args[{idx[4]}] = ctypes.c_int(args_1.domain.ranges[0].stop)
    sdfg_call_args[{idx[5]}] = ctypes.c_int(args_1.__gt_buffer_info__.elem_strides[0])
    table_E2V = offset_provider["E2V"]
    sdfg_call_args[{idx[6]}].value = table_E2V.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[7]}] = ctypes.c_int(table_E2V.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[8]}] = ctypes.c_int(table_E2V.__gt_buffer_info__.elem_strides[1])
    table_V2E = offset_provider["V2E"]
    sdfg_call_args[{idx[9]}].value = table_V2E.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[10]}] = ctypes.c_int(table_V2E.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[11]}] = ctypes.c_int(table_V2E.__gt_buffer_info__.elem_strides[1])\
"""
    )


def _binding_source_unstructured_with_zero_origin(use_metrics: bool) -> str:
    metrics_arg_index = 2
    idx = [0, 4, 1, 5, 6, 2, 8, 7, 3, 10, 9]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, sdfg_call_args, offset_provider):
    (
        args_0,
        args_1,
    ) = args
    sdfg_call_args[{idx[0]}].value = args_0.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[1]}] = ctypes.c_int(args_0.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[2]}].value = args_1.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[3]}] = ctypes.c_int(args_1.domain.ranges[0].stop)
    sdfg_call_args[{idx[4]}] = ctypes.c_int(args_1.__gt_buffer_info__.elem_strides[0])
    table_E2V = offset_provider["E2V"]
    sdfg_call_args[{idx[5]}].value = table_E2V.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[6]}] = ctypes.c_int(table_E2V.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[7]}] = ctypes.c_int(table_E2V.__gt_buffer_info__.elem_strides[1])
    table_V2E = offset_provider["V2E"]
    sdfg_call_args[{idx[8]}].value = table_V2E.__gt_buffer_info__.data_ptr
    sdfg_call_args[{idx[9]}] = ctypes.c_int(table_V2E.__gt_buffer_info__.elem_strides[0])
    sdfg_call_args[{idx[10]}] = ctypes.c_int(table_V2E.__gt_buffer_info__.elem_strides[1])\
"""
    )


# The difference between the two bindings versions is that one uses field domain
# with zero origin, therefore the range-start symbols are not present in the SDFG.
assert _binding_source_cartesian_with_zero_origin != _binding_source_cartesian
assert _binding_source_unstructured_with_zero_origin != _binding_source_unstructured


_dace_compile_call = dace_workflow.compilation.DaCeCompiler.__call__


def mocked_compile_call(
    self,
    inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    binding_source_ref: str,
):
    assert len(inp.library_deps) == 0

    # ignore assert statements
    binding_source_pruned = "\n".join(
        line
        for line in inp.binding_source.source_code.splitlines()
        if not line.lstrip().startswith("assert")
    )
    assert binding_source_pruned == binding_source_ref
    return _dace_compile_call(self, inp)


def mocked_compile_call_cartesian(
    self,
    inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    use_metrics: bool,
    use_zero_origin: bool,
):
    binding_ref_fun = (
        _binding_source_cartesian_with_zero_origin if use_zero_origin else _binding_source_cartesian
    )
    return mocked_compile_call(self, inp, binding_ref_fun(use_metrics))


def mocked_compile_call_unstructured(
    self,
    inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    use_metrics: bool,
    use_zero_origin: bool,
):
    binding_ref_fun = (
        _binding_source_unstructured_with_zero_origin
        if use_zero_origin
        else _binding_source_unstructured
    )
    return mocked_compile_call(self, inp, binding_ref_fun(use_metrics))


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
@pytest.mark.parametrize(
    "use_zero_origin", [False, True], ids=["no_zero_origin", "use_zero_origin"]
)
def test_cartesian_bind_sdfg(use_metrics, use_zero_origin, monkeypatch):
    M, N, K = (30, 20, 10)

    @gtx.field_operator
    def testee_op(
        a: tuple[int32, tuple[int32, cases.IJKField, int32]], b: tuple[tuple[cases.IJKField], int32]
    ) -> cases.IJKField:
        return (
            a[0] + 2 * a[1][0] + 3 * a[1][1] + 4 * b[0][0] + 5 * b[1]
        )  # skip 'a[1][2]' on purpose to cover unused scalar args

    @gtx.program
    def testee(
        a: tuple[int32, tuple[int32, cases.IJKField, int32]],
        b: tuple[tuple[cases.IJKField], int32],  # use 'b_0' to test tuple with single element
        M: int32,
        N: int32,
        K: int32,
        out: cases.IJKField,
    ):
        testee_op(a, b, out=out, domain={IDim: (1, M - 1), JDim: (2, N - 2), KDim: (3, K - 3)})

    backend = dace_runner.make_dace_backend(
        gpu=False,
        cached=False,
        auto_optimize=True,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
    )
    monkeypatch.setattr(
        dace_workflow.compilation.DaCeCompiler,
        "__call__",
        functools.partialmethod(
            mocked_compile_call_cartesian, use_metrics=use_metrics, use_zero_origin=use_zero_origin
        ),
    )

    static_args = {"M": [M], "N": [N], "K": [K]}
    program = (
        testee.with_grid_type(gtx_common.GridType.CARTESIAN)
        .with_backend(backend)
        .compile(enable_jit=False, offset_provider={}, **static_args)
    )

    test_case = cases.Case.from_cartesian_grid_descriptor(
        ffront_test_utils.simple_cartesian_grid(),
        backend=backend,
        allocator=backend,
    )

    sizes = {IDim: M, JDim: N, KDim: K}
    a = cases.allocate(test_case, testee, "a", sizes=sizes, strategy=cases.UniqueInitializer())()
    b = cases.allocate(test_case, testee, "b", sizes=sizes, strategy=cases.UniqueInitializer())()
    c = cases.allocate(test_case, testee, "out", sizes=sizes, strategy=cases.UniqueInitializer())()

    ref = c.asnumpy().copy()
    ref[1 : M - 1, 2 : N - 2, 3 : K - 3] = (
        a[0] + 2 * a[1][0] + 3 * a[1][1].asnumpy() + 4 * b[0][0].asnumpy() + 5 * b[1]
    )[1 : M - 1, 2 : N - 2, 3 : K - 3]

    program(a, b, out=c, M=M, N=N, K=K)
    assert np.all(c.asnumpy() == ref)


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
@pytest.mark.parametrize(
    "use_zero_origin", [False, True], ids=["no_zero_origin", "use_zero_origin"]
)
def test_unstructured_bind_sdfg(use_metrics, use_zero_origin, monkeypatch):
    @gtx.field_operator
    def testee_op(a: cases.VField) -> cases.VField:
        tmp = neighbor_sum(a(E2V), axis=E2VDim)
        tmp_2 = neighbor_sum(tmp(V2E), axis=V2EDim)
        return tmp_2

    @gtx.program
    def testee(a: cases.VField, b: cases.VField):
        testee_op(a, out=b)

    backend = dace_runner.make_dace_backend(
        gpu=False,
        cached=False,
        auto_optimize=True,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
    )
    monkeypatch.setattr(
        dace_workflow.compilation.DaCeCompiler,
        "__call__",
        functools.partialmethod(
            mocked_compile_call_unstructured,
            use_metrics=use_metrics,
            use_zero_origin=use_zero_origin,
        ),
    )

    SIMPLE_MESH = ffront_test_utils.simple_mesh(None)
    offset_provider = SIMPLE_MESH.offset_provider

    static_args = {}
    program = (
        testee.with_grid_type(gtx_common.GridType.UNSTRUCTURED)
        .with_backend(backend)
        .compile(enable_jit=False, offset_provider=offset_provider, **static_args)
    )

    test_case = cases.Case.from_mesh_descriptor(SIMPLE_MESH, backend=backend, allocator=backend)

    a = cases.allocate(test_case, testee, "a")()
    b = cases.allocate(test_case, testee, "b")()

    ref = np.sum(
        np.sum(a.asnumpy()[offset_provider["E2V"].asnumpy()], axis=1, initial=0)[
            offset_provider["V2E"].asnumpy()
        ],
        axis=1,
    )

    program(a, b, offset_provider=offset_provider)
    assert np.all(b.asnumpy() == ref)
