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

from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests import ffront_test_utils
from next_tests.unit_tests.test_common import IDim, JDim, KDim


_bind_func_name = "update_sdfg_args"


_bind_header = """\
import ctypes
from gt4py.next import common as gtx_common, field_utils


def _get_stride(ndarray, dim_index):
    return ndarray.strides[dim_index] // ndarray.itemsize


"""


def _binding_source(use_metrics: bool) -> str:
    # In this SDFG 'last_call_args[2]' is used to collect the stencil compute time.
    # Note that the position of 'gt_compute_time' in the SDFG arguments list is
    # defined by dace, based on alphabetical order from index 0 ('a', 'b', 'gt_compute_time').
    metrics_arg_index = 2
    idx = [21, 22, 0, 3, 4, 5, 6, 7, 8, 1, 9, 10, 11, 12, 13, 14, 23, 2, 15, 16, 17, 18, 19, 20]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
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
    last_call_args[{idx[0]}] = ctypes.c_int(args_0_0)
    (
        args_0_1_0,
        args_0_1_1,
        args_0_1_2,
    ) = args_0_1
    last_call_args[{idx[1]}] = ctypes.c_int(args_0_1_0)
    last_call_args[{idx[2]}].value = args_0_1_1._data_buffer_ptr_
    last_call_args[{idx[3]}] = ctypes.c_int(args_0_1_1.domain.ranges[0].start)
    last_call_args[{idx[4]}] = ctypes.c_int(args_0_1_1.domain.ranges[1].start)
    last_call_args[{idx[5]}] = ctypes.c_int(args_0_1_1.domain.ranges[2].start)
    last_call_args[{idx[6]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 0))
    last_call_args[{idx[7]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 1))
    last_call_args[{idx[8]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 2))
    (
        args_1_0,
        args_1_1,
    ) = args_1
    (args_1_0_0,) = args_1_0
    last_call_args[{idx[9]}].value = args_1_0_0._data_buffer_ptr_
    last_call_args[{idx[10]}] = ctypes.c_int(args_1_0_0.domain.ranges[0].start)
    last_call_args[{idx[11]}] = ctypes.c_int(args_1_0_0.domain.ranges[1].start)
    last_call_args[{idx[12]}] = ctypes.c_int(args_1_0_0.domain.ranges[2].start)
    last_call_args[{idx[13]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 0))
    last_call_args[{idx[14]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 1))
    last_call_args[{idx[15]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 2))
    last_call_args[{idx[16]}] = ctypes.c_int(args_1_1)
    last_call_args[{idx[17]}].value = args_5._data_buffer_ptr_
    last_call_args[{idx[18]}] = ctypes.c_int(args_5.domain.ranges[0].start)
    last_call_args[{idx[19]}] = ctypes.c_int(args_5.domain.ranges[1].start)
    last_call_args[{idx[20]}] = ctypes.c_int(args_5.domain.ranges[2].start)
    last_call_args[{idx[21]}] = ctypes.c_int(_get_stride(args_5.ndarray, 0))
    last_call_args[{idx[22]}] = ctypes.c_int(_get_stride(args_5.ndarray, 1))
    last_call_args[{idx[23]}] = ctypes.c_int(_get_stride(args_5.ndarray, 2))\
"""
    )


def _binding_source_with_zero_origin(use_metrics: bool) -> str:
    # In this SDFG 'last_call_args[2]' is used to collect the stencil compute time.
    # Note that the position of 'gt_compute_time' in the SDFG arguments list is
    # defined by dace, based on alphabetical order from index 0 ('a', 'b', 'gt_compute_time').
    metrics_arg_index = 2
    idx = [12, 13, 0, 3, 4, 5, 1, 6, 7, 8, 14, 2, 9, 10, 11]
    if use_metrics:
        idx = [idx + 1 if idx >= metrics_arg_index else idx for idx in idx]
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
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
    last_call_args[{idx[0]}] = ctypes.c_int(args_0_0)
    (
        args_0_1_0,
        args_0_1_1,
        args_0_1_2,
    ) = args_0_1
    last_call_args[{idx[1]}] = ctypes.c_int(args_0_1_0)
    last_call_args[{idx[2]}].value = args_0_1_1._data_buffer_ptr_
    last_call_args[{idx[3]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 0))
    last_call_args[{idx[4]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 1))
    last_call_args[{idx[5]}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 2))
    (
        args_1_0,
        args_1_1,
    ) = args_1
    (args_1_0_0,) = args_1_0
    last_call_args[{idx[6]}].value = args_1_0_0._data_buffer_ptr_
    last_call_args[{idx[7]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 0))
    last_call_args[{idx[8]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 1))
    last_call_args[{idx[9]}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 2))
    last_call_args[{idx[10]}] = ctypes.c_int(args_1_1)
    last_call_args[{idx[11]}].value = args_5._data_buffer_ptr_
    last_call_args[{idx[12]}] = ctypes.c_int(_get_stride(args_5.ndarray, 0))
    last_call_args[{idx[13]}] = ctypes.c_int(_get_stride(args_5.ndarray, 1))
    last_call_args[{idx[14]}] = ctypes.c_int(_get_stride(args_5.ndarray, 2))\
"""
    )


# The difference between the two bindings versions is that one uses field domain
# with zero origin, therefore the range-start symbols are not present in the SDFG.
assert _binding_source_with_zero_origin != _binding_source


_dace_compile_call = dace_workflow.compilation.DaCeCompiler.__call__


def mocked_compile_call(
    self,
    inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    use_metrics: bool,
    use_zero_origin: bool,
):
    assert len(inp.library_deps) == 0

    # ignore assert statements
    binding_source_pruned = "\n".join(
        line
        for line in inp.binding_source.source_code.splitlines()
        if not line.lstrip().startswith("assert")
    )

    binding_source_ref = _binding_source_with_zero_origin if use_zero_origin else _binding_source
    assert binding_source_pruned == binding_source_ref(use_metrics)
    return _dace_compile_call(self, inp)


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
@pytest.mark.parametrize(
    "use_zero_origin", [False, True], ids=["no_zero_origin", "use_zero_origin"]
)
def test_bind_sdfg(use_metrics, use_zero_origin, monkeypatch):
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
            mocked_compile_call, use_metrics=use_metrics, use_zero_origin=use_zero_origin
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
