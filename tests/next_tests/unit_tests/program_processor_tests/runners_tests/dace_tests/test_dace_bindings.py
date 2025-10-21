# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test the bindings stage of the dace backend workflow."""

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
    i = (
        1 if use_metrics else 0
    )  # 'last_call_args[0]' is reserved for '__return' value, which is used to collect the stencil compute time
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    args_0, args_1, args_2, args_3, args_4, args_5 = args
    args_0_0, args_0_1 = args_0
    last_call_args[{i + 21}] = ctypes.c_int(args_0_0)
    args_0_1_0, args_0_1_1, args_0_1_2 = args_0_1
    last_call_args[{i + 22}] = ctypes.c_int(args_0_1_0)
    last_call_args[{i + 0}].value = args_0_1_1.data_ptr()
    last_call_args[{i + 3}] = ctypes.c_int(args_0_1_1.domain.ranges[0].start)
    last_call_args[{i + 4}] = ctypes.c_int(args_0_1_1.domain.ranges[1].start)
    last_call_args[{i + 5}] = ctypes.c_int(args_0_1_1.domain.ranges[2].start)
    last_call_args[{i + 6}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 0))
    last_call_args[{i + 7}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 1))
    last_call_args[{i + 8}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 2))
    args_1_0, args_1_1 = args_1
    args_1_0_0 = args_1_0[0]
    last_call_args[{i + 1}].value = args_1_0_0.data_ptr()
    last_call_args[{i + 9}] = ctypes.c_int(args_1_0_0.domain.ranges[0].start)
    last_call_args[{i + 10}] = ctypes.c_int(args_1_0_0.domain.ranges[1].start)
    last_call_args[{i + 11}] = ctypes.c_int(args_1_0_0.domain.ranges[2].start)
    last_call_args[{i + 12}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 0))
    last_call_args[{i + 13}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 1))
    last_call_args[{i + 14}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 2))
    last_call_args[{i + 23}] = ctypes.c_int(args_1_1)
    last_call_args[{i + 2}].value = args_5.data_ptr()
    last_call_args[{i + 15}] = ctypes.c_int(args_5.domain.ranges[0].start)
    last_call_args[{i + 16}] = ctypes.c_int(args_5.domain.ranges[1].start)
    last_call_args[{i + 17}] = ctypes.c_int(args_5.domain.ranges[2].start)
    last_call_args[{i + 18}] = ctypes.c_int(_get_stride(args_5.ndarray, 0))
    last_call_args[{i + 19}] = ctypes.c_int(_get_stride(args_5.ndarray, 1))
    last_call_args[{i + 20}] = ctypes.c_int(_get_stride(args_5.ndarray, 2))\
"""
    )


def _binding_source_with_zero_origin(use_metrics: bool) -> str:
    i = (
        1 if use_metrics else 0
    )  # 'last_call_args[0]' is reserved for '__return' value, which is used to collect the stencil compute time
    return (
        _bind_header
        + f"""\
def {_bind_func_name}(device, sdfg_argtypes, args, last_call_args):
    args_0, args_1, args_2, args_3, args_4, args_5 = args
    args_0_0, args_0_1 = args_0
    last_call_args[{i + 12}] = ctypes.c_int(args_0_0)
    args_0_1_0, args_0_1_1, args_0_1_2 = args_0_1
    last_call_args[{i + 13}] = ctypes.c_int(args_0_1_0)
    last_call_args[{i + 0}].value = args_0_1_1.data_ptr()
    last_call_args[{i + 3}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 0))
    last_call_args[{i + 4}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 1))
    last_call_args[{i + 5}] = ctypes.c_int(_get_stride(args_0_1_1.ndarray, 2))
    args_1_0, args_1_1 = args_1
    args_1_0_0 = args_1_0[0]
    last_call_args[{i + 1}].value = args_1_0_0.data_ptr()
    last_call_args[{i + 6}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 0))
    last_call_args[{i + 7}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 1))
    last_call_args[{i + 8}] = ctypes.c_int(_get_stride(args_1_0_0.ndarray, 2))
    last_call_args[{i + 14}] = ctypes.c_int(args_1_1)
    last_call_args[{i + 2}].value = args_5.data_ptr()
    last_call_args[{i + 9}] = ctypes.c_int(_get_stride(args_5.ndarray, 0))
    last_call_args[{i + 10}] = ctypes.c_int(_get_stride(args_5.ndarray, 1))
    last_call_args[{i + 11}] = ctypes.c_int(_get_stride(args_5.ndarray, 2))\
"""
    )


# The difference between the two bindings versions is that one uses field domain
# with zero origin, therefore the range-start symbols are not present in the SDFG.
assert _binding_source_with_zero_origin != _binding_source


@pytest.mark.parametrize(
    "static_domain_config",
    [(False, _binding_source), (True, _binding_source_with_zero_origin)],
)
@pytest.mark.parametrize("use_metrics", [False, True])
def test_bind_sdfg(static_domain_config, use_metrics, monkeypatch):
    M, N, K = (30, 20, 10)
    use_zero_origin, binding_source_ref = static_domain_config

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

    compilable_source: (
        stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python] | None
    ) = None

    dace_compile_call = dace_workflow.compilation.DaCeCompiler.__call__

    def mocked_compile_call(
        self,
        inp: stages.CompilableSource[languages.SDFG, languages.LanguageSettings, languages.Python],
    ):
        nonlocal compilable_source
        compilable_source = inp
        return dace_compile_call(self, inp)

    monkeypatch.setattr(dace_workflow.compilation.DaCeCompiler, "__call__", mocked_compile_call)

    backend = dace_runner.make_dace_backend(
        gpu=False,
        cached=False,
        auto_optimize=True,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
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

    program(a, b, out=c, M=M, N=N, K=K)
    assert np.all(c.asnumpy() == (a[0] + 2 * a[1][0] + 3 * a[1][1] + 4 * b[0][0] + 5 * b[1]))

    assert compilable_source is not None
    assert len(compilable_source.library_deps) == 0

    # ignore assert statements
    binding_source_pruned = "\n".join(
        line
        for line in compilable_source.binding_source.source_code.splitlines()
        if not line.lstrip().startswith("assert")
    )

    assert binding_source_pruned == binding_source_ref(use_metrics)
