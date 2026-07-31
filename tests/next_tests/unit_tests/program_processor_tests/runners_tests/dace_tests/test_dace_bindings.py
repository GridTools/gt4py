# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest


dace = pytest.importorskip("dace")

from gt4py import next as gtx
from gt4py.eve import codegen
from gt4py.next import common as gtx_common, config as gtx_config, int32, neighbor_sum
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import code_specs, stages
from gt4py.next.otf.binding import interface
from gt4py.next.program_processors.runners import dace as dace_runner
from gt4py.next.program_processors.runners.dace import workflow as dace_workflow
from gt4py.next.program_processors.runners.dace.workflow import (
    bindings as dace_wf_bindings,
    common as dace_wf_common,
)
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests import cases, cases_utils
from next_tests.integration_tests.cases import E2V, V2E, E2VDim, V2EDim
from next_tests.unit_tests.test_common import IDim, JDim, KDim


_bind_func_name = "update_sdfg_args"

_METRIC_LEVEL = dace_wf_common.SDFG_ARG_METRIC_LEVEL
_COMPUTE_TIME = dace_wf_common.SDFG_ARG_METRIC_COMPUTE_TIME

_float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
_int32_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
_bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
_ij_field_type = ts.FieldType(dims=[IDim, JDim], dtype=_float64_type)
_zero_dim_field_type = ts.FieldType(dims=[], dtype=_float64_type)


def _make_testee_program() -> itir.Program:
    """A program signature covering fields, scalars, bools, zero-dimensional fields and nested tuples.

    Note that the bindings generator only inspects the parameters of the program,
    thus the body can be left empty.
    """
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[
            itir.Sym(id="a", type=_ij_field_type),
            itir.Sym(id="s", type=_int32_type),
            itir.Sym(id="flag", type=_bool_type),
            itir.Sym(id="zd", type=_zero_dim_field_type),
            itir.Sym(
                id="t",
                type=ts.TupleType(
                    types=[_int32_type, ts.TupleType(types=[_ij_field_type]), _ij_field_type]
                ),
            ),
        ],
        declarations=[],
        body=[],
    )


def _make_testee_arguments() -> tuple[tuple, gtx_common.OffsetProvider]:
    """Runtime arguments matching the signature of `_make_testee_program()`."""

    def make_ij_field(seed: int) -> gtx.Field:
        # Use a non-zero origin such that we can check that the correct one is passed on.
        domain = gtx_common.domain({IDim: (1, 5), JDim: (2, 8)})
        return gtx.as_field(domain, np.arange(seed, seed + 24, dtype=np.float64).reshape(4, 6))

    args = (
        make_ij_field(0),
        np.int32(42),
        np.bool_(True),
        gtx.as_field([], np.asarray(41.5)),
        (np.int32(7), (make_ij_field(100),), make_ij_field(200)),
    )
    offset_provider = {
        "V2E": gtx.as_connectivity(
            domain={cases.Vertex: 2, V2EDim: 4},
            codomain=cases.Edge,
            data=np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=gtx.IndexType),
            skip_value=None,
        )
    }
    return args, offset_provider


def _create_testee_bindings(
    use_metrics: bool, eval_mode: bool = False, backend: str = "dace"
) -> str:
    _, offset_provider = _make_testee_arguments()
    return dace_wf_bindings._create_sdfg_bindings(
        prog=_make_testee_program(),
        offset_provider_type=gtx_common.offset_provider_to_type(offset_provider),
        bind_func_name=_bind_func_name,
        use_metrics=use_metrics,
        eval_mode=eval_mode,
        backend=backend,
    )


def _compile_bindings(binding_source: str):
    """Turn the generated source into a callable, as `CompiledDaceProgram` does."""
    namespace: dict = {}
    exec(binding_source, namespace)
    return namespace[_bind_func_name]


def _expected_testee_binding_source(use_metrics: bool) -> str:
    metric_args = f"{_METRIC_LEVEL}, {_COMPUTE_TIME}, " if use_metrics else ""
    return f"""\
def {_bind_func_name}(
    args,
    offset_provider,
    {_METRIC_LEVEL},
    {_COMPUTE_TIME},
):
    __gtx_expanded_names_a, __gtx_expanded_names_s, __gtx_expanded_names_flag, __gtx_expanded_names_zd, __gtx_expanded_names_t, = args
    __gtx_expanded_names_t_0, __gtx_expanded_names_t_1, __gtx_expanded_names_t_2, = __gtx_expanded_names_t
    __gtx_expanded_names_t_1_0, = __gtx_expanded_names_t_1
    return (
        (__gtx_expanded_names_a.ndarray, (__gtx_expanded_names_a.__dace_origin__)),
        (__gtx_expanded_names_s),
        (__gtx_expanded_names_flag),
        __gtx_expanded_names_zd.as_scalar(),
        (
            (__gtx_expanded_names_t_0),
            ((__gtx_expanded_names_t_1_0.ndarray, (__gtx_expanded_names_t_1_0.__dace_origin__)),),
            (__gtx_expanded_names_t_2.ndarray, (__gtx_expanded_names_t_2.__dace_origin__)),
        ),
        (offset_provider['V2E'].ndarray, (0, 0)),
        {metric_args}
    )
"""


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
def test_create_sdfg_bindings_source(use_metrics):
    """The generated source is fully unrolled and only depends on the program parameters."""
    binding_source = _create_testee_bindings(use_metrics)

    assert codegen.format_python_source(binding_source) == codegen.format_python_source(
        _expected_testee_binding_source(use_metrics)
    )


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
def test_binding_function_processes_arguments(use_metrics):
    """The compiled binding function translates the arguments for `user_bind_call()`."""
    args, offset_provider = _make_testee_arguments()
    a, s, flag, zd, t = args
    binding_function = _compile_bindings(_create_testee_bindings(use_metrics))

    metric_level = 2
    compute_time = np.zeros(1, dtype=np.float64)
    processed = binding_function(args, offset_provider, metric_level, compute_time)

    assert isinstance(processed, tuple)
    assert len(processed) == (8 if use_metrics else 6)

    # A field is passed as a `(buffer, origin)` pair; the buffer is forwarded, not copied.
    assert processed[0][0] is a.ndarray
    assert processed[0][1] == a.__dace_origin__ == (1, 2)

    # Scalars are passed through unchanged.
    assert processed[1] == s
    assert processed[2] == flag

    # A zero-dimensional field is passed as a scalar.
    assert processed[3] == zd.as_scalar()

    # A tuple argument keeps its structure, with every member processed.
    processed_t = processed[4]
    assert processed_t[0] == t[0]
    assert processed_t[1][0][0] is t[1][0].ndarray and processed_t[1][0][1] == (1, 2)
    assert processed_t[2][0] is t[2].ndarray and processed_t[2][1] == (1, 2)

    # An offset provider is passed as a `(buffer, origin)` pair with zero origin.
    assert processed[5][0] is offset_provider["V2E"].ndarray
    assert processed[5][1] == (0, 0)

    if use_metrics:
        assert processed[6] == metric_level
        assert processed[7] is compute_time


def _assert_same_processing(generated_result, eval_result) -> None:
    assert type(generated_result) is type(eval_result), (
        f"{type(generated_result)} vs. {type(eval_result)}"
    )
    if isinstance(generated_result, tuple):
        assert len(generated_result) == len(eval_result)
        for generated_member, eval_member in zip(generated_result, eval_result):
            _assert_same_processing(generated_member, eval_member)
    elif isinstance(generated_result, np.ndarray):
        # Buffers must be forwarded, not copied.
        assert generated_result is eval_result
    else:
        assert generated_result == eval_result


@pytest.mark.parametrize("use_metrics", [False, True], ids=["no_metrics", "use_metrics"])
def test_eval_mode_matches_generated_mode(use_metrics):
    """Generation mode and eval mode must process the arguments in exactly the same way."""
    args, offset_provider = _make_testee_arguments()
    generated_mode_function = _compile_bindings(
        _create_testee_bindings(use_metrics, eval_mode=False)
    )
    eval_mode_function = _compile_bindings(_create_testee_bindings(use_metrics, eval_mode=True))

    metric_level = 2
    compute_time = np.zeros(1, dtype=np.float64)
    _assert_same_processing(
        generated_mode_function(args, offset_provider, metric_level, compute_time),
        eval_mode_function(args, offset_provider, metric_level, compute_time),
    )


def test_create_sdfg_bindings_gtfn():
    """The experimental GTFN flavour differs from dace in origin handling and casts."""
    binding_source = _create_testee_bindings(use_metrics=False, backend="gtfn")

    # GTFN uses the negated origin, i.e. `__gt_origin__` instead of `__dace_origin__`.
    assert "__gt_origin__" in binding_source
    assert "__dace_origin__" not in binding_source
    # GTFN expects a zero-dimensional field as a field, not as a scalar.
    assert ".as_scalar()" not in binding_source
    # GTFN requires bool scalars to be cast to Python bools.
    assert "bool(__gtx_expanded_names_flag)" in binding_source


def test_create_sdfg_bindings_gtfn_metrics_not_implemented():
    with pytest.raises(NotImplementedError):
        _create_testee_bindings(use_metrics=True, backend="gtfn")


def test_bind_sdfg_stage():
    """`bind_sdfg()` splits the translation output into program and binding source."""
    binding_code = f"def {_bind_func_name}(args, offset_provider, level, time):\n    return ()"
    jsdfg = {"attributes": {"name": "testee"}}
    inp = stages.ProgramSource(
        entry_point=interface.Function("testee", parameters=()),
        source_code=(binding_code, jsdfg),  # type: ignore[arg-type] # The translation stage also puts a tuple on the `str` typed field.
        library_deps=(),
        code_spec=code_specs.SDFGCodeSpec(),
    )

    ext = dace_wf_bindings.bind_sdfg(inp, _bind_func_name)

    assert ext.program_source.source_code is jsdfg
    assert ext.program_source.entry_point == inp.entry_point
    assert ext.binding_source.source_code == binding_code
    assert ext.binding_source.library_deps == tuple()


_dace_compile_call = dace_workflow.compilation.DaCeCompiler.__call__


def _capture_binding_source(monkeypatch) -> dict:
    """Monkeypatch `DaCeCompiler.__call__` to capture the binding source it receives.

    Also force in-process compilation, otherwise the compilation would run in a
    worker process where the monkeypatch is not effective.
    """
    monkeypatch.setattr(gtx_config, "BUILD_JOBS", 0)
    captured: dict = {}

    def mocked_compile_call(self, inp):
        captured["binding_source"] = inp.binding_source.source_code
        return _dace_compile_call(self, inp)

    monkeypatch.setattr(dace_workflow.compilation.DaCeCompiler, "__call__", mocked_compile_call)
    return captured


def _expected_cartesian_binding_source(use_metrics: bool) -> str:
    metric_args = f"{_METRIC_LEVEL}, {_COMPUTE_TIME}, " if use_metrics else ""
    return f"""\
def {_bind_func_name}(
    args,
    offset_provider,
    {_METRIC_LEVEL},
    {_COMPUTE_TIME},
):
    __gtx_expanded_names_a, __gtx_expanded_names_b, __gtx_expanded_names_M, __gtx_expanded_names_N, __gtx_expanded_names_K, __gtx_expanded_names_out, = args
    __gtx_expanded_names_a_0, __gtx_expanded_names_a_1, = __gtx_expanded_names_a
    __gtx_expanded_names_a_1_0, __gtx_expanded_names_a_1_1, __gtx_expanded_names_a_1_2, = __gtx_expanded_names_a_1
    __gtx_expanded_names_b_0, __gtx_expanded_names_b_1, = __gtx_expanded_names_b
    __gtx_expanded_names_b_0_0, = __gtx_expanded_names_b_0
    return (
        (
            (__gtx_expanded_names_a_0),
            (
                (__gtx_expanded_names_a_1_0),
                (__gtx_expanded_names_a_1_1.ndarray, (__gtx_expanded_names_a_1_1.__dace_origin__)),
                (__gtx_expanded_names_a_1_2),
            ),
        ),
        (
            ((__gtx_expanded_names_b_0_0.ndarray, (__gtx_expanded_names_b_0_0.__dace_origin__)),),
            (__gtx_expanded_names_b_1),
        ),
        (__gtx_expanded_names_M),
        (__gtx_expanded_names_N),
        (__gtx_expanded_names_K),
        (__gtx_expanded_names_out.ndarray, (__gtx_expanded_names_out.__dace_origin__)),
        {metric_args}
    )
"""


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

    @gtx.program(enable_jit=False)
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
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
    )
    captured = _capture_binding_source(monkeypatch)

    test_case = cases.Case.from_cartesian_grid_descriptor(
        cases_utils.simple_cartesian_grid(),
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

    static_args = {"M": [M], "N": [N], "K": [K]}
    program = (
        testee.with_grid_type(gtx_common.GridType.CARTESIAN)
        .with_backend(backend)
        .compile(offset_provider={}, **static_args)
    )
    program(a, b, out=c, M=M, N=N, K=K)
    assert np.all(c.asnumpy() == ref)

    # The binding source only depends on the program parameters and the offset
    # providers; in particular it is independent of `use_zero_origin`, since the
    # origin is always passed and, if not needed, ignored on the SDFG side.
    assert codegen.format_python_source(captured["binding_source"]) == codegen.format_python_source(
        _expected_cartesian_binding_source(use_metrics)
    )


def _expected_unstructured_binding_source(
    use_metrics: bool, offset_provider: gtx_common.OffsetProvider
) -> str:
    metric_args = f"{_METRIC_LEVEL}, {_COMPUTE_TIME}, " if use_metrics else ""
    offset_provider_args = "".join(
        f"(offset_provider['{name}'].ndarray, (0, 0)), " for name in offset_provider
    )
    return f"""\
def {_bind_func_name}(
    args,
    offset_provider,
    {_METRIC_LEVEL},
    {_COMPUTE_TIME},
):
    __gtx_expanded_names_a, __gtx_expanded_names_b, = args
    return (
        (__gtx_expanded_names_a.ndarray, (__gtx_expanded_names_a.__dace_origin__)),
        (__gtx_expanded_names_b.ndarray, (__gtx_expanded_names_b.__dace_origin__)),
        {offset_provider_args}
        {metric_args}
    )
"""


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

    @gtx.program(enable_jit=False)
    def testee(a: cases.VField, b: cases.VField):
        testee_op(a, out=b)

    backend = dace_runner.make_dace_backend(
        gpu=False,
        use_metrics=use_metrics,
        use_zero_origin=use_zero_origin,
    )
    captured = _capture_binding_source(monkeypatch)

    SIMPLE_MESH = cases_utils.simple_mesh(None)
    offset_provider = SIMPLE_MESH.offset_provider

    test_case = cases.Case.from_mesh_descriptor(SIMPLE_MESH, backend=backend, allocator=backend)

    a = cases.allocate(test_case, testee, "a")()
    b = cases.allocate(test_case, testee, "b")()

    ref = np.sum(
        np.sum(a.asnumpy()[offset_provider["E2V"].asnumpy()], axis=1, initial=0)[
            offset_provider["V2E"].asnumpy()
        ],
        axis=1,
    )

    program = (
        testee.with_grid_type(gtx_common.GridType.UNSTRUCTURED)
        .with_backend(backend)
        .compile(offset_provider=offset_provider)
    )
    program(a, b, offset_provider=offset_provider)
    assert np.all(b.asnumpy() == ref)

    assert codegen.format_python_source(captured["binding_source"]) == codegen.format_python_source(
        _expected_unstructured_binding_source(use_metrics, offset_provider)
    )
