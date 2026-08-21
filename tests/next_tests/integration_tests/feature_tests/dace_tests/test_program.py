# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.otf import arguments, workflow

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import (
    Cell,
    E2V,
    Edge,
    IDim,
    JDim,
    KDim,
    Vertex,
    mesh_descriptor,  # noqa: F401
)


dace = pytest.importorskip("dace")

from gt4py.next.program_processors.runners import dace as dace_backends


# Override the exec_alloc_descriptor with a custom Backend,
# see https://docs.pytest.org/en/latest/how-to/fixtures.html#override-a-fixture-on-a-test-module-level
@pytest.fixture(
    params=[
        pytest.param(dace_backends.run_dace_cpu, marks=pytest.mark.requires_dace),
        pytest.param(
            dace_backends.run_dace_gpu, marks=(pytest.mark.requires_gpu, pytest.mark.requires_dace)
        ),
    ]
)
def exec_alloc_descriptor(request):
    yield request.param


@pytest.fixture
def unstructured(request, exec_alloc_descriptor, mesh_descriptor):  # noqa: F811
    yield cases.Case(
        backend=exec_alloc_descriptor,
        offset_provider=mesh_descriptor.offset_provider,
        default_sizes={
            Vertex: mesh_descriptor.num_vertices,
            Edge: mesh_descriptor.num_edges,
            Cell: mesh_descriptor.num_cells,
        },
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )


def test_halo_exchange_helper_attrs(unstructured):
    local_int = gtx.int

    @gtx.field_operator(backend=unstructured.backend)
    def testee_op(
        a: gtx.Field[[Vertex, KDim], gtx.int],
    ) -> gtx.Field[[Vertex, KDim], gtx.int]:
        return a + local_int(10)

    @gtx.program(backend=unstructured.backend)
    def testee_prog(
        a: gtx.Field[[Vertex, KDim], gtx.int],
        b: gtx.Field[[Vertex, KDim], gtx.int],
        c: gtx.Field[[Vertex, KDim], gtx.int],
    ):
        testee_op(b, out=c)
        testee_op(a, out=b)

    dace_storage_type = (
        dace.StorageType.GPU_Global
        if unstructured.backend == dace_backends.run_dace_gpu
        else dace.StorageType.Default
    )

    rows = dace.symbol("rows")
    cols = dace.symbol("cols")

    @dace.program
    def testee_dace(
        a: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        b: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
        c: dace.data.Array(dtype=dace.int64, shape=(rows, cols), storage=dace_storage_type),
    ):
        testee_prog(a, b, c)

    # if simplify=True, DaCe might inline the nested SDFG coming from Program.__sdfg__,
    # effectively erasing the attributes we want to test for here
    sdfg = testee_dace.to_sdfg(simplify=False)

    testee = next(
        subgraph for subgraph in sdfg.all_sdfgs_recursive() if subgraph.name == "testee_prog"
    )

    assert testee.gt4py_program_input_fields == {"a": Vertex, "b": Vertex}
    assert testee.gt4py_program_output_fields == {"b": Vertex, "c": Vertex}


@gtx.field_operator
def _shift_over_e2v(a: gtx.Field[gtx.Dims[Vertex], gtx.float64]):
    return a(E2V[0])


@gtx.program
def _unstructured_prog(
    a: gtx.Field[gtx.Dims[Vertex], gtx.float64], b: gtx.Field[gtx.Dims[Edge], gtx.float64]
):
    _shift_over_e2v(a, out=b)


@gtx.field_operator
def _add_ten(a: gtx.Field[[IDim, KDim], gtx.float64]) -> gtx.Field[[IDim, KDim], gtx.float64]:
    return a + 10.0


@gtx.program
def _cartesian_prog(
    a: gtx.Field[[IDim, KDim], gtx.float64], b: gtx.Field[[IDim, KDim], gtx.float64]
):
    _add_ten(a, out=b)


@pytest.mark.parametrize("with_connectivities", [False, True], ids=["cartesian", "unstructured"])
def test_sdfg_conversion_does_not_mutate_gtir_cache(
    exec_alloc_descriptor,
    mesh_descriptor,  # noqa: F811
    with_connectivities,
):
    """Regression test: `__sdfg__` must leave the `past_to_itir` cache entry pristine.

    The SDFG conversion runs the field-view transforms itself (they need the runtime
    connectivity tables) and used to write the result back into the frozen stage
    returned by the in-memory-cached `past_to_itir` step, together with replacing
    the runtime connectivity tables in its args by the mere offset provider *types*.
    Because a plain `compile()` of the same program hits the very same cache entry,
    every later consumer saw an already-transformed program without its neighbor
    tables, which breaks domain inference for unstructured programs.
    """
    if with_connectivities:
        program = _unstructured_prog.with_compilation_options(
            connectivities=mesh_descriptor.offset_provider
        )
    else:
        program = _cartesian_prog
    program = program.with_backend(exec_alloc_descriptor)
    offset_provider = program.compilation_options.connectivities or {}

    # The very input `__sdfg__` builds internally, so it shares its cache entry.
    stage_input = workflow.ProgramWithArgs(
        definition=program.past_stage,
        args=arguments.CompileTimeArgs(
            args=tuple(p.type for p in program.past_stage.past_node.params),
            kwargs={},
            column_axis=None,
            offset_provider=offset_provider,
            argument_descriptor_contexts={},
        ),
    )
    past_to_itir = program.backend.frontend.past_to_itir
    cached_stage = past_to_itir(stage_input)
    definition_before = str(cached_stage.definition)
    offset_provider_before = cached_stage.args.offset_provider

    program.__sdfg__()
    program.__sdfg__()

    assert past_to_itir(stage_input) is cached_stage  # the entry is really cached
    # Neither the lowered program nor the connectivity tables in its args were
    # replaced: the entry is still what an unrelated `compile()` of the same
    # program expects to get back.
    assert str(cached_stage.definition) == definition_before
    assert cached_stage.args.offset_provider is offset_provider_before
    if with_connectivities:
        assert all(
            common.is_neighbor_table(connectivity)
            for connectivity in cached_stage.args.offset_provider.values()
        )
