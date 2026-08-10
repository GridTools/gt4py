# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`GT4PyWriteBackBufferElimination` on lowering-produced input.

The transformation tests build their SDFGs by hand. These tests drive it from real
GT4Py programs, so that the pattern it matches stays reachable from what the
lowering emits, and so that the assumption it makes about Views stays true.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

dace = pytest.importorskip("dace")

from dace import data as dace_data

import gt4py.next as gtx
from gt4py.next import constructors, neighbor_sum
from gt4py.next.otf import runners
from gt4py.next.program_processors.runners import dace as gtx_dace
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations


IDim = gtx.Dimension("I")
I_SIZE = 8

Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

C2E_TABLE = np.array(
    [
        [0, 10, 3, 9],
        [1, 11, 4, 10],
        [3, 13, 6, 12],
        [4, 14, 7, 13],
        [6, 16, 0, 15],
        [7, 17, 1, 16],
        [2, 9, 5, 11],
        [5, 12, 8, 14],
    ],
    dtype=gtx.IndexType,
)
N_CELLS, N_EDGES = 8, 18


# The transformation only matches a transient that is written back into a program
#  output and is additionally consumed. `base` below is such a transient: it is a
#  program output and is read again inside the branch. The branch matters as well,
#  because it is what leaves the producer of the transient in a state of its own,
#  so the write back copy ends up in a later state.


@gtx.field_operator
def _dual_role_output(inp: gtx.Field[[IDim], gtx.float64], flag: gtx.int32):
    base = inp + 1.0
    if flag == 1:
        scaled = base * 2.0
    else:
        scaled = base * 3.0
    return base, scaled


@gtx.program
def dual_role_output_program(
    inp: gtx.Field[[IDim], gtx.float64],
    base_out: gtx.Field[[IDim], gtx.float64],
    scaled_out: gtx.Field[[IDim], gtx.float64],
    flag: gtx.int32,
):
    _dual_role_output(
        inp,
        flag,
        out=(base_out, scaled_out),
        # The bounds have to be literals: the transformation only matches if the copy
        #  covers the transient in full, which a symbolic domain does not express.
        domain={IDim: (0, 8)},
    )


@gtx.field_operator
def _dual_role_output_reduced(inp: gtx.Field[[Edge], gtx.float64], flag: gtx.int32):
    base = inp + 1.0
    if flag == 1:
        reduced = neighbor_sum(base(C2E), axis=C2EDim)
    else:
        reduced = neighbor_sum(base(C2E) * 2.0, axis=C2EDim)
    return base, reduced


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def dual_role_output_reduced_program(
    inp: gtx.Field[[Edge], gtx.float64],
    base_out: gtx.Field[[Edge], gtx.float64],
    reduced_out: gtx.Field[[Cell], gtx.float64],
    flag: gtx.int32,
):
    _dual_role_output_reduced(
        inp,
        flag,
        out=(base_out, reduced_out),
        domain=({Edge: (0, 18)}, {Cell: (0, 8)}),
    )


@pytest.fixture
def uncached_dace_cpu():
    """`run_dace_cpu` without the persistent translation cache.

    The cache stores the optimized SDFG, so a second run of these tests would replay
    it and never call the transformation.
    """
    executor = gtx_dace.run_dace_cpu.executor
    return dataclasses.replace(
        gtx_dace.run_dace_cpu,
        executor=dataclasses.replace(executor, translation=executor.translation.step),
    )


@pytest.fixture
def in_process_compilation(monkeypatch):
    """Compile in the calling process so that monkeypatched passes are visible."""
    from gt4py.next import config as gtx_config

    monkeypatch.setattr(gtx_config, "BUILD_JOBS_MODE", gtx_config.BuildJobsMode.SERIAL)
    runners.reset_default_runner()
    yield
    runners.reset_default_runner()


@pytest.fixture
def removed_buffers(monkeypatch):
    """Collects the names `GT4PyWriteBackBufferElimination` reports as removed."""
    removed: list[str] = []
    original = gtx_transformations.GT4PyWriteBackBufferElimination.apply_pass

    def apply_pass(self, sdfg, pipeline_results):
        result = original(self, sdfg, pipeline_results)
        if result:
            removed.extend(result)
        return result

    monkeypatch.setattr(
        gtx_transformations.GT4PyWriteBackBufferElimination, "apply_pass", apply_pass
    )
    return removed


def _view_targets(sdfg: dace.SDFG) -> dict[str, list[str]]:
    """Maps each viewed data name to the Views of it, at any nesting or scope level."""
    targets: dict[str, list[str]] = {}
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for node in state.data_nodes():
                if not isinstance(node.desc(nsdfg), dace_data.View):
                    continue
                try:
                    viewed = gtx_transformations.utils.track_view(node, state, nsdfg).data
                except Exception:  # noqa: BLE001 [blind-except]
                    # The Views the lowering builds inside a Map are fed through the
                    #  MapEntry and carry no `views` connector, so `track_view` cannot
                    #  resolve them. Their incoming Memlet names the data they view.
                    sources = {
                        edge.data.data
                        for edge in state.in_edges(node)
                        if edge.data.data not in (None, node.data)
                    } or {
                        edge.data.data
                        for edge in state.out_edges(node)
                        if edge.data.data not in (None, node.data)
                    }
                    if len(sources) != 1:
                        continue
                    viewed = next(iter(sources))
                targets.setdefault(viewed, []).append(node.data)
    return targets


@pytest.fixture
def selected_candidates(monkeypatch):
    """Records, per selected transient, the Views of it that exist at that moment."""
    seen: dict[str, list[str]] = {}
    original = gtx_transformations.GT4PyWriteBackBufferElimination._find_candidates

    def find_candidates(self, sdfg, reachable):
        candidates = original(self, sdfg, reachable)
        targets = _view_targets(sdfg)
        for name, _, _ in candidates:
            seen[name] = targets.get(name, [])
        return candidates

    monkeypatch.setattr(
        gtx_transformations.GT4PyWriteBackBufferElimination,
        "_find_candidates",
        find_candidates,
    )
    return seen


@pytest.fixture
def view_targets_after_lowering(monkeypatch):
    """`_view_targets` of the freshly lowered SDFG, before any transformation."""
    targets: list[dict[str, list[str]]] = []
    original = gtx_transformations.gt_auto_optimize

    def gt_auto_optimize(sdfg, *args, **kwargs):
        targets.append(_view_targets(sdfg))
        return original(sdfg, *args, **kwargs)

    monkeypatch.setattr(
        gtx_transformations.auto_optimize, "gt_auto_optimize", gt_auto_optimize
    )
    monkeypatch.setattr(gtx_transformations, "gt_auto_optimize", gt_auto_optimize)
    return targets


@pytest.mark.parametrize("flag", [0, 1])
def test_write_back_buffer_elimination_from_lowering(
    uncached_dace_cpu, in_process_compilation, removed_buffers, flag
):
    inp = gtx.as_field((IDim,), np.arange(I_SIZE, dtype=np.float64))
    base_out = gtx.as_field((IDim,), np.zeros(I_SIZE))
    scaled_out = gtx.as_field((IDim,), np.zeros(I_SIZE))

    dual_role_output_program.with_backend(uncached_dace_cpu)(
        inp, base_out, scaled_out, flag, offset_provider={}
    )

    assert removed_buffers, "GT4PyWriteBackBufferElimination did not match"

    expected_base = inp.asnumpy() + 1.0
    np.testing.assert_allclose(base_out.asnumpy(), expected_base)
    np.testing.assert_allclose(
        scaled_out.asnumpy(), expected_base * (2.0 if flag == 1 else 3.0)
    )


def test_write_back_buffer_elimination_selects_unviewed_data(
    uncached_dace_cpu,
    in_process_compilation,
    removed_buffers,
    selected_candidates,
    view_targets_after_lowering,
):
    """The transformation redirects accesses of `T` to `G` and then propagates the
    strides of `G`, but `gt_propagate_strides_of` only descends into NestedSDFGs. A
    View of `T` would keep the strides it inherited from the contiguous `T` and
    silently read `G` wrongly. This holds today only because the Views the lowering
    emits for neighbour reductions are gone by the time the transformation runs.

    The reduction below reads the selected transient, so the lowering does build a
    View of it. That the same View resolution finds Views right after lowering is
    asserted too, to keep the check on the selected candidates from passing
    vacuously.
    """
    offset_provider = {
        "C2E": constructors.as_connectivity(
            domain={Cell: N_CELLS, C2EDim: 4},
            codomain=Edge,
            data=C2E_TABLE,
            skip_value=None,
        )
    }
    inp = gtx.as_field((Edge,), np.arange(N_EDGES, dtype=np.float64))
    base_out = gtx.as_field((Edge,), np.zeros(N_EDGES))
    reduced_out = gtx.as_field((Cell,), np.zeros(N_CELLS))

    dual_role_output_reduced_program.with_backend(uncached_dace_cpu)(
        inp, base_out, reduced_out, 1, offset_provider=offset_provider
    )

    assert removed_buffers, "GT4PyWriteBackBufferElimination did not match"
    assert selected_candidates, "no candidate was recorded"
    assert any(targets for targets in view_targets_after_lowering), (
        "the lowering built no Views, so this test would not detect a View of `T`"
    )
    assert not any(selected_candidates.values()), (
        f"a selected transient is viewed: {selected_candidates}; "
        "`gt_propagate_strides_of` does not update Views"
    )

    expected_base = inp.asnumpy() + 1.0
    np.testing.assert_allclose(base_out.asnumpy(), expected_base)
    np.testing.assert_allclose(
        reduced_out.asnumpy(), expected_base[C2E_TABLE].sum(axis=1)
    )
