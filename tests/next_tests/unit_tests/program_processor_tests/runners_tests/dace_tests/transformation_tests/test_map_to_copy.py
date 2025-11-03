# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_sdfg_1(
    consumer_is_map: bool,
) -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("simple_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    sdfg.add_array(
        "a",
        shape=(10, 20),
        dtype=dace.float64,
        transient=False,
    )
    sdfg.add_array(
        "b",
        shape=(10, 4),
        dtype=dace.float64,
        transient=True,
    )
    sdfg.add_array(
        "c",
        shape=(10, 3),
        dtype=dace.float64,
        transient=False,
    )

    a, b, c = (state.add_access(name) for name in "abc")
    state.add_mapped_tasklet(
        "copy",
        map_ranges={"__i": "0:10", "__k": "3:6"},
        inputs={"__in": dace.Memlet("a[__i, __k + 1]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[__i, __k - 2]")},
        external_edges=True,
        input_nodes={a},
        output_nodes={b},
    )

    if consumer_is_map:
        state.add_mapped_tasklet(
            "computation",
            map_ranges={"__i": "0:10", "__k": "6:8"},
            inputs={"__in": dace.Memlet("b[__i, __k - 4]")},
            code="__out = __in + 1.03",
            outputs={"__out": dace.Memlet("c[__i, __k - 6]")},
            external_edges=True,
            input_nodes={b},
            output_nodes={c},
        )
    else:
        state.add_nedge(b, c, dace.Memlet("b[0:10, 2:4] -> [0:10, 1:3]"))
    sdfg.validate()

    return sdfg, state


@pytest.mark.parametrize("bypass_node", [True, False])
@pytest.mark.parametrize("consumer_is_map", [True, False])
def test_map_to_copy_map_consumer(bypass_node: bool, consumer_is_map: bool):
    sdfg, state = _make_sdfg_1(consumer_is_map=consumer_is_map)

    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    if consumer_is_map:
        assert util.count_nodes(sdfg, dace_nodes.Tasklet) == 2
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    else:
        assert util.count_nodes(sdfg, dace_nodes.Tasklet) == 1
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applied = sdfg.apply_transformations_repeated(
        gtx_transformations.MapToCopy(
            single_use_data=({sdfg: {"b"}} if bypass_node else None),
        ),
        validate=True,
        validate_all=True,
    )
    assert nb_applied == 1

    if consumer_is_map:
        tlet_after = util.count_nodes(sdfg, dace_nodes.Tasklet, True)
        assert len(tlet_after) == 1
        assert {tlet.label for tlet in tlet_after} == {"computation"}
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1
    else:
        assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 0
        assert util.count_nodes(sdfg, dace_nodes.Tasklet) == 0

    ac_after = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    if bypass_node:
        assert len(ac_after) == 2
        assert set("ac") == {ac.data for ac in ac_after}
    else:
        assert len(ac_after) == 3
        assert set("acb") == {ac.data for ac in ac_after}

    ac_a = next(iter(ac for ac in ac_after if ac.data == "a"))
    assert state.out_degree(ac_a) == 1
    assert state.in_degree(ac_a) == 0

    ac_c = next(iter(ac for ac in ac_after if ac.data == "c"))
    assert state.out_degree(ac_c) == 0
    assert state.in_degree(ac_c) == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)
