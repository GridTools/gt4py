# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest
import copy
import numpy as np

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_simple_linear_chain_sdfg() -> dace.SDFG:
    """Creates a simple linear chain.

    All intermediates have the same size.
    """
    sdfg = dace.SDFG(util.unique_name("simple_linear_chain_sdfg"))

    for name in ["a", "b", "c", "d", "e"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=True,
        )
    sdfg.arrays["a"].transient = False
    sdfg.arrays["e"].transient = False

    state = sdfg.add_state(is_start_block=True)
    b, c, d, e = (state.add_access(name) for name in "bcde")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[__i]")},
        output_nodes={b},
        external_edges=True,
    )
    state.add_nedge(b, c, dace.Memlet("b[0:10] -> [0:10]"))
    state.add_nedge(c, d, dace.Memlet("c[0:10] -> [0:10]"))
    state.add_nedge(d, e, dace.Memlet("d[0:10] -> [0:10]"))
    sdfg.validate()
    return sdfg


def _make_diff_sizes_linear_chain_sdfg() -> (
    tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.Tasklet]
):
    """Creates a linear chain of copies.

    The main differences compared to the SDFG made by `_make_simple_linear_chain_sdfg()`
    is that here the intermediate arrays have different sizes, that become bigger.
    It essentially checks the adjusting of the memlet subset during copying.

    The function returns a tuple with the following content.
    - The SDFG that was generated.
    - The SDFG state.
    - The AccessNode that is used as final output, refers to `e`.
    - The Tasklet that is within the Map.
    """
    sdfg = dace.SDFG(util.unique_name("diff_size_linear_chain_sdfg"))

    array_size_increment = 10
    array_size = 10
    for name in ["a", "b", "c", "d", "e"]:
        sdfg.add_array(
            name,
            shape=(array_size,),
            dtype=dace.float64,
            transient=True,
        )
        array_size += array_size_increment
    sdfg.arrays["a"].transient = False
    sdfg.arrays["e"].transient = False
    assert sdfg.arrays["e"].shape[0] == 50

    state = sdfg.add_state(is_start_block=True)
    b, c, d, e = (state.add_access(name) for name in "bcde")

    tasklet, _, _ = state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[__i + 3]")},
        output_nodes={b},
        external_edges=True,
    )
    state.add_nedge(b, c, dace.Memlet("b[0:20] -> [10:30]"))
    state.add_nedge(c, d, dace.Memlet("c[0:30] -> [2:32]"))
    state.add_nedge(d, e, dace.Memlet("d[0:40] -> [3:43]"))
    sdfg.validate()
    return sdfg, state, e, tasklet


def _make_multi_stage_reduction_sdfg() -> dace.SDFG:
    """Creates an SDFG that has a two stage copy reduction."""
    sdfg = dace.SDFG(util.unique_name("multi_stage_reduction"))
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    # This is the size of the arrays, if not mentioned here, then its size is 10.
    array_sizes: dict[str, int] = {"d": 20, "f": 40, "o1": 40}
    def_array_size = 10

    array_names: list[str] = ["i1", "i2", "i3", "i4", "a", "b", "c", "d", "e", "f", "o1"]
    for name in array_names:
        sdfg.add_array(
            name,
            shape=(array_sizes.get(name, def_array_size),),
            dtype=dace.float64,
            transient=(len(name) == 1),
        )

    a, b, c, d, e, f = (state.add_access(name) for name in "abcdef")

    state.add_mapped_tasklet(
        "comp_i1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("i1[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("a[__i]")},
        output_nodes={a},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp_i2",
        map_ranges={"__j": "0:10"},
        inputs={"__in": dace.Memlet("i2[__j]")},
        code="__out = __in + 2.",
        outputs={"__out": dace.Memlet("b[__j]")},
        output_nodes={b},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp_i3",
        map_ranges={"__k": "0:10"},
        inputs={"__in": dace.Memlet("i3[__k]")},
        code="__out = __in + 3.",
        outputs={"__out": dace.Memlet("c[__k]")},
        output_nodes={c},
        external_edges=True,
    )

    state.add_nedge(state.add_access("i4"), e, dace.Memlet("i4[0:10] -> [0:10]"))

    state.add_nedge(b, d, dace.Memlet("b[0:10] -> [0:10]"))
    state.add_nedge(c, d, dace.Memlet("c[0:10] -> [10:20]"))

    state.add_nedge(a, f, dace.Memlet("a[0:10] -> [0:10]"))
    state.add_nedge(d, f, dace.Memlet("d[0:20] -> [10:30]"))
    state.add_nedge(e, f, dace.Memlet("e[0:10] -> [30:40]"))

    state.add_nedge(f, state.add_access("o1"), dace.Memlet("f[0:40] -> [0:40]"))

    sdfg.validate()
    return sdfg


def _make_not_fully_copied() -> dace.SDFG:
    """
    Make an SDFG where two intermediate array is not fully copied. Thus the
    transformation only applies once, when `d` is removed.
    """
    sdfg = dace.SDFG(util.unique_name("not_fully_copied_intermediate"))

    for name in ["a", "b", "c", "d", "e"]:
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=True,
        )
    sdfg.arrays["a"].transient = False
    sdfg.arrays["e"].transient = False

    state = sdfg.add_state(is_start_block=True)
    b, c, d, e = (state.add_access(name) for name in "bcde")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in",
        outputs={"__out": dace.Memlet("b[__i]")},
        output_nodes={b},
        external_edges=True,
    )
    state.add_nedge(b, c, dace.Memlet("b[2:10] -> [0:8]"))
    state.add_nedge(c, d, dace.Memlet("c[0:8] -> [0:8]"))
    state.add_nedge(d, e, dace.Memlet("d[0:10] -> [0:10]"))
    sdfg.validate()
    return sdfg


def _make_possible_cyclic_sdfg() -> dace.SDFG:
    """
    If the transformation would remove `a1` then it would create a cycle. Thus the
    transformation should not apply.
    """
    sdfg = dace.SDFG(util.unique_name("possible_cyclic_sdfg"))

    anames = ["i1", "a1", "a2", "o1"]
    for name in anames:
        sdfg.add_array(
            name,
            shape=((30,) if name in ["o1", "a2"] else (10,)),
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["a1"].transient = True
    sdfg.arrays["a2"].transient = True

    state = sdfg.add_state(is_start_block=True)
    i1, a1, a2, o1 = (state.add_access(name) for name in anames)

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("i1[__i]")},
        code="__out = __in + 1",
        outputs={"__out": dace.Memlet("a2[__i]")},
        input_nodes={i1},
        output_nodes={a2},
        external_edges=True,
    )

    state.add_nedge(i1, a1, dace.Memlet("i1[0:10] -> [0:10]"))
    state.add_nedge(a1, a2, dace.Memlet("a1[0:10] -> [10:20]"))

    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__j": "0:10"},
        inputs={"__in": dace.Memlet("a1[__j]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("a2[__j + 20]")},
        input_nodes={a1},
        output_nodes={a2},
        external_edges=True,
    )

    state.add_nedge(a2, o1, dace.Memlet("a2[0:30] -> [0:30]"))

    sdfg.validate()
    return sdfg


def test_simple_linear_chain():
    sdfg = _make_simple_linear_chain_sdfg()

    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 2
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert nb_applies == 3


def test_diff_size_linear_chain():
    sdfg, state, output, tasklet = _make_diff_sizes_linear_chain_sdfg()

    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 2
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert nb_applies == 3
    assert output in acnodes
    assert state.in_degree(output) == 1
    assert state.out_degree(output) == 0

    # Look if the subsets were correctly adapted, for that we look at the output
    #  AccessNode and the tasklet inside the map.
    output_memlet: dace.Memlet = next(iter(state.in_edges(output))).data
    assert output_memlet.dst_subset.min_element()[0] == 18
    assert output_memlet.dst_subset.max_element()[0] == 27

    tasklet_memlet: dace.Memlet = next(iter(state.out_edges(tasklet))).data
    assert str(tasklet_memlet.subset[0][0] - 18).strip() == "__i"


def test_multi_stage_reduction():
    sdfg = _make_multi_stage_reduction_sdfg()

    # Make the input
    ref = {
        "i1": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "i2": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "i3": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "i4": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "o1": np.zeros(40, dtype=np.float64),
    }
    res = copy.deepcopy(ref)

    # Generate the reference solution.
    csdfg_ref = sdfg.compile()
    csdfg_ref(**ref)

    # Apply the transformation.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Run the processed SDFG
    csdfg_res = sdfg.compile()
    csdfg_res(**res)

    # Perform all the checks.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 5
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_not_fully_copied():
    sdfg = _make_not_fully_copied()

    # Apply the transformation.
    #  It will only remove `d` all the others are retained, because they are not read
    #  correctly, i.e. fully.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Perform all the checks.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 4
    assert nb_applies == 1
    assert "d" not in acnodes


def test_possible_cyclic_sdfg():
    sdfg = _make_possible_cyclic_sdfg()

    # Apply the transformation.
    #  It will not remove `a1`, because it it would and replace it with `a2` then
    #  the resulting SDFG is cyclic. It will, however, replace `a2` with `o1`.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Perform all the checks.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 3
    assert nb_applies == 1
    assert "o1" not in acnodes
