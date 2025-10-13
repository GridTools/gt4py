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

from dace import libraries as dace_libnode
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


def _make_diff_sizes_pull_chain_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.Tasklet
]:
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
    sdfg = dace.SDFG(util.unique_name("diff_size_linear_pull_chain_sdfg"))

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


def _make_diff_sizes_push_chain_sdfg() -> tuple[
    dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.Tasklet
]:
    """Creates a linear chain of copies.

    Same as `_make_simple_linear_pull_chain_sdfg()` but the intermediates become
    smaller and smaller, so the full shape of the destination array is always written.
    """
    sdfg = dace.SDFG(util.unique_name("diff_size_linear_push_chain_sdfg"))

    array_size_decrement = 10
    array_size = 50
    for name in ["a", "b", "c", "d", "e"]:
        sdfg.add_array(
            name,
            shape=(array_size,),
            dtype=dace.float64,
            transient=True,
        )
        array_size -= array_size_decrement
    sdfg.arrays["a"].transient = False
    sdfg.arrays["e"].transient = False
    assert sdfg.arrays["e"].shape[0] == 10

    state = sdfg.add_state(is_start_block=True)
    a, b, c, d = (state.add_access(name) for name in "abcd")

    state.add_nedge(a, b, dace.Memlet("a[3:43] -> [0:40]"))
    state.add_nedge(b, c, dace.Memlet("b[2:32] -> [0:30]"))
    state.add_nedge(c, d, dace.Memlet("c[10:30] -> [0:20]"))
    tasklet, _, _ = state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("d[__i + 3]")},
        input_nodes={d},
        code="__out = __in",
        outputs={"__out": dace.Memlet("e[__i]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg, state, a, tasklet


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


def _make_linear_chain_with_nested_sdfg_sdfg() -> tuple[dace.SDFG, dace.SDFG]:
    """
    The structure is very similar than `_make_diff_sizes_linear_chain_sdfg()`, with
    the main difference that the Map is inside a NestedSDFG.
    """

    def make_inner_sdfg() -> dace.SDFG:
        inner_sdfg = dace.SDFG("inner_sdfg")
        inner_state = inner_sdfg.add_state(is_start_block=True)
        for name in ["i0", "o0"]:
            inner_sdfg.add_array(name=name, shape=(10, 10), dtype=dace.float64, transient=False)
        inner_state.add_mapped_tasklet(
            "inner_comp",
            map_ranges={
                "__i0": "0:10",
                "__i1": "0:10",
            },
            inputs={"__in": dace.Memlet("i0[__i0, __i1]")},
            code="__out = __in + 10.",
            outputs={"__out": dace.Memlet("o0[__i0, __i1]")},
            external_edges=True,
        )
        inner_sdfg.validate()
        return inner_sdfg

    inner_sdfg = make_inner_sdfg()

    sdfg = dace.SDFG(util.unique_name("linear_chain_with_nested_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    array_size_increment = 10
    array_size = 10
    for name in ["a", "b", "c", "d", "e"]:
        sdfg.add_array(
            name,
            shape=(array_size, array_size),
            dtype=dace.float64,
            transient=True,
        )
        if name != "a":
            array_size += array_size_increment
    assert sdfg.arrays["a"].shape == sdfg.arrays["b"].shape
    assert sdfg.arrays["e"].shape == (40, 40)
    sdfg.arrays["a"].transient = False
    sdfg.arrays["e"].transient = False
    a, b, c, d, e = (state.add_access(name) for name in "abcde")

    nsdfg = state.add_nested_sdfg(
        inner_sdfg,
        inputs={"i0"},
        outputs={"o0"},
        symbol_mapping={},
    )

    state.add_edge(a, None, nsdfg, "i0", sdfg.make_array_memlet("a"))
    state.add_edge(nsdfg, "o0", b, None, sdfg.make_array_memlet("b"))

    state.add_nedge(b, c, dace.Memlet("b[0:10, 0:10] -> [5:15, 3:13]"))
    state.add_nedge(c, d, dace.Memlet("c[0:20, 0:20] -> [2:22, 6:26]"))
    state.add_nedge(d, e, dace.Memlet("d[0:30, 0:30] -> [1:31, 8:38]"))
    sdfg.validate()
    return sdfg, inner_sdfg


def _make_a1_has_output_sdfg() -> dace.SDFG:
    """Here `a1` has an output degree of 2, one to `a2` and one to another output."""
    sdfg = dace.SDFG(util.unique_name("a1_has_an_additional_output_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    # All other arrays have a size of 10.
    anames = ["i1", "i2", "i3", "a1", "a2", "o1", "o2"]
    def_array_size = 10
    asizes = {"a1": 20, "a2": 30, "o2": 30}
    for name in anames:
        sdfg.add_array(
            name=name,
            shape=(asizes.get(name, def_array_size),),
            dtype=dace.float64,
            transient=name[0] == "a",
        )
    a1, a2 = (state.add_access("a1"), state.add_access("a2"))

    state.add_nedge(state.add_access("i1"), a1, dace.Memlet("i1[0:10] -> [0:10]"))
    state.add_nedge(state.add_access("i2"), a1, dace.Memlet("i2[0:10] -> [10:20]"))

    state.add_nedge(a1, state.add_access("o1"), dace.Memlet("a1[5:15] -> [0:10]"))

    state.add_nedge(state.add_access("i3"), a2, dace.Memlet("i3[0:10] -> [0:10]"))
    state.add_nedge(a1, a2, dace.Memlet("a1[0:20] -> [10:30]"))

    state.add_nedge(a2, state.add_access("o2"), dace.Memlet("a2[0:30] -> [0:30]"))

    sdfg.validate()
    return sdfg


def _make_copy_chain_with_reduction_node(
    output_an_array: bool,
) -> tuple[
    dace.SDFG,
    dace.SDFGState,
    dace_libnode.Reduce,
    dace_nodes.AccessNode,
    dace_libnode.standar.Reduce,
    dace_nodes.AccessNode,
    dace_nodes.AccessNode,
]:
    sdfg = dace.SDFG(util.unique_name("copy_chain_remover_with_reduction_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    if output_an_array:
        input_shape = (10, 3, 20)
        output_shape = (10, 2, 20)
        reduce_axes = [1]
        # Actually we should use `(10, 20)` as shape, but then the transformation
        #  does not apply, this is a limitation in the transformation.
        acc_shape = (10, 1, 20)
    else:
        input_shape = (3,)
        acc_shape = ()  # Is a scalar.
        output_shape = (2,)
        reduce_axes = None

    for i in range(2):
        sdfg.add_array(
            f"data{i}",
            shape=input_shape,
            dtype=dace.float64,
            transient=False,
        )
        if output_an_array:
            sdfg.add_array(
                f"acc{i}",
                shape=acc_shape,
                dtype=dace.float64,
                transient=True,
            )
        else:
            sdfg.add_scalar(
                f"acc{i}",
                dtype=dace.float64,
                transient=True,
            )
    sdfg.add_array(
        "output",
        shape=output_shape,
        dtype=dace.float64,
        transient=False,
    )

    accumulators: list[dace_nodes.AccessNode] = []
    reducers: list[dace_libnode.standard.Reduce] = []
    for i in range(2):
        data_ac = state.add_access(f"data{i}")
        acc_ac = state.add_access(f"acc{i}")
        reduce_node = state.add_reduce(
            wcr="lambda x, y: x + y",
            axes=reduce_axes,
            identity=0.0,
        )
        state.add_nedge(
            data_ac,
            reduce_node,
            dace.Memlet(f"{data_ac.data}[" + ", ".join(f"0:{s}" for s in input_shape) + "]"),
        )
        if output_an_array:
            state.add_nedge(
                reduce_node,
                acc_ac,
                dace.Memlet(f"{acc_ac.data}[0:{acc_shape[0]}, 0, 0:{acc_shape[-1]}]"),
            )
        else:
            state.add_nedge(reduce_node, acc_ac, dace.Memlet(f"{acc_ac.data}[0]"))
        accumulators.append(acc_ac)
        reducers.append(reduce_node)

    red0, red1 = reducers
    output_ac = state.add_access("output")

    for i, acc in enumerate(accumulators):
        if output_an_array:
            state.add_nedge(
                acc,
                output_ac,
                dace.Memlet(
                    f"{acc.data}[0:{output_shape[0]}, 0, 0:{output_shape[-1]}] -> [0:{output_shape[0]}, {i}, 0:{output_shape[-1]}]"
                ),
            )
        else:
            state.add_nedge(acc, output_ac, dace.Memlet(f"{acc.data}[0] -> [{i}]"))
    sdfg.validate()

    return sdfg, state, red0, accumulators[0], red1, accumulators[1], output_ac


def test_simple_linear_chain():
    sdfg = _make_simple_linear_chain_sdfg()

    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 2
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert nb_applies == 3


def test_diff_size_linear_pull_chain():
    sdfg, state, output, tasklet = _make_diff_sizes_pull_chain_sdfg()

    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert nb_applies == 3
    assert len(acnodes) == 2
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
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


def test_diff_size_linear_push_chain():
    sdfg, state, input, tasklet = _make_diff_sizes_push_chain_sdfg()

    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert nb_applies == 3
    assert len(acnodes) == 2
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert input in acnodes
    assert state.in_degree(input) == 0
    assert state.out_degree(input) == 1

    # Look if the subsets were correctly adapted, for that we look at the output
    #  AccessNode and the tasklet inside the map.
    input_memlet: dace.Memlet = next(iter(state.out_edges(input))).data
    assert input_memlet.src_subset.min_element()[0] == 18
    assert input_memlet.src_subset.max_element()[0] == 27

    tasklet_memlet: dace.Memlet = next(iter(state.in_edges(tasklet))).data
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
    util.compile_and_run_sdfg(sdfg, **ref)

    # Apply the transformation.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Run the processed SDFG
    util.compile_and_run_sdfg(sdfg, **res)

    # Perform all the checks.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 5
    assert not any(ac.desc(sdfg).transient for ac in acnodes)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_not_fully_copied():
    sdfg = _make_not_fully_copied()

    # NOTE: In the unprocessed SDFG, especially in the `(d) -> (e)` Memlet.
    #  we read uninitialized memory from `d`, because we only write into `[0:8]`
    #  but read from it in the range `[0:10]`, thus the range `[8:10]` in `e`
    #  has undefined value. We have to take that into account.
    ref = {
        "a": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "e": np.array(np.random.rand(10), dtype=np.float64, copy=True),
    }
    res = copy.deepcopy(ref)
    org = copy.deepcopy(ref)

    # Compile and run the original SDFG
    util.compile_and_run_sdfg(sdfg, **ref)

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

    # Now run the test and compare the results, see above for the ranges.
    util.compile_and_run_sdfg(sdfg, **res)
    assert np.all(ref["a"] == res["a"])
    assert np.all(org["a"] == res["a"])
    assert np.all(res["e"][0:8] == ref["e"][0:8])
    assert np.all(res["e"][8:10] == org["e"][8:10])


def test_possible_cyclic_sdfg():
    sdfg = _make_possible_cyclic_sdfg()

    # Apply the transformation.
    #  The first iteration will replace `a2` with `o1`, in pull mode, since the
    #  full shape of `a2` is copied into `o1`. In a second iteration, the transformation
    #  will be applied in push mode: the full shape of the global `i1` is copied
    #  into `a1`, thus `a1` will be replaced with `i1`.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)
    assert nb_applies == 2

    # Perform all the checks.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert {node.data for node in acnodes} == {"i1", "o1"}


def test_a1_additional_output():
    sdfg = _make_a1_has_output_sdfg()

    # Make the input
    ref = {
        "i1": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "i2": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "i3": np.array(np.random.rand(10), dtype=np.float64, copy=True),
        "o1": np.zeros(10, dtype=np.float64),
        "o2": np.zeros(30, dtype=np.float64),
    }
    res = copy.deepcopy(ref)

    util.compile_and_run_sdfg(sdfg, **ref)

    # Apply the transformation.
    #  The transformation removes `a1` and `a2`.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Perform the tests.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert len(acnodes) == 5
    assert nb_applies == 2
    assert not any(acnode.data.startswith("a") for acnode in acnodes)

    # Now run the SDFG, which is essentially to check if the subsets were handled
    #  correctly. This is especially important for `o1` which is composed of both
    #  `i1` and `i2`.
    util.compile_and_run_sdfg(sdfg, **res)

    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


def test_linear_chain_with_nested_sdfg():
    sdfg, inner_sdfg = _make_linear_chain_with_nested_sdfg_sdfg()

    # Ensure that the SDFG was constructed in the correct way.
    assert inner_sdfg.arrays["i0"].strides == sdfg.arrays["a"].strides
    assert inner_sdfg.arrays["o0"].strides == sdfg.arrays["b"].strides
    assert inner_sdfg.arrays["i0"].shape == inner_sdfg.arrays["o0"].shape
    assert inner_sdfg.arrays["i0"].shape == sdfg.arrays["a"].shape

    def ref_comp(a, e):
        def inner_ref(i0, o0):
            for i in range(10):
                for j in range(10):
                    o0[i, j] = i0[i, j] + 10

        b, c, d = np.zeros_like(a), np.zeros((20, 20)), np.zeros((30, 30))
        inner_ref(i0=a, o0=b)
        c[5:15, 3:13] = b
        d[2:22, 6:26] = c
        e[1:31, 8:38] = d

    # Make the input
    ref = {
        "a": np.array(np.random.rand(10, 10), dtype=np.float64, copy=True),
        "e": np.zeros((40, 40), dtype=np.float64),
    }
    res = copy.deepcopy(ref)

    ref_comp(**ref)

    # Apply the transformation.
    #  It should remove all non transient arrays.
    nb_applies = gtx_transformations.gt_remove_copy_chain(sdfg, validate_all=True)

    # Perform the tests.
    acnodes: list[dace_nodes.AccessNode] = util.count_nodes(
        sdfg, dace_nodes.AccessNode, return_nodes=True
    )
    assert {ac.data for ac in acnodes} == {"a", "e"}
    assert util.count_nodes(sdfg, dace_nodes.NestedSDFG) == 1

    # The shapes should be the same as before.
    assert inner_sdfg.arrays["i0"].shape == inner_sdfg.arrays["o0"].shape
    assert inner_sdfg.arrays["i0"].shape == sdfg.arrays["a"].shape

    # The strides of `i0` should also be the same as before, but the strides
    #  of `o0` should now be the same as `e`.
    assert inner_sdfg.arrays["i0"].strides == sdfg.arrays["a"].strides
    assert inner_sdfg.arrays["o0"].strides == sdfg.arrays["e"].strides

    # Now run the transformed SDFG to see if the same output is generated.
    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref.keys())


@pytest.mark.parametrize("output_an_array", [False, True])
def test_copy_chain_remover_with_reduction(output_an_array: bool):
    sdfg, state, red0, acc0, red1, acc1, output = _make_copy_chain_with_reduction_node(
        output_an_array
    )

    def apply_to(a1, a2):
        candidate = {
            gtx_transformations.CopyChainRemover.node_a1: a1,
            gtx_transformations.CopyChainRemover.node_a2: a2,
        }
        copy_chain_remover = gtx_transformations.CopyChainRemover(
            single_use_data={sdfg: {acc0.data, acc1.data}},
        )
        copy_chain_remover.setup_match(
            sdfg=sdfg,
            cfg_id=state.parent_graph.cfg_id,
            state_id=state.block_id,
            subgraph=candidate,
            expr_index=0,
            override=True,
        )
        assert copy_chain_remover.can_be_applied(state, 0, sdfg, permissive=False)
        copy_chain_remover.apply(state, sdfg)

    assert sdfg.number_of_nodes() == 1
    assert state.number_of_nodes() == 7

    assert all(e.dst is acc0 for e in state.out_edges(red0))
    assert state.in_degree(acc0) == 1
    assert state.out_degree(acc0) == 1

    assert all(e.dst is acc1 for e in state.out_edges(red1))
    assert state.in_degree(acc1) == 1
    assert state.out_degree(acc1) == 1

    assert all(e.src in [acc0, acc1] for e in state.in_edges(output))
    assert state.out_degree(output) == 0

    # Now remove the `acc0` intermediate.
    apply_to(a1=acc0, a2=output)

    # The accumulator `acc0` has been removed.
    sdfg.validate()
    assert state.number_of_nodes() == 6

    access_nodes: list[dace_nodes.AccessNode] = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(access_nodes) == 4
    assert acc0 not in access_nodes
    assert acc1 in access_nodes
    assert output in access_nodes

    assert state.out_degree(red0) == 1
    red0_oedge: dace.sdfg.graph.MultiDiConnectorGraph[dace.Memlet] = next(
        iter(state.out_edges(red0))
    )
    assert red0_oedge.dst is output
    red0_oedge_mlet: dace.Memlet = red0_oedge.data
    assert red0_oedge_mlet.src_subset is None

    if output_an_array:
        assert len(red0_oedge_mlet.dst_subset) == 3
        assert red0_oedge_mlet.dst_subset == dace.subsets.Range.from_string("0:10, 0, 0:20")

    else:
        assert len(red0_oedge_mlet.dst_subset) == 1
        assert red0_oedge_mlet.dst_subset == dace.subsets.Range.from_string("0")

    assert state.out_degree(red1) == 1
    assert all(e.dst is acc1 for e in state.out_edges(red1))

    # Now the accumulator `acc1` will be removed.
    apply_to(a1=acc1, a2=output)

    sdfg.validate()
    assert state.number_of_nodes() == 5

    access_nodes = util.count_nodes(sdfg, dace_nodes.AccessNode, True)
    assert len(access_nodes) == 3
    assert acc0 not in access_nodes
    assert acc1 not in access_nodes
    assert output in access_nodes

    assert state.out_degree(red1) == 1
    red1_oedge: dace.sdfg.graph.MultiDiConnectorGraph[dace.Memlet] = next(
        iter(state.out_edges(red1))
    )
    assert red1_oedge.dst is output
    red1_oedge_mlet: dace.Memlet = red1_oedge.data
    assert red1_oedge_mlet.src_subset is None

    if output_an_array:
        assert len(red1_oedge_mlet.dst_subset) == 3
        assert red1_oedge_mlet.dst_subset == dace.subsets.Range.from_string("0:10, 1, 0:20")
    else:
        assert len(red1_oedge_mlet.dst_subset) == 1
        assert red1_oedge_mlet.dst_subset == dace.subsets.Range.from_string("1")
