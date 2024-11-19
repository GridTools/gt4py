# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests the redundant array removal transformation."""

import pytest
import numpy as np
import copy

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util


dace = pytest.importorskip("dace")
import dace
from dace.sdfg import nodes as dace_nodes


def test_gt4py_redundant_array_elimination():
    sdfg: dace.SDFG = dace.SDFG(util.unique_name("test_gt4py_redundant_array_elimination"))
    sdfg.add_array("a", (10, 10), dace.float64, transient=False)
    sdfg.add_array("b", (20, 20), dace.float64, transient=False)
    sdfg.add_array("c", (2, 2), dace.float64, transient=False)
    sdfg.add_array("d", (2, 2), dace.float64, transient=False)
    sdfg.add_array("tmp1", (20, 20), dace.float64, transient=True)
    sdfg.add_array("tmp2", (20, 20), dace.float64, transient=True)

    pre_state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    pre_state.add_mapped_tasklet(
        "set_tmp2_to_zero",
        map_ranges={"__i0": "0:20", "__i1": "0:20"},
        inputs={},
        code="__out = 0.0",
        outputs={"__out": dace.Memlet("tmp2[__i0, __i1]")},
        external_edges=True,
    )

    state: dace.SDFGState = sdfg.add_state_after(pre_state)
    a, tmp1, tmp2 = (state.add_access(name) for name in ["a", "tmp1", "tmp2"])
    state.add_mapped_tasklet(
        "compute",
        map_ranges={"__i0": "0:10", "__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("tmp1[5 + __i0, 6 + __i1]")},
        input_nodes={a},
        output_nodes={tmp1},
        external_edges=True,
    )
    state.add_nedge(
        tmp1,
        tmp2,
        dace.Memlet("tmp1[0:20, 0:20]"),
    )
    state.add_nedge(
        state.add_access("c"),
        tmp1,
        dace.Memlet.simple(
            data="c",
            subset_str="0:2, 0:2",
            other_subset_str="18:20, 17:19",
        ),
    )
    state.add_nedge(
        state.add_access("d"),
        tmp1,
        dace.Memlet("tmp1[0:2, 1:3]"),
    )

    state2: dace.SDFGState = sdfg.add_state_after(state)
    state2.add_nedge(
        state2.add_access("tmp2"),
        state2.add_access("b"),
        dace.Memlet("tmp2[0:20, 0:20]"),
    )
    sdfg.validate()

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRednundantArrayElimination(),
        validate_all=True,
    )
    assert count == 1

    a = np.random.rand(10, 10).astype(np.float64)
    c = np.random.rand(2, 2).astype(np.float64)
    d = np.random.rand(2, 2).astype(np.float64)
    b = np.zeros((20, 20), dtype=np.float64)

    # Compose the reference solution
    b_ref = np.zeros((20, 20), dtype=np.float64)
    b_ref[5:15, 6:16] = a + 1.0
    b_ref[18:20, 17:19] = c
    b_ref[0:2, 1:3] = d

    sdfg(a=a, b=b, c=c, d=d)
    assert np.allclose(b, b_ref)


def test_gt4py_redundant_array_elimination_unequal_shape():
    """Tests the array elimination when the arrays have different shapes."""
    sdfg: dace.SDFG = dace.SDFG(util.unique_name("test_gt4py_redundant_array_elimination"))
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    sdfg.add_array("origin", (30,), dace.float64, transient=False)
    sdfg.add_array("t", (20,), dace.float64, transient=True)
    sdfg.add_array("v", (10,), dace.float64, transient=False)

    origin, t, v = (state.add_access(name) for name in ["origin", "t", "v"])

    state.add_nedge(
        origin,
        t,
        dace.Memlet(data="origin", subset="5:20", other_subset="2:17"),
    )
    state.add_nedge(
        t,
        v,
        dace.Memlet(data="t", subset="3:11", other_subset="1:9"),
    )
    sdfg.validate()

    origin = np.array(np.random.rand(30), dtype=np.float64, copy=True)
    v_ref = np.zeros((10), dtype=np.float64)
    v_ref_init = v_ref.copy()
    v_res = v_ref.copy()

    csdfg_org = sdfg.compile()
    csdfg_org(origin=origin, v=v_ref)
    assert not np.allclose(v_ref, v_ref_init)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRednundantArrayElimination(),
        validate=True,
        validate_all=True,
    )
    assert count == 1, f"Transformation was applied {count}, but expected once."

    csdfg_opt = sdfg.compile()
    csdfg_opt(origin=origin, v=v_res)
    assert np.allclose(v_ref, v_res), f"Expected {v_ref}, but got {v_res}."


def test_gt4py_redundant_array_elimination_unequal_shape_2():
    sdfg: dace.SDFG = dace.SDFG(
        util.unique_name("test_gt4py_redundant_array_elimination_full_read")
    )
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)
    array_names = ["input_", "read", "write", "output_"]
    for name in array_names:
        sdfg.add_array(
            name,
            shape=(50,),
            dtype=dace.float64,
            transient=True,
        )
    sdfg.arrays["input_"].transient = False
    sdfg.arrays["write"].shape = (100,)
    sdfg.arrays["write"].total_size = 100
    sdfg.arrays["output_"].transient = False
    sdfg.arrays["output_"].shape = (100,)
    sdfg.arrays["output_"].total_size = 100
    input_, read, write, output_ = (state.add_access(name) for name in array_names)
    state.remove_node(output_)

    def _mk_memlet(an):
        return dace.Memlet.from_array(an.data, an.desc(sdfg))

    state.add_nedge(input_, read, _mk_memlet(input_))
    state.add_nedge(
        read,
        write,
        dace.Memlet.simple(data="read", subset_str="0:50", other_subset_str="10:60"),
    )

    state2 = sdfg.add_state_after(state)
    write2 = state2.add_access("write")
    state2.add_nedge(write2, state2.add_access("output_"), _mk_memlet(write2))
    sdfg.validate()

    call_args = {
        "input_": np.array(np.random.rand(50), dtype=np.float64, copy=True),
        "output_": np.array(np.random.rand(100), dtype=np.float64, copy=True),
    }

    transformation_applied = False
    try:
        gtx_transformations.GT4PyRednundantArrayElimination.apply_to(
            verify=True,
            sdfg=sdfg,
            read=read,
            write=write,
        )
        transformation_applied = True
    except ValueError as e:
        pass
    assert transformation_applied

    # Now compile the SDFG again to see if there were changes.
    csdfg = sdfg.compile()
    csdfg(**call_args)

    assert np.allclose(
        call_args["input_"],
        call_args["output_"][10:60],
    )


def _make_too_big_copy_sdfg(
    write_is_global: bool,
    offset_starts_at_zero: bool,
) -> tuple[dace.SDFG, dace.SDFGState, dace_nodes.AccessNode, dace_nodes.AccessNode]:
    """SDFGs for the `test_gt4py_redundant_array_elimination_too_big_*()` tests.

    Tests the following scenario:
    ```python
        read[0:100] = ...
        write[0:50] = read[a:(a+50)]
    ```

    This will be simplified to `write[0:100] = ...`, under the condition, that
    - `write` is a transient
    - `a` is equal to `0`.

    Args:
        write_is_global: Makes `write` a global, i.e. non-transient.
        offset_starts_at_zero: Make `a` to `o`.
    """
    sdfg: dace.SDFG = dace.SDFG(util.unique_name("test_gt4py_redundant_array_elimination_too_big"))
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    arr_names = ["input_", "tmp", "read", "write", "return_"]
    for name in arr_names:
        sdfg.add_array(
            name=name,
            shape=(100,),
            dtype=dace.float64,
            transient=True,
        )
    for name in ["input_", "return_"]:
        sdfg.arrays[name].transient = False
    if write_is_global:
        sdfg.arrays["write"].transient = False
    input_, tmp, read, write = (state.add_access(name) for name in arr_names[:-1])

    def _mk_memlet(an):
        return dace.Memlet.from_array(an.data, an.desc(sdfg))

    state.add_nedge(input_, tmp, _mk_memlet(tmp))
    state.add_nedge(tmp, read, _mk_memlet(read))
    if offset_starts_at_zero:
        state.add_nedge(
            read,
            write,
            dace.Memlet.simple(data="read", subset_str="0:50", other_subset_str="0:50"),
        )
    else:
        state.add_nedge(
            read,
            write,
            dace.Memlet.simple(data="read", subset_str="0:50", other_subset_str="10:60"),
        )

    state2 = sdfg.add_state_after(state)
    return_s2 = state2.add_access("return_")
    write_s2 = state2.add_access("write")
    state2.add_nedge(
        write_s2,
        return_s2,
        _mk_memlet(return_s2),
    )
    sdfg.validate()

    return sdfg, state, read, write


def _apply_and_run_to_big_copy_sdfg(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    read: dace_nodes.AccessNode,
    write: dace_nodes.AccessNode,
    should_apply: bool,
) -> bool:
    """Tests SDFG for the `test_gt4py_redundant_array_elimination_too_big_*` family.

    This function tests SDFG that were generated by the `_make_too_big_copy_sdfg()`
    function. It will first run the SDFG to get the ground truth. Then the
    transformation is applied and the SDFG is checked again.

    Args:
        sdfg: The SDFG that should be checked, not modified yet.
        state: The state inside the SDFG.
        read: The access node representing `read`
        write: The access node representing `write`.
        should_apply: Indicate if the transformation should apply or not.
    """

    if not should_apply:
        # In case the transformation should not apply, we only check this.
        # TODO: change to `can_be_applied_to()` once we update DaCe.
        expected_result = False
        try:
            gtx_transformations.GT4PyRednundantArrayElimination.apply_to(
                verify=True,
                sdfg=sdfg,
                read=read,
                write=write,
            )
        except ValueError as e:
            if str(e).startswith("Transformation cannot be"):
                expected_result = True
        assert expected_result, "The transformation applied but it should not."
        return True

    # Now generate the input arguments.
    ref_call_args = {}
    for aname, adesc in sdfg.arrays.items():
        if adesc.transient:
            continue
        ref_call_args[aname] = np.array(np.random.rand(100), dtype=np.float64, copy=True)
    res_call_args = copy.deepcopy(ref_call_args)

    # Now call the SDFG before we modify it.
    csdfg_ref = sdfg.compile()
    csdfg_ref(**ref_call_args)

    # Now we apply the transformation to the SDFG.
    transformation_applied = False
    try:
        gtx_transformations.GT4PyRednundantArrayElimination.apply_to(
            verify=True,
            sdfg=sdfg,
            read=read,
            write=write,
        )
        transformation_applied = True
    except ValueError as e:
        pass
    assert transformation_applied

    # Now compile the SDFG again to see if there were changes.
    sdfg.name = sdfg.name + "_transformed"
    sdfg._regenerate_code = True
    sdfg._recompile = True
    csdfg_res = sdfg.compile()
    csdfg_res(**res_call_args)

    # Now we compare the two result. It is important that we only compare the
    #  lower 50 entries, this is because how the SDFG is constructed.
    assert np.allclose(
        ref_call_args["return_"][0:50],
        res_call_args["return_"][0:50],
    ), f"Comparison failed."

    return True


def _test_gt4py_redundant_array_elimination_too_big(
    write_is_global: bool,
    offset_starts_at_zero: bool,
    should_apply: bool,
):
    test_sdfg = _make_too_big_copy_sdfg(
        write_is_global=write_is_global,
        offset_starts_at_zero=offset_starts_at_zero,
    )
    _apply_and_run_to_big_copy_sdfg(
        *test_sdfg,
        should_apply=should_apply,
    )


def test_gt4py_redundant_array_elimination_too_big_trans_same_off():
    """
    Because `write` is a transient and the offsets are the same, it will apply.
    """
    _test_gt4py_redundant_array_elimination_too_big(
        write_is_global=False,
        offset_starts_at_zero=True,
        should_apply=True,
    )


def test_gt4py_redundant_array_elimination_too_big_trans_diff_off():
    """
    `write` is a transient, but the offsets are not the same, the transformation does not apply.
    """
    _test_gt4py_redundant_array_elimination_too_big(
        write_is_global=False, offset_starts_at_zero=False, should_apply=False
    )


def test_gt4py_redundant_array_elimination_too_big_global_same_off():
    """
    Same situation as in `test_gt4py_redundant_array_elimination_too_big_trans_same_off()`,
    but now `write` is not a transient. Because of this the transformation will no
    longer apply because this would change the semantics, which is observable, since
    `write` is global.
    """
    _test_gt4py_redundant_array_elimination_too_big(
        write_is_global=True,
        offset_starts_at_zero=True,
        should_apply=False,
    )


def test_gt4py_redundant_array_elimination_too_big_global_diff_off():
    _test_gt4py_redundant_array_elimination_too_big(
        write_is_global=True,
        offset_starts_at_zero=False,
        should_apply=False,
    )


def test_gt4py_redundant_array_elimination_self_write():
    """
    The producer of `read` is also `write`.

    This is only allowed if `write` is global, however, in any cases it is forbidden.
    """
    sdfg: dace.SDFG = dace.SDFG(
        util.unique_name("test_gt4py_redundant_array_elimination_same_read")
    )
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    for name in ["input_", "tmp"]:
        sdfg.add_array(name, shape=(100,), dtype=dace.float64, transient=True)
    sdfg.arrays["input_"].transient = False

    input_ = state.add_access("input_")
    read = state.add_access("tmp")
    write = state.add_access("input_")  # Intentional

    def _mk_memlet(an):
        return dace.Memlet.from_array(an.data, an.desc(sdfg))

    state.add_nedge(input_, read, _mk_memlet(input_))
    state.add_nedge(read, write, _mk_memlet(write))

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRednundantArrayElimination(),
        validate_all=True,
    )
    assert count == 0


def _make_global_write_test_sdfg(
    read_used_for_filter: bool,
) -> tuple[dace.SDFG, dace.SDFGState]:
    """
    Generates the SDFG where `write` is global.

    Used in the `test_gt4py_redundant_array_elimination_global_write_*()` tests.
    """

    sdfg: dace.SDFG = dace.SDFG(
        util.unique_name("test_gt4py_redundant_array_elimination_same_read")
    )
    state: dace.SDFGState = sdfg.add_state(is_start_block=True)

    sdfg.add_array("read", shape=(100,), dtype=dace.float64, transient=True)
    sdfg.add_array("write", shape=(100,), dtype=dace.float64, transient=False)

    sdfg.add_array("tmp1", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp2", shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_scalar("tmps", dtype=dace.float64, transient=False)

    read = state.add_access("read")
    write = state.add_access("write")
    tmp1 = state.add_access("tmp1")
    tmp2 = state.add_access("tmp2")
    tmps = state.add_access("tmps")

    state.add_nedge(read, write, dace.Memlet("read[0:50] -> [0:50]"))

    state.add_nedge(tmp1, read, dace.Memlet("tmp1[0:10] -> [20:30]"))

    if read_used_for_filter:
        state.add_nedge(tmp2, read, dace.Memlet("tmp2[0:10] -> [45:55]"))
    else:
        state.add_nedge(tmp2, read, dace.Memlet("tmp2[0:10] -> [40:50]"))
    state.add_nedge(tmps, read, dace.Memlet("tmps[0] -> [3]"))
    sdfg.validate()

    return (sdfg, state)


def test_gt4py_redundant_array_elimination_global_write_in_bound():
    """
    `write` is global data, everything is in bound, so it can be applied.
    """
    sdfg, state = _make_global_write_test_sdfg(read_used_for_filter=False)
    access_nodes_pre = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_pre) == 5
    assert state.number_of_nodes() == 5
    assert sum(dnode.data == "read" for dnode in access_nodes_pre)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRedundantArrayElimination(),
        validate_all=True,
    )
    assert count == 1

    access_nodes_after = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert state.number_of_nodes() == 4
    assert len(access_nodes_after) == 4
    assert not any(dnode.data == "read" for dnode in access_nodes_after)


def test_gt4py_redundant_array_elimination_global_write_filter():
    """
    `write` is global data, `read` used for filtering.

    Because of the filtering, the transformation can not be applied.
    """
    sdfg, state = _make_global_write_test_sdfg(read_used_for_filter=True)
    access_nodes_pre = util.count_nodes(sdfg, dace_nodes.AccessNode, return_nodes=True)
    assert len(access_nodes_pre) == 5
    assert state.number_of_nodes() == 5
    assert sum(dnode.data == "read" for dnode in access_nodes_pre)

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.GT4PyRedundantArrayElimination(),
        validate_all=True,
    )
    assert count == 0
