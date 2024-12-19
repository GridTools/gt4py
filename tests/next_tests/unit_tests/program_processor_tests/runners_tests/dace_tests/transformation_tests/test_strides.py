# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

dace = pytest.importorskip("dace")
from dace import symbolic as dace_symbolic
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)

from . import util

import dace


def _make_strides_propagation_level3_sdfg() -> dace.SDFG:
    """Generates the level 3 SDFG (nested-nested) SDFG for `test_strides_propagation()`."""
    sdfg = dace.SDFG(util.unique_name("level3"))
    state = sdfg.add_state(is_start_block=True)
    names = ["a3", "c3"]

    for name in names:
        stride_name = name + "_stride"
        stride_sym = dace_symbolic.pystr_to_symbolic(stride_name)
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
            strides=(stride_sym,),
        )

    state.add_mapped_tasklet(
        "compL3",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("a3[__i0]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("c3[__i0]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _make_strides_propagation_level2_sdfg() -> tuple[dace.SDFG, dace_nodes.NestedSDFG]:
    """Generates the level 2 SDFG (nested) SDFG for `test_strides_propagation()`.

    The function returns the level 2 SDFG and the NestedSDFG node that contains
    the level 3 SDFG.
    """
    sdfg = dace.SDFG(util.unique_name("level2"))
    state = sdfg.add_state(is_start_block=True)
    names = ["a2", "a2_alias", "b2", "c2"]

    for name in names:
        stride_name = name + "_stride"
        stride_sym = dace_symbolic.pystr_to_symbolic(stride_name)
        sdfg.add_symbol(stride_name, dace.int64)
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
            strides=(stride_sym,),
        )

    state.add_mapped_tasklet(
        "compL2_1",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("a2[__i0]")},
        code="__out = __in1 + 10",
        outputs={"__out": dace.Memlet("b2[__i0]")},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        "compL2_2",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("c2[__i0]")},
        code="__out = __in1",
        outputs={"__out": dace.Memlet("a2_alias[__i0]")},
        external_edges=True,
    )

    # This is the nested SDFG we have here.
    sdfg_level3 = _make_strides_propagation_level3_sdfg()

    nsdfg = state.add_nested_sdfg(
        sdfg=sdfg_level3,
        parent=sdfg,
        inputs={"a3"},
        outputs={"c3"},
        symbol_mapping={s3: s3 for s3 in sdfg_level3.free_symbols},
    )

    state.add_edge(state.add_access("a2"), None, nsdfg, "a3", dace.Memlet("a2[0:10]"))
    state.add_edge(nsdfg, "c3", state.add_access("c2"), None, dace.Memlet("c2[0:10]"))
    sdfg.validate()

    return sdfg, nsdfg


def _make_strides_propagation_level1_sdfg() -> (
    tuple[dace.SDFG, dace_nodes.NestedSDFG, dace_nodes.NestedSDFG]
):
    """Generates the level 1 SDFG (top) SDFG for `test_strides_propagation()`.

    Note that the SDFG is valid, but will be indeterminate. The only point of
    this SDFG is to have a lot of different situations that have to be handled
    for renaming.

    Returns:
        A tuple of length three, with the following members:
        - The top level SDFG.
        - The NestedSDFG node that contains the level 2 SDFG (member of the top level SDFG).
        - The NestedSDFG node that contains the lebel 3 SDFG (member of the level 2 SDFG).
    """

    sdfg = dace.SDFG(util.unique_name("level1"))
    state = sdfg.add_state(is_start_block=True)
    names = ["a1", "b1", "c1"]

    for name in names:
        stride_name = name + "_stride"
        stride_sym = dace_symbolic.pystr_to_symbolic(stride_name)
        sdfg.add_symbol(stride_name, dace.int64)
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
            strides=(stride_sym,),
        )

    sdfg_level2, nsdfg_level3 = _make_strides_propagation_level2_sdfg()

    nsdfg_level2: dace_nodes.NestedSDFG = state.add_nested_sdfg(
        sdfg=sdfg_level2,
        parent=sdfg,
        inputs={"a2", "c2"},
        outputs={"a2_alias", "b2", "c2"},
        symbol_mapping={s: s for s in sdfg_level2.free_symbols},
    )

    for inner_name in nsdfg_level2.in_connectors:
        outer_name = inner_name[0] + "1"
        state.add_edge(
            state.add_access(outer_name),
            None,
            nsdfg_level2,
            inner_name,
            dace.Memlet(f"{outer_name}[0:10]"),
        )
    for inner_name in nsdfg_level2.out_connectors:
        outer_name = inner_name[0] + "1"
        state.add_edge(
            nsdfg_level2,
            inner_name,
            state.add_access(outer_name),
            None,
            dace.Memlet(f"{outer_name}[0:10]"),
        )

    sdfg.validate()

    return sdfg, nsdfg_level2, nsdfg_level3


def test_strides_propagation_use_symbol_mapping():
    # Note that the SDFG we are building here is not really meaningful.
    sdfg_level1, nsdfg_level2, nsdfg_level3 = _make_strides_propagation_level1_sdfg()

    # Tests if all strides are distinct in the beginning and match what we expect.
    for sdfg in [sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg]:
        for aname, adesc in sdfg.arrays.items():
            exp_stride = f"{aname}_stride"
            actual_stride = adesc.strides[0]
            assert len(adesc.strides) == 1
            assert (
                str(actual_stride) == exp_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."

            nsdfg = sdfg.parent_nsdfg_node
            if nsdfg is not None:
                assert exp_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[exp_stride]) == exp_stride

    # Now we propagate `a` and `b`, but not `c`.
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "a1", ignore_symbol_mapping=False)
    sdfg_level1.validate()
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "b1", ignore_symbol_mapping=False)
    sdfg_level1.validate()

    # Because `ignore_symbol_mapping=False` the strides of the data descriptor should
    #  not have changed. But the `symbol_mapping` has been updated for `a` and `b`.
    #  However, the symbols will only point one level above.
    for level, sdfg in enumerate([sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg], start=1):
        for aname, adesc in sdfg.arrays.items():
            nsdfg = sdfg.parent_nsdfg_node
            original_stride = f"{aname}_stride"

            if aname.startswith("c"):
                target_symbol = f"{aname}_stride"
            else:
                target_symbol = f"{aname[0]}{level - 1}_stride"

            if nsdfg is not None:
                assert original_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[original_stride]) == target_symbol
            assert len(adesc.strides) == 1
            assert (
                str(adesc.strides[0]) == original_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."

    # Now we also propagate `c` thus now all data descriptors have the same stride
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "c1", ignore_symbol_mapping=False)
    sdfg_level1.validate()
    for level, sdfg in enumerate([sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg], start=1):
        for aname, adesc in sdfg.arrays.items():
            nsdfg = sdfg.parent_nsdfg_node
            original_stride = f"{aname}_stride"
            target_symbol = f"{aname[0]}{level-1}_stride"
            if nsdfg is not None:
                assert original_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[original_stride]) == target_symbol
            assert len(adesc.strides) == 1
            assert (
                str(adesc.strides[0]) == original_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."


def test_strides_propagation_ignore_symbol_mapping():
    # Note that the SDFG we are building here is not really meaningful.
    sdfg_level1, nsdfg_level2, nsdfg_level3 = _make_strides_propagation_level1_sdfg()

    # Tests if all strides are distinct in the beginning and match what we expect.
    for sdfg in [sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg]:
        for aname, adesc in sdfg.arrays.items():
            exp_stride = f"{aname}_stride"
            actual_stride = adesc.strides[0]
            assert len(adesc.strides) == 1
            assert (
                str(actual_stride) == exp_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."

            nsdfg = sdfg.parent_nsdfg_node
            if nsdfg is not None:
                assert exp_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[exp_stride]) == exp_stride

    # Now we propagate `a` and `b`, but not `c`.
    # TODO(phimuell): Create a version where we can set `ignore_symbol_mapping=False`.
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "a1", ignore_symbol_mapping=True)
    sdfg_level1.validate()
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "b1", ignore_symbol_mapping=True)
    sdfg_level1.validate()

    # After the propagation `a` and `b` should use the same stride (the one that
    #  it has on level 1, but `c` should still be level depending.
    for sdfg in [sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg]:
        for aname, adesc in sdfg.arrays.items():
            original_stride = f"{aname}_stride"
            if aname.startswith("c"):
                exp_stride = f"{aname}_stride"
            else:
                exp_stride = f"{aname[0]}1_stride"
            assert len(adesc.strides) == 1
            assert (
                str(adesc.strides[0]) == exp_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."

            nsdfg = sdfg.parent_nsdfg_node
            if nsdfg is not None:
                assert original_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[original_stride]) == exp_stride

    # Now we also propagate `c` thus now all data descriptors have the same stride
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "c1", ignore_symbol_mapping=True)
    sdfg_level1.validate()
    for sdfg in [sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg]:
        for aname, adesc in sdfg.arrays.items():
            exp_stride = f"{aname[0]}1_stride"
            original_stride = f"{aname}_stride"
            assert len(adesc.strides) == 1
            assert (
                str(adesc.strides[0]) == exp_stride
            ), f"Expected that '{aname}' has strides '{exp_stride}', but found '{adesc.strides}'."

            nsdfg = sdfg.parent_nsdfg_node
            if nsdfg is not None:
                assert original_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[original_stride]) == exp_stride
