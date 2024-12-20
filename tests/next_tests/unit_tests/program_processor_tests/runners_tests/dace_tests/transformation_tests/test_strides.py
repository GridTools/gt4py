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
                assert str(nsdfg.symbol_mapping[original_stride]) == original_stride

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
                # The symbol mapping must should not be updated.
                assert original_stride in nsdfg.symbol_mapping
                assert str(nsdfg.symbol_mapping[original_stride]) == original_stride


def _make_strides_propagation_dependent_symbol_nsdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("nested_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    array_names = ["a2", "b2"]
    for name in array_names:
        stride_sym = dace.symbol(f"{name}_stride", dtype=dace.uint64)
        sdfg.add_symbol(stride_sym.name, stride_sym.dtype)
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            strides=(stride_sym,),
            transient=False,
        )

    state.add_mapped_tasklet(
        "nested_comp",
        map_ranges={"__i0": "0:10"},
        inputs={"__in1": dace.Memlet("a2[__i0]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("b2[__i0]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _make_strides_propagation_dependent_symbol_sdfg() -> tuple[dace.SDFG, dace_nodes.NestedSDFG]:
    sdfg_level1 = dace.SDFG(util.unique_name("nested_level"))
    state = sdfg_level1.add_state(is_start_block=True)

    array_names = ["a1", "b1"]
    for name in array_names:
        stride_sym1 = dace.symbol(f"{name}_1stride", dtype=dace.uint64)
        stride_sym2 = dace.symbol(f"{name}_2stride", dtype=dace.int64)
        sdfg_level1.add_symbol(stride_sym1.name, stride_sym1.dtype)
        sdfg_level1.add_symbol(stride_sym2.name, stride_sym2.dtype)
        stride_sym = stride_sym1 * stride_sym2
        sdfg_level1.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            strides=(stride_sym,),
            transient=False,
        )

    sdfg_level2 = _make_strides_propagation_dependent_symbol_nsdfg()

    for sym, sym_dtype in sdfg_level2.symbols.items():
        sdfg_level1.add_symbol(sym, sym_dtype)

    nsdfg = state.add_nested_sdfg(
        sdfg=sdfg_level2,
        parent=sdfg_level1,
        inputs={"a2"},
        outputs={"b2"},
        symbol_mapping={s: s for s in sdfg_level2.symbols},
    )

    state.add_edge(state.add_access("a1"), None, nsdfg, "a2", dace.Memlet("a1[0:10]"))
    state.add_edge(nsdfg, "b2", state.add_access("b1"), None, dace.Memlet("b1[0:10]"))
    sdfg_level1.validate()

    return sdfg_level1, nsdfg


def test_strides_propagation_dependent_symbol():
    sdfg_level1, nsdfg_level2 = _make_strides_propagation_dependent_symbol_sdfg()
    sym1_dtype = dace.uint64
    sym2_dtype = dace.int64

    # Ensure that the special symbols are not already present inside the nested SDFG.
    for aname, adesc in sdfg_level1.arrays.items():
        sym1 = f"{aname}_1stride"
        sym2 = f"{aname}_2stride"
        for sym, dtype in [(sym1, sym1_dtype), (sym2, sym2_dtype)]:
            assert sym in {fs.name for fs in adesc.strides[0].free_symbols}
            assert sym not in nsdfg_level2.symbol_mapping
            assert sym not in nsdfg_level2.sdfg.symbols
            assert sym in sdfg_level1.symbols
            assert sdfg_level1.symbols[sym] == dtype

    # Now propagate `a1` and `b1`.
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "a1", ignore_symbol_mapping=True)
    sdfg_level1.validate()
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "b1", ignore_symbol_mapping=True)
    sdfg_level1.validate()

    # Now we check if the update has worked.
    for aname, adesc in sdfg_level1.arrays.items():
        sym1 = f"{aname}_1stride"
        sym2 = f"{aname}_2stride"
        adesc2 = nsdfg_level2.sdfg.arrays[aname.replace("1", "2")]
        assert adesc2.strides == adesc.strides

        for sym, dtype in [(sym1, sym1_dtype), (sym2, sym2_dtype)]:
            assert sym in nsdfg_level2.symbol_mapping
            assert nsdfg_level2.symbol_mapping[sym].name == sym
            assert sym in sdfg_level1.symbols
            assert sdfg_level1.symbols[sym] == dtype
            assert sym in nsdfg_level2.sdfg.symbols
            assert nsdfg_level2.sdfg.symbols[sym] == dtype
