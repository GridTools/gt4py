# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import dace
import numpy as np
from dace import symbolic as dace_symbolic
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from . import util


def _make_strides_propagation_level3_sdfg() -> dace.SDFG:
    """Generates the level 3 SDFG (nested-nested) SDFG for `test_strides_propagation()`."""
    sdfg = dace.SDFG(util.unique_name("level3"))
    state = sdfg.add_state(is_start_block=True)
    names = ["a3", "c3"]

    for name in names:
        stride_name = name + "_stride"
        stride_sym = dace_symbolic.pystr_to_symbolic(stride_name)
        sdfg.add_symbol(stride_name, dace.int32)
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
        sdfg.add_symbol(stride_name, dace.int32)
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

    # This is the nested SDFG we have here. Its connectors have to be equivalent
    #  to the data they are connected to, thus the stride symbols are mapped to
    #  the ones of this level, which is applied inside the nested SDFG when it is
    #  integrated into this SDFG.
    nsdfg = state.add_nested_sdfg(
        sdfg=_make_strides_propagation_level3_sdfg(),
        inputs={"a3"},
        outputs={"c3"},
        symbol_mapping={"a3_stride": "a2_stride", "c3_stride": "c2_stride"},
    )

    state.add_edge(state.add_access("a2"), None, nsdfg, "a3", dace.Memlet("a2[0:10]"))
    state.add_edge(nsdfg, "c3", state.add_access("c2"), None, dace.Memlet("c2[0:10]"))
    nsdfg.integrate_into_parent()
    sdfg.validate()

    return sdfg, nsdfg


def _make_strides_propagation_level1_sdfg() -> tuple[
    dace.SDFG, dace_nodes.NestedSDFG, dace_nodes.NestedSDFG
]:
    """Generates the level 1 SDFG (top) SDFG for `test_strides_propagation()`.

    Note that the SDFG is valid, but will be indeterminate. The only point of
    this SDFG is to have a lot of different situations that have to be handled
    for the propagation. `a1` is connected to the connectors `a2` and `a2_alias`.

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
        sdfg.add_symbol(stride_name, dace.int32)
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
        inputs={"a2", "c2"},
        outputs={"a2_alias", "b2", "c2"},
        symbol_mapping={
            f"{name}_stride": f"{name[0]}1_stride" for name in ["a2", "a2_alias", "b2", "c2"]
        },
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

    nsdfg_level2.integrate_into_parent()
    sdfg.validate()

    return sdfg, nsdfg_level2, nsdfg_level3


def _get_stride_of(sdfg: dace.SDFG, prefix: str) -> dict[str, str]:
    """Returns the stride of all data of `sdfg`, whose name starts with `prefix`."""
    return {
        aname: str(adesc.strides[0])
        for aname, adesc in sdfg.arrays.items()
        if aname.startswith(prefix)
    }


def test_strides_propagation():
    # Note that the SDFG we are building here is not really meaningful.
    sdfg_level1, nsdfg_level2, nsdfg_level3 = _make_strides_propagation_level1_sdfg()
    all_sdfgs = [sdfg_level1, nsdfg_level2.sdfg, nsdfg_level3.sdfg]

    # Since the connectors are equivalent to the data they are connected to, all
    #  levels use the stride symbols of the top level.
    for sdfg in all_sdfgs:
        for aname, adesc in sdfg.arrays.items():
            assert [str(s) for s in adesc.strides] == [f"{aname[0]}1_stride"]

    # Now we change the strides of `a1` and propagate them, but not the ones of `c1`.
    sdfg_level1.add_symbol("a1_new_stride", dace.int32)
    sdfg_level1.arrays["a1"].set_shape((10,), (dace_symbolic.pystr_to_symbolic("a1_new_stride"),))
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "a1")
    sdfg_level1.validate()

    # The new strides have been propagated to all levels, including `a2_alias`, which is
    #  also connected to `a1`. Inside the nested SDFGs the new symbol is mapped 1:1.
    for sdfg in all_sdfgs:
        assert set(_get_stride_of(sdfg, "a").values()) == {"a1_new_stride"}
        assert set(_get_stride_of(sdfg, "b").values()) <= {"b1_stride"}
        assert set(_get_stride_of(sdfg, "c").values()) == {"c1_stride"}
        if (nsdfg := sdfg.parent_nsdfg_node) is not None:
            assert str(nsdfg.symbol_mapping["a1_new_stride"]) == "a1_new_stride"

    # Now we also change and propagate the strides of `c1`.
    sdfg_level1.add_symbol("c1_new_stride", dace.int32)
    sdfg_level1.arrays["c1"].set_shape((10,), (dace_symbolic.pystr_to_symbolic("c1_new_stride"),))
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "c1")
    sdfg_level1.validate()
    for sdfg in all_sdfgs:
        assert set(_get_stride_of(sdfg, "a").values()) == {"a1_new_stride"}
        assert set(_get_stride_of(sdfg, "c").values()) == {"c1_new_stride"}


def _make_strides_propagation_dependent_symbol_nsdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("strides_propagation_dependent_symbol_nsdfg"))
    state = sdfg.add_state(is_start_block=True)

    array_names = ["a2", "b2"]
    for name in array_names:
        stride_sym = dace.symbol(f"{name}_stride", dtype=dace.uint32)
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
    sdfg_level1 = dace.SDFG(util.unique_name("strides_propagation_dependent_symbol_sdfg"))
    state = sdfg_level1.add_state(is_start_block=True)

    array_names = ["a1", "b1"]
    for name in array_names:
        stride_sym = dace.symbol(f"{name}_stride", dtype=dace.uint32)
        sdfg_level1.add_symbol(stride_sym.name, stride_sym.dtype)
        sdfg_level1.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            strides=(stride_sym,),
            transient=False,
        )

    nsdfg = state.add_nested_sdfg(
        sdfg=_make_strides_propagation_dependent_symbol_nsdfg(),
        inputs={"a2"},
        outputs={"b2"},
        symbol_mapping={"a2_stride": "a1_stride", "b2_stride": "b1_stride"},
    )

    state.add_edge(state.add_access("a1"), None, nsdfg, "a2", dace.Memlet("a1[0:10]"))
    state.add_edge(nsdfg, "b2", state.add_access("b1"), None, dace.Memlet("b1[0:10]"))
    nsdfg.integrate_into_parent()
    sdfg_level1.validate()

    return sdfg_level1, nsdfg


def test_strides_propagation_symbolic_expression():
    sdfg_level1, nsdfg_level2 = _make_strides_propagation_dependent_symbol_sdfg()

    # Now change the strides of `a1` and `b1` to a symbolic expression.
    for aname, adesc in sdfg_level1.arrays.items():
        stride_sym1 = dace.symbol(f"{aname}_1stride", dtype=dace.uint32)
        stride_sym2 = dace.symbol(f"{aname}_2stride", dtype=dace.int32)
        sdfg_level1.add_symbol(stride_sym1.name, stride_sym1.dtype)
        sdfg_level1.add_symbol(stride_sym2.name, stride_sym2.dtype)
        adesc.set_shape((10,), (stride_sym1 * stride_sym2,))

        # Ensure that the symbols are not already present inside the nested SDFG.
        for sym in [stride_sym1.name, stride_sym2.name]:
            assert sym not in nsdfg_level2.symbol_mapping
            assert sym not in nsdfg_level2.sdfg.symbols

    # Now propagate `a1` and `b1`.
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "a1")
    gtx_transformations.gt_propagate_strides_of(sdfg_level1, "b1")
    sdfg_level1.validate()

    # The inner descriptors use the same expression, and the symbols that appear in
    #  it have been mapped 1:1 into the nested SDFG, keeping their type.
    for aname, adesc in sdfg_level1.arrays.items():
        adesc2 = nsdfg_level2.sdfg.arrays[aname.replace("1", "2")]
        assert adesc2.strides == adesc.strides

        for sym, dtype in [(f"{aname}_1stride", dace.uint32), (f"{aname}_2stride", dace.int32)]:
            assert str(nsdfg_level2.symbol_mapping[sym]) == sym
            assert nsdfg_level2.sdfg.symbols[sym] == dtype


def _make_strides_propagation_shared_symbols_nsdfg() -> dace.SDFG:
    sdfg = dace.SDFG(util.unique_name("strides_propagation_shared_symbols_nsdfg"))
    state = sdfg.add_state(is_start_block=True)

    # NOTE: Both arrays have the same symbols used for strides.
    array_names = ["a2", "b2"]
    stride_sym0 = dace.symbol("__stride_0", dtype=dace.uint32)
    stride_sym1 = dace.symbol("__stride_1", dtype=dace.uint32)
    sdfg.add_symbol(stride_sym0.name, stride_sym0.dtype)
    sdfg.add_symbol(stride_sym1.name, stride_sym1.dtype)
    for name in array_names:
        sdfg.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            strides=(stride_sym0, stride_sym1),
            transient=False,
        )

    state.add_mapped_tasklet(
        "nested_comp",
        map_ranges={
            "__i0": "0:10",
            "__i1": "0:10",
        },
        inputs={"__in1": dace.Memlet("a2[__i0, __i1]")},
        code="__out = __in1 + 10.",
        outputs={"__out": dace.Memlet("b2[__i0, __i1]")},
        external_edges=True,
    )
    sdfg.validate()
    return sdfg


def _make_strides_propagation_shared_symbols_sdfg() -> tuple[dace.SDFG, dace_nodes.NestedSDFG]:
    sdfg_level1 = dace.SDFG(util.unique_name("strides_propagation_shared_symbols_sdfg"))
    state = sdfg_level1.add_state(is_start_block=True)

    # NOTE: Both arrays use the same symbols as strides.
    #   Furthermore, they are the same as in the nested SDFG, i.e. they are shared.
    array_names = ["a1", "b1"]
    stride_sym0 = dace.symbol("__stride_0", dtype=dace.uint32)
    stride_sym1 = dace.symbol("__stride_1", dtype=dace.uint32)
    sdfg_level1.add_symbol(stride_sym0.name, stride_sym0.dtype)
    sdfg_level1.add_symbol(stride_sym1.name, stride_sym1.dtype)
    for name in array_names:
        sdfg_level1.add_array(
            name,
            shape=(10, 10),
            dtype=dace.float64,
            strides=(
                stride_sym0,
                stride_sym1,
            ),
            transient=False,
        )

    sdfg_level2 = _make_strides_propagation_shared_symbols_nsdfg()
    nsdfg = state.add_nested_sdfg(
        sdfg=sdfg_level2,
        inputs={"a2"},
        outputs={"b2"},
        symbol_mapping={s: s for s in sdfg_level2.symbols},
    )

    state.add_edge(state.add_access("a1"), None, nsdfg, "a2", dace.Memlet("a1[0:10, 0:10]"))
    state.add_edge(nsdfg, "b2", state.add_access("b1"), None, dace.Memlet("b1[0:10, 0:10]"))
    sdfg_level1.validate()

    return sdfg_level1, nsdfg


def test_strides_propagation_shared_symbols_sdfg():
    """Tests what happens if symbols are (unintentionally) shared between descriptor.

    This test looks rather artificial, but it is actually quite likely. Because
    transients will most likely have the same shape and if the strides are not
    set explicitly, which is the case, the strides will also be related to their
    shape. This test explores the situation, where we can, for whatever reason,
    only propagate the strides of one such data descriptor.
    """

    def ref(a1, b1):
        for i in range(10):
            for j in range(10):
                b1[i, j] = a1[i, j] + 10.0

    sdfg_level1, nsdfg_level2 = _make_strides_propagation_shared_symbols_sdfg()

    res_args = {
        "a1": np.array(np.random.rand(10, 10), order="C", dtype=np.float64, copy=True),
        "b1": np.array(np.random.rand(10, 10), order="F", dtype=np.float64, copy=True),
    }
    ref_args = copy.deepcopy(res_args)

    # Now we change the strides of `b1`, and then we propagate the new strides
    #  into the nested SDFG. We want to keep (for whatever reasons) strides of `a1`.
    stride_b1_sym0 = dace.symbol("__b1_stride_0", dtype=dace.uint32)
    stride_b1_sym1 = dace.symbol("__b1_stride_1", dtype=dace.uint32)
    sdfg_level1.add_symbol(stride_b1_sym0.name, stride_b1_sym0.dtype)
    sdfg_level1.add_symbol(stride_b1_sym1.name, stride_b1_sym1.dtype)

    desc_b1 = sdfg_level1.arrays["b1"]
    desc_b1.set_shape((10, 10), (stride_b1_sym0, stride_b1_sym1))

    # Now we propagate the data into it.
    gtx_transformations.gt_propagate_strides_of(
        sdfg=sdfg_level1,
        data_name="b1",
    )

    # Now we have to prepare the call arguments, i.e. adding the strides
    itemsize = res_args["b1"].itemsize
    res_args.update(
        {
            "__b1_stride_0": res_args["b1"].strides[0] // itemsize,
            "__b1_stride_1": res_args["b1"].strides[1] // itemsize,
            "__stride_0": res_args["a1"].strides[0] // itemsize,
            "__stride_1": res_args["a1"].strides[1] // itemsize,
        }
    )
    ref(**ref_args)
    sdfg_level1(**res_args)
    assert np.allclose(ref_args["b1"], res_args["b1"])
