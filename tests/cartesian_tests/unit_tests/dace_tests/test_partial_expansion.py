# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from cartesian_tests.integration_tests.multi_feature_tests.stencil_definitions import (
    EXTERNALS_REGISTRY,
    REGISTRY,
)


dace = pytest.importorskip("dace")

from gt4py.cartesian.backend.dace_backend import (  # noqa: E402 (needs to be guarded by above importorskip)
    SDFGManager,
)
from gt4py.cartesian.gtc import (  # noqa: E402 (needs to be guarded by above importorskip)
    daceir as dcir,
)
from gt4py.cartesian.gtc.dace.partial_expansion import (  # noqa: E402 (needs to be guarded by above importorskip)
    partially_expand,
)


def get_sdfg(name, backend):
    definition = REGISTRY[name]
    externals = EXTERNALS_REGISTRY[name]
    lazy_stencil = gtscript.lazy_stencil(
        definition=definition, backend=backend, externals=externals
    )
    sdfg_manager = SDFGManager(lazy_stencil.builder)
    return sdfg_manager.unexpanded_sdfg()


@gtscript.lazy_stencil(backend="dace:cpu")
def stencil_inextents_1(inp1: gtscript.Field[np.float64], outp1: gtscript.Field[np.float64]):  # type: ignore[valid-type]
    with computation(PARALLEL), interval(...):
        outp1 = (  # noqa F841: local variable 'outp' is assigned to but never used
            inp1[0, -1, 0] + inp1[0, 1, 0]  # type: ignore[index]
        )


@gtscript.lazy_stencil(backend="dace:cpu")
def stencil_inextents_2(inp2: gtscript.Field[np.float64], outp2: gtscript.Field[np.float64]):  # type: ignore[valid-type]
    with computation(PARALLEL), interval(...):
        outp2 = (  # noqa F841: local variable 'outp' is assigned to but never used
            inp2[0, -1, 0] + inp2[0, 1, 0]  # type: ignore[index]
        )


@gtscript.lazy_stencil(backend="dace:cpu")
def stencil_inextents_3(
    field_a: gtscript.Field[np.float64],  # type: ignore[valid-type]
    field_b: gtscript.Field[np.float64],  # type: ignore[valid-type]
    tmp: gtscript.Field[np.float64],  # type: ignore[valid-type]
):
    with computation(PARALLEL), interval(...):
        field_b = (  # noqa F841: local variable 'field_b' is assigned to but never used
            field_a[0, -1, 0] + field_a[0, 1, 0]  # type: ignore[index]
        )
        tmp = 7.0  # noqa F841: local variable 'tmp' is assigned to but never used


@gtscript.lazy_stencil(backend="dace:cpu")
def stencil_inextents_4(
    field_b: gtscript.Field[np.float64],  # type: ignore[valid-type]
    field_c: gtscript.Field[np.float64],  # type: ignore[valid-type]
    tmp: gtscript.Field[np.float64],  # type: ignore[valid-type]
):
    with computation(PARALLEL), interval(...):
        field_c = (  # noqa F841: local variable 'field_c' is assigned to but never used
            tmp * field_b[0, -1, 0] + field_b[0, 1, 0]  # type: ignore[index]
        )


@gtscript.lazy_stencil(backend="dace:cpu")
def stencil_noextents(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):  # type: ignore[valid-type]
    with computation(PARALLEL), interval(...):
        outp = inp  # noqa F841: local variable 'outp' is assigned to but never used


class TestMultipleStencilsPartialExpansionIJ:
    def test_no_dependency(self):
        inp1 = np.ones((5, 5, 5))
        inp2 = np.ones((5, 5, 5))
        outp1 = np.ones((3, 3, 3))
        outp2 = np.ones((3, 3, 3))

        @dace.program
        def orchestration():
            stencil_inextents_1(inp1=inp1, outp1=outp1)
            stencil_inextents_2(inp2=inp2, outp2=outp2)

        sdfg = orchestration.to_sdfg()
        partially_expand(sdfg, dims="IJ")
        assert (
            len(
                list(
                    filter(
                        lambda n: isinstance(n[0], dace.nodes.MapEntry), sdfg.all_nodes_recursive()
                    )
                )
            )
            == 1
        )

    def test_dependency_no_extent_name_mix(self):
        inp = np.ones((3, 3, 3))
        outp = np.ones((3, 3, 3))
        tmp = np.ones((3, 3, 3))

        @dace.program
        def orchestration():
            stencil_noextents(inp=inp, outp=tmp)
            stencil_noextents(inp=tmp, outp=outp)

        sdfg = orchestration.to_sdfg()
        partially_expand(sdfg, dims="IJ")
        assert (
            len(
                list(
                    filter(
                        lambda n: isinstance(n[0], dace.nodes.MapEntry), sdfg.all_nodes_recursive()
                    )
                )
            )
            == 1
        )

    def test_dependency_extent_name_mix_apply(self):
        inp = np.ones((3, 3, 3))
        tmp = np.ones((3, 3, 3))
        out1 = np.ones((3, 3, 3))
        out2 = np.ones((3, 3, 3))

        @dace.program
        def orchestration():
            stencil_inextents_3(field_a=inp, field_b=out1, tmp=tmp)
            stencil_inextents_4(field_b=inp, field_c=out2, tmp=tmp)

        sdfg = orchestration.to_sdfg()
        partially_expand(sdfg, dims="IJ")

        assert (
            len(
                list(
                    filter(
                        lambda n: isinstance(n[0], dace.nodes.MapEntry), sdfg.all_nodes_recursive()
                    )
                )
            )
            == 1
        )

    def test_dependency_extent_name_mix_noapply(self):
        inp = np.ones((3, 3, 3))
        outp = np.ones((3, 3, 3))
        tmp = np.ones((3, 3, 3))

        @dace.program
        def orchestration(tmp: dace.float64[3, 3, 3]):  # type: ignore[name-defined]
            stencil_inextents_1(inp1=inp, outp1=tmp)
            stencil_inextents_1(inp1=tmp, outp1=outp)

        sdfg = orchestration.to_sdfg(tmp=tmp)
        sdfg.arrays["tmp"].transient = True
        partially_expand(sdfg, dims="IJ")
        assert (
            len(
                list(
                    filter(
                        lambda n: isinstance(n[0], dace.nodes.MapEntry), sdfg.all_nodes_recursive()
                    )
                )
            )
            == 0
        )


class TestDimsEffective:
    """Test that the dimensions parameter of partially_expand is effective."""

    @pytest.mark.parametrize("dimensions", ["I", "J", "IJ"])
    def test_tile_dims(self, dimensions):
        sdfg = get_sdfg("copy_stencil", backend="dace:cpu")

        partially_expand(sdfg, dims=dimensions)

        maps = list(
            node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)
        )
        assert set(maps[0].params) == set(dcir.Axis(d).tile_symbol() for d in dimensions)

    def test_not_tile_empty_dims(self):
        sdfg = get_sdfg("copy_stencil", backend="dace:cpu")

        partially_expand(sdfg, dims=[])

        maps = list(
            node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)
        )
        assert len(maps) == 0

    def test_dependency_extent_name_mix_apply(self):
        inp = np.ones((3, 3, 3))
        outp = np.ones((3, 3, 3))
        tmp = np.ones((3, 3, 3))

        @dace.program
        def orchestration(tmp: dace.float64[3, 3, 3]):  # type: ignore[name-defined]
            stencil_inextents_1(inp1=inp, outp1=tmp)
            stencil_inextents_1(inp1=tmp, outp1=outp)

        sdfg = orchestration.to_sdfg(tmp=tmp)
        sdfg.arrays["tmp"].transient = True
        partially_expand(sdfg, dims="I")
        maps = list(
            node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)
        )
        assert len(maps) == 1

    def test_dependency_extent_name_mix_noapply(self):
        inp = np.ones((3, 3, 3))
        outp = np.ones((3, 3, 3))
        tmp = np.ones((3, 3, 3))

        @dace.program
        def orchestration(tmp: dace.float64[3, 3, 3]):  # type: ignore[name-defined]
            stencil_inextents_1(inp1=inp, outp1=tmp)
            stencil_inextents_1(inp1=tmp, outp1=outp)

        sdfg = orchestration.to_sdfg(tmp=tmp)
        sdfg.arrays["tmp"].transient = True
        partially_expand(sdfg, dims="J")
        maps = list(
            node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)
        )
        assert len(maps) == 0


@pytest.mark.parametrize("stencil_name", REGISTRY.keys())
def test_cpu_single_map(stencil_name):
    """Test partial expansion captures full gtscript stencils.

    When applying the partial expansion to any gtscript function, the resulting SDFG should be a single Map with
    schedule CPU_Multicore (resulting in a single `#pragma omp parallel for` on the top level in the generated code.)
    Technically, this is not an invariant fo the current implementation e.g. in the case of some horizontal regions, but
    none of the stencils in REGISTRY have that property.
    """
    sdfg = get_sdfg(stencil_name, backend="dace:cpu")

    partially_expand(sdfg, dims="IJ")

    maps = list(
        node for node, _ in sdfg.all_nodes_recursive() if isinstance(node, dace.nodes.MapEntry)
    )

    assert len(maps) == 1
    assert maps[0].schedule == dace.ScheduleType.CPU_Multicore
