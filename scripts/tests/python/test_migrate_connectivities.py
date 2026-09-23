#
# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Tests for the ``migrate_connectivities`` dev script."""

from __future__ import annotations

import pathlib
import textwrap

import migrate_connectivities


DIMENSIONS = textwrap.dedent(
    """\
    import gt4py.next as gtx

    KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
    EdgeDim = gtx.Dimension("Edge")
    CellDim = gtx.Dimension("Cell")
    CEDim = gtx.Dimension("CE")
    E2CDim = gtx.Dimension("E2C", gtx.DimensionKind.LOCAL)
    C2EDim = gtx.Dimension("C2E", gtx.DimensionKind.LOCAL)
    E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
    C2E = gtx.FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
    C2CE = gtx.FieldOffset("C2CE", source=CEDim, target=(CellDim, C2EDim))
    Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
    """
)

STENCIL = textwrap.dedent(
    """\
    import gt4py.next as gtx
    from gt4py.next.ffront.experimental import as_offset

    from pkg import dimension as dims
    from pkg.dimension import E2C, KDim, Koff


    def stencil(a, k):
        b = a(Koff[1]) + a(Koff[-1]) + a(Koff[k]) + a(dims.Koff[1])
        return b(as_offset(Koff, k)) + b(as_offset(dims.Koff, k))


    def run(prog, grid):
        prog(offset_provider={"E2C": grid.e2c, "Koff": KDim})
        return KDim.value
    """
)


def _migrate(**sources: str) -> tuple[dict[str, str], list[str]]:
    results, notes = migrate_connectivities.migrate(
        {pathlib.Path(name): source for name, source in sources.items()}
    )
    return {str(path): source for path, source in results.items()}, notes


def test_declarations():
    results, _ = _migrate(dimension=DIMENSIONS)

    assert results["dimension"] == textwrap.dedent(
        """\
        import typing
        import gt4py.next as gtx

        class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...
        class EdgeDim(gtx.DimensionIndex): ...
        class CellDim(gtx.DimensionIndex): ...
        class CEDim(gtx.DimensionIndex): ...
        class E2CDim(gtx.LocalDimensionIndex): ...
        class C2EDim(gtx.LocalDimensionIndex): ...
        class E2C(gtx.NeighborConnectivity[EdgeDim, CellDim]):
            Local: typing.TypeAlias = E2CDim
        class C2E(gtx.NeighborConnectivity[CellDim, EdgeDim]):
            Local: typing.TypeAlias = C2EDim
        class C2CE(gtx.NeighborConnectivity[CellDim, CEDim]):
            Local: typing.TypeAlias = C2EDim
        """
    )


def test_declarations_run():
    results, _ = _migrate(dimension=DIMENSIONS)
    namespace: dict = {"__name__": "migrated_dimension"}
    exec(results["dimension"], namespace)

    assert namespace["C2E"].Local is namespace["C2EDim"]
    assert namespace["C2EDim"].owner is namespace["C2E"]
    # `C2CE` shares `C2E`'s local dimension and is named by its own tag
    assert namespace["C2CE"].Local is namespace["C2EDim"]
    assert namespace["C2CE"].offset_tag == namespace["C2CE"].tag


def test_cartesian_offset_uses_across_modules():
    results, _ = _migrate(dimension=DIMENSIONS, stencil=STENCIL)

    stencil = results["stencil"]
    assert "from pkg.dimension import E2C, KDim\n" in stencil
    assert "a(KDim + 1) + a(KDim - 1) + a(KDim + (k)) + a(dims.KDim + 1)" in stencil
    assert "b(as_offset(KDim, k)) + b(as_offset(dims.KDim, k))" in stencil
    assert "Koff" not in stencil.replace('"Koff"', "")


def test_what_is_left_is_reported():
    _, notes = _migrate(dimension=DIMENSIONS, stencil=STENCIL)

    assert any("offset-provider key 'E2C'" in note for note in notes)
    assert any("offset-provider key 'Koff'" in note for note in notes)
    assert any("'KDim.value'" in note for note in notes)


BARE = textwrap.dedent(
    """\
    from gt4py.next import Dimension, DimensionKind, FieldOffset as FO

    LOCAL = DimensionKind.LOCAL
    Vertex = Dimension("Vertex")
    Edge = Dimension("Edge")
    V2EDim = Dimension("V2E", LOCAL)
    V2E = FO("V2E_TAG", source=Edge, target=(Vertex, V2EDim))
    """
)


def test_unqualified_names_and_aliases():
    results, notes = _migrate(bare=BARE, user='table = {"V2E_TAG": t}\n')
    migrated = results["bare"]

    assert "FO" not in migrated
    assert "from gt4py.next import Dimension, DimensionKind\n" in migrated
    assert (
        "from gt4py.next import DimensionIndex, LocalDimensionIndex, NeighborConnectivity\n"
        in migrated
    )
    assert "class V2EDim(LocalDimensionIndex): ..." in migrated
    assert "class V2E(NeighborConnectivity[Vertex, Edge]):" in migrated
    assert "    Local: typing.TypeAlias = V2EDim" in migrated
    namespace: dict = {"__name__": "migrated_bare"}
    exec(migrated, namespace)
    assert namespace["V2E"].Local is namespace["V2EDim"]
    # a key spelled with the offset's tag is reported, naming the class
    assert any("'V2E_TAG'" in note and "{V2E: table}" in note for note in notes)


def test_shadowed_names_and_all_are_left_alone():
    source = textwrap.dedent(
        """\
        from pkg.dimension import KDim, Koff

        __all__ = ["Koff"]


        def helper(Koff):
            return Koff + 1


        def uses(a):
            return a(Koff[1]) + a(dims.KDim.value)
        """
    )
    results, notes = _migrate(dimension=DIMENSIONS, user=source)

    assert "return Koff + 1" in results["user"]
    assert "a(KDim + 1)" in results["user"]
    assert any("'__all__' lists the removed offset 'Koff'" in note for note in notes)
    assert any("'KDim.value'" in note for note in notes)
