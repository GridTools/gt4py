from typing import Iterator

import pytest

from gt4py.backend.gtc_backend import common, gtir, pnir
from gt4py.backend.gtc_backend.gtir_to_pnir import GtirToPnir


@pytest.fixture
def gtir_to_pnir() -> Iterator[GtirToPnir]:
    yield GtirToPnir()


@pytest.fixture
def copies_stencil() -> Iterator[gtir.Stencil]:
    yield gtir.Stencil(
        vertical_loops=[
            gtir.VerticalLoop(
                loop_order=common.LoopOrder.PARALLEL,
                vertical_intervals=[
                    gtir.VerticalInterval(
                        start=gtir.AxisBound.from_start(offset=1),
                        end=gtir.AxisBound.from_end(offset=2),
                        horizontal_loops=[
                            gtir.HorizontalLoop(
                                stmt=gtir.AssignStmt(
                                    left=gtir.FieldAccess.centered(name="a"),
                                    right=gtir.FieldAccess.centered(name="b"),
                                )
                            ),
                            gtir.HorizontalLoop(
                                stmt=gtir.AssignStmt(
                                    left=gtir.FieldAccess.centered(name="b"),
                                    right=gtir.FieldAccess(
                                        name="a", offset=gtir.CartesianOffset(i=2, j=1, k=0)
                                    ),
                                )
                            ),
                        ],
                    )
                ],
            )
        ]
    )


def test_computation(copies_stencil: gtir.Stencil, gtir_to_pnir: GtirToPnir) -> None:
    inp = gtir.Computation(
        name="test_computation",
        params=[
            gtir.FieldDecl(name="a", dtype=common.DataType.INT8),
            gtir.FieldDecl(name="b", dtype=common.DataType.FLOAT64),
        ],
        stencils=[copies_stencil, copies_stencil],
        fields_metadata=gtir.FieldsMetadata(
            metas={
                "a": gtir.FieldMetadataBuilder()
                .name("a")
                .dtype(common.DataType.AUTO)
                .access(gtir.AccessKind.READ_WRITE)
                .build(),
                "b": gtir.FieldMetadataBuilder()
                .name("b")
                .dtype(common.DataType.AUTO)
                .access(gtir.AccessKind.READ_WRITE)
                .build(),
            }
        ),
    )
    out = gtir_to_pnir.visit(inp)
    assert isinstance(out, pnir.Stencil)
    assert isinstance(out.stencil_obj, pnir.StencilObject)
    assert isinstance(out.computation, pnir.Module)
    assert isinstance(out.computation.run, pnir.RunFunction)
    assert out.computation.run.field_params == ["a", "b"]
    assert out.computation.run.scalar_params == []
    assert len(out.computation.run.k_loops) == 2
    assert isinstance(out.computation.run.k_loops[0], pnir.KLoop)
    assert isinstance(out.computation.run.k_loops[1], pnir.KLoop)


def test_stencil(copies_stencil: gtir.Stencil, gtir_to_pnir: GtirToPnir) -> None:
    out = gtir_to_pnir.visit(copies_stencil)
    assert len(out) == 1
    assert isinstance(out[0], pnir.KLoop)


def test_vertical_loop(copies_stencil, gtir_to_pnir) -> None:
    out = gtir_to_pnir.visit(copies_stencil.vertical_loops[0])
    assert len(out) == 1
    assert isinstance(out[0], pnir.KLoop)


def test_vertical_interval(copies_stencil: gtir.Stencil, gtir_to_pnir: GtirToPnir) -> None:
    vertical_interval = copies_stencil.vertical_loops[0].vertical_intervals[0]
    out = gtir_to_pnir.visit(vertical_interval)
    assert isinstance(out, pnir.KLoop)
    assert isinstance(out.lower, gtir.AxisBound)
    assert isinstance(out.upper, gtir.AxisBound)
    assert out.lower.offset == 1
    assert out.lower.level == common.LevelMarker.START
    assert out.upper.offset == 2
    assert out.upper.level == common.LevelMarker.END
    assert len(out.ij_loops) == 2
    assert isinstance(out.ij_loops[0], pnir.IJLoop)
