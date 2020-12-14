from typing import Iterator

import pytest

from gt4py.gtc import common, gtir
from gt4py.gtc.python import pnir
from gt4py.gtc.python.gtir_to_pnir import GtirToPnir


@pytest.fixture
def gtir_to_pnir() -> Iterator[GtirToPnir]:
    yield GtirToPnir()


@pytest.fixture
def copies_vertical_loop() -> Iterator[gtir.VerticalLoop]:
    yield gtir.VerticalLoop(
        loop_order=common.LoopOrder.PARALLEL,
        interval=gtir.Interval(
            start=gtir.AxisBound.from_start(offset=1),
            end=gtir.AxisBound.from_end(offset=2),
        ),
        body=[
            gtir.ParAssignStmt(
                left=gtir.FieldAccess.centered(name="a"),
                right=gtir.FieldAccess.centered(name="b"),
            ),
            gtir.ParAssignStmt(
                left=gtir.FieldAccess.centered(name="b"),
                right=gtir.FieldAccess(name="a", offset=gtir.CartesianOffset(i=2, j=1, k=0)),
            ),
        ],
        temporaries=[],
    )


def test_computation(copies_vertical_loop: gtir.VerticalLoop, gtir_to_pnir: GtirToPnir) -> None:
    inp = gtir.Stencil(
        name="test_computation",
        params=[
            gtir.FieldDecl(name="a", dtype=common.DataType.INT8),
            gtir.FieldDecl(name="b", dtype=common.DataType.FLOAT64),
        ],
        vertical_loops=[copies_vertical_loop, copies_vertical_loop],
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


def test_vertical_loop(copies_vertical_loop: gtir.VerticalLoop, gtir_to_pnir: GtirToPnir) -> None:
    out = gtir_to_pnir.visit(copies_vertical_loop)
    assert len(out) == 1
    assert isinstance(out[0], pnir.KLoop)


def test_vertical_interval(
    copies_vertical_loop: gtir.VerticalLoop, gtir_to_pnir: GtirToPnir
) -> None:
    vertical_interval = copies_vertical_loop.interval
    out_lower, out_upper = gtir_to_pnir.visit(vertical_interval)
    assert isinstance(out_lower, gtir.AxisBound)
    assert isinstance(out_upper, gtir.AxisBound)
    assert out_lower.offset == 1
    assert out_lower.level == common.LevelMarker.START
    assert out_upper.offset == 2
    assert out_upper.level == common.LevelMarker.END
