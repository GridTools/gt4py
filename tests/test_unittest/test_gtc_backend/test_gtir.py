import pytest
from eve import SourceLocation

from gt4py.backend.gtc_backend.common import DataType, LevelMarker, LoopOrder
from gt4py.backend.gtc_backend.gtir import (
    AssignStmt,
    AxisBound,
    Computation,
    FieldAccess,
    FieldDecl,
    HorizontalLoop,
    Stencil,
    VerticalInterval,
    VerticalLoop,
)


@pytest.fixture
def copy_assign():
    yield AssignStmt(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        left=FieldAccess(name="a", loc=SourceLocation(line=2, column=0, source="copy_gtir")),
        right=FieldAccess(name="b", loc=SourceLocation(line=2, column=2, source="copy_gtir")),
    )


@pytest.fixture
def copy_h_loop(copy_assign):
    yield HorizontalLoop(
        loc=SourceLocation(line=2, column=0, source="copy_gtir"), stmt=copy_assign
    )


@pytest.fixture
def copy_interval(copy_h_loop):
    yield VerticalInterval(
        loc=SourceLocation(line=1, column=10, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
        horizontal_loops=[copy_h_loop],
    )


@pytest.fixture
def copy_v_loop(copy_interval):
    yield VerticalLoop(
        loc=SourceLocation(line=1, column=0, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        vertical_intervals=[copy_interval],
    )


@pytest.fixture
def copy_stencil(copy_v_loop):
    yield Stencil(
        loc=SourceLocation(line=0, column=0, source="copy_gtir"), vertical_loops=[copy_v_loop]
    )


@pytest.fixture
def copy_computation(copy_stencil):
    yield Computation(
        name="copy_gtir",
        loc=SourceLocation(line=0, column=0, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        stencils=[copy_stencil],
    )


def test_copy(copy_computation):
    assert copy_computation
