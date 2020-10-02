import pytest

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

from devtools import debug

from gt4py.backend.gtc_backend.python_naive_codegen import PythonNaiveCodegen

@pytest.fixture
def copy_assign():
    yield AssignStmt(left=FieldAccess(name="a"), right=FieldAccess(name="b"))


@pytest.fixture
def copy_h_loop(copy_assign):
    yield HorizontalLoop(stmt=copy_assign)


@pytest.fixture
def copy_interval(copy_h_loop):
    yield VerticalInterval(
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
        horizontal_loops=[copy_h_loop],
    )


@pytest.fixture
def copy_v_loop(copy_interval):
    yield VerticalLoop(loop_order=LoopOrder.FORWARD, vertical_intervals=[copy_interval])


@pytest.fixture
def copy_stencil(copy_v_loop):
    yield Stencil(vertical_loops=[copy_v_loop])


@pytest.fixture
def copy_computation(copy_stencil):
    yield Computation(
        name="copy_gtir",
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        stencils=[copy_stencil],
    )


def test_copy(copy_computation):
    assert copy_computation

def test_naive_python_copy(copy_computation):
    print(PythonNaiveCodegen.apply(copy_computation))
    assert False
