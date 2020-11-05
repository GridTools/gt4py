import ast

import pytest
from devtools import debug
from eve import SourceLocation

from gtc2.common import BinaryOperator, DataType, LevelMarker, LoopOrder
from gtc2.gtir import (
    AssignStmt,
    AxisBound,
    BinaryOp,
    CartesianOffset,
    Computation,
    FieldAccess,
    FieldDecl,
    HorizontalLoop,
    Stencil,
    VerticalInterval,
    VerticalLoop,
)
from gtc2.python_naive_codegen import PythonNaiveCodegen


@pytest.fixture
def copy_assign():
    yield AssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="a", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="b", loc=SourceLocation(line=3, column=3, source="copy_gtir")
        ),
    )


@pytest.fixture
def copy_h_loop(copy_assign):
    yield HorizontalLoop(
        loc=SourceLocation(line=3, column=1, source="copy_gtir"), stmt=copy_assign
    )


@pytest.fixture
def copy_interval(copy_h_loop):
    yield VerticalInterval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
        horizontal_loops=[copy_h_loop],
    )


@pytest.fixture
def copy_v_loop(copy_interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        vertical_intervals=[copy_interval],
    )


@pytest.fixture
def copy_stencil(copy_v_loop):
    yield Stencil(
        loc=SourceLocation(line=1, column=1, source="copy_gtir"), vertical_loops=[copy_v_loop]
    )


@pytest.fixture
def copy_computation(copy_stencil):
    yield Computation(
        name="copy_gtir",
        loc=SourceLocation(line=1, column=1, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        stencils=[copy_stencil],
    )


def test_copy(copy_computation):
    print(debug(copy_computation))
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


def test_naive_python_copy(copy_computation):
    assert ast.parse(PythonNaiveCodegen.apply(copy_computation))


def test_naive_python_avg():
    horizontal_avg = Computation(
        name="horizontal_avg",
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        stencils=[
            Stencil(
                vertical_loops=[
                    VerticalLoop(
                        loop_order=LoopOrder.FORWARD,
                        vertical_intervals=[
                            VerticalInterval(
                                start=AxisBound(level=LevelMarker.START, offset=0),
                                end=AxisBound(level=LevelMarker.END, offset=0),
                                horizontal_loops=[
                                    HorizontalLoop(
                                        stmt=AssignStmt(
                                            left=FieldAccess.centered(name="a"),
                                            right=BinaryOp(
                                                left=FieldAccess(
                                                    name="b",
                                                    offset=CartesianOffset(i=-1, j=0, k=0),
                                                ),
                                                right=FieldAccess(
                                                    name="b", offset=CartesianOffset(i=1, j=0, k=0)
                                                ),
                                                op=BinaryOperator.ADD,
                                            ),
                                        )
                                    )
                                ],
                            )
                        ],
                    )
                ]
            )
        ],
    )
    assert ast.parse(PythonNaiveCodegen.apply(horizontal_avg))
