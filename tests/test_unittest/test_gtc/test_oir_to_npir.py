# -*- coding: utf-8 -*-
import pytest

from gtc import common, oir
from gtc.python import npir
from gtc.python.oir_to_npir import OirToNpir


class VerticalLoopBuilder:
    def __init__(self):
        self._start = oir.AxisBound.start()
        self._end = oir.AxisBound.end()
        self._horizontal_executions = []
        self._loop_order = common.LoopOrder.PARALLEL
        self._declarations = []

    def build(self):
        return oir.VerticalLoop(
            interval=oir.Interval(
                start=self._start,
                end=self._end,
            ),
            horizontal_executions=self._horizontal_executions,
            loop_order=self._loop_order,
            declarations=self._declarations,
        )

    def add_horizontal_execution(self, h_exec: oir.HorizontalExecution) -> "VerticalLoopBuilder":
        self._horizontal_executions.append(h_exec)
        return self


class HorizontalExecutionBuilder:
    def __init__(self):
        self._body = []
        self._mask = None

    def build(self):
        return oir.HorizontalExecution(
            body=[],
            mask=None,
        )

    def add_assign(self, assign: oir.AssignStmt) -> "HorizontalExecutionBuilder":
        self._body.append(assign)
        return self


@pytest.fixture(params=[True, False])
def parallel_k(request):
    yield request.param


def test_stencil_to_computation():
    stencil = oir.Stencil(
        name="stencil",
        params=[
            oir.FieldDecl(
                name="a",
                dtype=common.DataType.FLOAT64,
            ),
            oir.ScalarDecl(
                name="b",
                dtype=common.DataType.INT32,
            ),
        ],
        vertical_loops=[VerticalLoopBuilder().build()],
    )
    computation = OirToNpir().visit(stencil)

    assert computation.field_params == ["a"]
    assert computation.params == ["a", "b"]
    assert len(computation.vertical_passes) == 1


def test_vertical_loop_to_vertical_pass():
    vertical_loop = VerticalLoopBuilder().build()
    vertical_pass = OirToNpir().visit(vertical_loop)

    assert vertical_pass.body == []


def test_horizontal_execution_to_vector_assigns():
    horizontal_execution = HorizontalExecutionBuilder().build()
    horizontal_region = OirToNpir().visit(horizontal_execution)
    assert horizontal_region.body == []
    assert horizontal_region.padding.lower == (0, 0, 0)
    assert horizontal_region.padding.upper == (0, 0, 0)


def test_assign_stmt_to_vector_assign(parallel_k):
    assign_stmt = oir.AssignStmt(
        left=oir.FieldAccess(
            name="a", offset=common.CartesianOffset.zero(), dtype=common.DataType.FLOAT64
        ),
        right=oir.FieldAccess(
            name="b", offset=common.CartesianOffset(i=-1, j=22, k=0), dtype=common.DataType.FLOAT64
        ),
    )

    ctx = OirToNpir.ComputationContext()
    v_assign = OirToNpir().visit(
        assign_stmt, ctx=ctx, h_ctx=OirToNpir.HorizontalContext(), parallel_k=parallel_k, mask=None
    )
    assert isinstance(v_assign, npir.VectorAssign)
    assert v_assign.left.k_offset.parallel is parallel_k
    assert v_assign.right.k_offset.parallel is parallel_k


def test_temp_assign(parallel_k):
    assign_stmt = oir.AssignStmt(
        left=oir.FieldAccess(
            name="a",
            offset=common.CartesianOffset.zero(),
            dtype=common.DataType.FLOAT64,
        ),
        right=oir.FieldAccess(
            name="b", offset=common.CartesianOffset(i=-1, j=22, k=0), dtype=common.DataType.FLOAT64
        ),
    )
    ctx = OirToNpir.ComputationContext(
        symbol_table={"a": oir.Temporary(name="a", dtype=common.DataType.FLOAT64)}
    )
    _ = OirToNpir().visit(
        assign_stmt, ctx=ctx, h_ctx=OirToNpir.HorizontalContext(), parallel_k=parallel_k, mask=None
    )
    assert len(ctx.temp_defs) == 1
    assert isinstance(ctx.temp_defs["a"].left, npir.VectorTemp)
    assert isinstance(ctx.temp_defs["a"].right, npir.EmptyTemp)


def test_field_access_to_field_slice(parallel_k):
    field_access = oir.FieldAccess(
        name="a",
        offset=common.CartesianOffset(i=-1, j=2, k=0),
        dtype=common.DataType.FLOAT64,
    )

    ctx = OirToNpir.ComputationContext()
    h_ctx = OirToNpir.HorizontalContext()
    parallel_field_slice = OirToNpir().visit(
        field_access, ctx=ctx, h_ctx=h_ctx, parallel_k=parallel_k
    )
    assert parallel_field_slice.k_offset.parallel is parallel_k
    assert parallel_field_slice.i_offset.offset.value == -1
    assert ctx.domain_padding.lower[0] == 1
    assert ctx.domain_padding.upper[1] == 2
    assert h_ctx.padding.lower[0] == 1
    assert h_ctx.padding.upper[1] == 2

    # ctx.domain_padding should be correctly extended when visiting another field access
    OirToNpir().visit(
        oir.FieldAccess(
            name="b",
            offset=common.CartesianOffset(i=-2, j=-1, k=4),
            dtype=common.DataType.FLOAT64,
        ),
        ctx=ctx,
        h_ctx=h_ctx,
        parallel_k=parallel_k,
    )
    assert ctx.domain_padding.lower[0] == 2
    assert ctx.domain_padding.lower[1] == 1
    assert h_ctx.padding.lower[0] == 2
    assert h_ctx.padding.lower[1] == 1
    assert ctx.domain_padding.upper[1] == 2
    assert ctx.domain_padding.upper[2] == 4
    assert h_ctx.padding.upper[1] == 2
    assert h_ctx.padding.upper[2] == 0  # horizontal context does not contain vertical padding


def test_binary_op_to_vector_arithmetic():
    binop = oir.BinaryOp(
        op=common.ArithmeticOperator.ADD,
        left=oir.Literal(dtype=common.DataType.INT32, value="2"),
        right=oir.Literal(dtype=common.DataType.INT32, value="2"),
    )
    result = OirToNpir().visit(binop)
    assert isinstance(result, npir.VectorArithmetic)
    assert isinstance(result.left, npir.BroadCast)
    assert isinstance(result.right, npir.BroadCast)


@pytest.mark.parametrize("broadcast", [True, False])
def test_literal(broadcast):
    gtir_literal = oir.Literal(value="42", dtype=common.DataType.INT32)
    result = OirToNpir().visit(gtir_literal, broadcast=broadcast)
    npir_literal = result
    if broadcast:
        assert isinstance(result, npir.BroadCast)
        assert isinstance(result, npir.VectorExpression)
        npir_literal = result.expr
    assert gtir_literal.dtype == npir_literal.dtype
    assert gtir_literal.kind == npir_literal.kind
    assert gtir_literal.value == npir_literal.value


@pytest.mark.parametrize("broadcast", [True, False])
def test_cast(broadcast):
    itof = oir.Cast(
        dtype=common.DataType.FLOAT64, expr=oir.Literal(value="42", dtype=common.DataType.INT32)
    )
    result = OirToNpir().visit(itof, broadcast=broadcast)
    assert isinstance(result, npir.BroadCast if broadcast else npir.Cast)
    cast = result
    if broadcast:
        assert isinstance(result, npir.BroadCast)
        assert isinstance(result, npir.VectorExpression)
        cast = result.expr
    assert cast.dtype == itof.dtype
    assert cast.expr.value == "42"


def test_native_func_call():
    oir_node = oir.NativeFuncCall(
        func=common.NativeFunction.SQRT,
        args=[
            oir.FieldAccess(
                name="a",
                offset=common.CartesianOffset.zero(),
                dtype=common.DataType.FLOAT64,
            ),
        ],
    )
    result = OirToNpir().visit(
        oir_node,
        parallel_k=True,
        ctx=OirToNpir.ComputationContext(),
        h_ctx=OirToNpir.HorizontalContext(),
    )
    assert isinstance(result, npir.VectorExpression)
