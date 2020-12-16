import pytest

from gt4py.gtc import common, oir
from gt4py.gtc.python import npir
from gt4py.gtc.python.oir_to_npir import OirToNpir


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


class HorizontalExecutionBuilder:
    def __init__(self):
        self._body = []
        self._mask = None

    def build(self):
        return oir.HorizontalExecution(
            body=[],
            mask=None,
        )


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
    vector_assigns = OirToNpir().visit(horizontal_execution)
    assert vector_assigns == []


def test_assign_stmt_to_vector_assign(parallel_k):
    assign_stmt = oir.AssignStmt(
        left=oir.FieldAccess(
            name="a", offset=common.CartesianOffset.zero(), dtype=common.DataType.FLOAT64
        ),
        right=oir.FieldAccess(
            name="b", offset=common.CartesianOffset(i=-1, j=22, k=0), dtype=common.DataType.FLOAT64
        ),
    )

    ctx = OirToNpir.Context()
    v_assign = OirToNpir().visit(assign_stmt, ctx=ctx, parallel_k=parallel_k)
    assert isinstance(v_assign, npir.VectorAssign)
    assert v_assign.left.k_offset.parallel is parallel_k
    assert v_assign.right.k_offset.parallel is parallel_k


def test_field_access_to_field_slice(parallel_k):
    field_access = oir.FieldAccess(
        name="a",
        offset=common.CartesianOffset(i=-1, j=2, k=0),
        dtype=common.DataType.FLOAT64,
    )

    ctx = OirToNpir.Context()
    parallel_field_slice = OirToNpir().visit(field_access, ctx=ctx, parallel_k=parallel_k)
    assert parallel_field_slice.k_offset.parallel is parallel_k
    assert parallel_field_slice.i_offset.offset.value == -1
    assert ctx.domain_padding["lower"][0] == 1
    assert ctx.domain_padding["upper"][1] == 2

    # ctx.domain_padding should be correctly extended when visiting another field access
    OirToNpir().visit(
        oir.FieldAccess(
            name="b",
            offset=common.CartesianOffset(i=-2, j=-1, k=4),
            dtype=common.DataType.FLOAT64,
        ),
        ctx=ctx,
        parallel_k=parallel_k,
    )
    assert ctx.domain_padding["lower"][0] == 2
    assert ctx.domain_padding["lower"][1] == 1
    assert ctx.domain_padding["upper"][1] == 2
    assert ctx.domain_padding["upper"][2] == 4
