from gt4py.gtc import common, oir
from gt4py.gtc.python import npir
from gt4py.gtc.python.oir_to_npir import OirToNpir


EMPTY_VERTICAL_LOOP = oir.VerticalLoop(
    interval=oir.Interval(
        start=oir.AxisBound.start(),
        end=oir.AxisBound.end(),
    ),
    horizontal_executions=[],
    loop_order=common.LoopOrder.PARALLEL,
    declarations=[],
)


EMPTY_HORIZONTAL_EXECUTION = oir.HorizontalExecution(
    body=[],
    mask=None,
)


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
        vertical_loops=[EMPTY_VERTICAL_LOOP],
    )
    computation = OirToNpir().visit(stencil)

    assert computation.field_params == ["a"]
    assert computation.params == ["a", "b"]
    assert len(computation.vertical_passes) == 1


def test_vertical_loop_to_vertical_pass():
    vertical_loop = oir.VerticalLoop(
        interval=oir.Interval(
            start=oir.AxisBound.start(),
            end=oir.AxisBound.end(),
        ),
        horizontal_executions=[],
        loop_order=common.LoopOrder.PARALLEL,
        declarations=[],
    )
    vertical_pass = OirToNpir().visit(vertical_loop)

    assert vertical_pass.body == []


def test_horizontal_execution_to_vector_assigns():
    horizontal_execution = EMPTY_HORIZONTAL_EXECUTION
    vector_assigns = OirToNpir().visit(horizontal_execution)
    assert vector_assigns == []


def test_assign_stmt_to_vector_assign():
    assign_stmt = oir.AssignStmt(
        left=oir.FieldAccess(
            name="a", offset=common.CartesianOffset.zero(), dtype=common.DataType.FLOAT64
        ),
        right=oir.FieldAccess(
            name="b", offset=common.CartesianOffset(i=-1, j=22, k=0), dtype=common.DataType.FLOAT64
        ),
    )

    ctx = OirToNpir.Context().set_parallel_k(value=True)
    v_assign = OirToNpir().visit(assign_stmt, ctx=ctx)
    assert isinstance(v_assign, npir.VectorAssign)
    assert v_assign.left.k_offset.parallel is True
    assert v_assign.right.k_offset.parallel is True


def test_field_access_to_field_slice():
    field_access = oir.FieldAccess(
        name="a",
        offset=common.CartesianOffset(i=-1, j=2, k=0),
        dtype=common.DataType.FLOAT64,
    )

    ctx = OirToNpir.Context()
    # parallel k case
    parallel_ctx = ctx.set_parallel_k(value=True)
    parallel_field_slice = OirToNpir().visit(field_access, ctx=parallel_ctx)
    assert parallel_field_slice.k_offset.parallel is True
    assert parallel_field_slice.i_offset.offset.value == -1
    assert parallel_ctx.domain_padding["lower"][0] == 1
    assert parallel_ctx.domain_padding["upper"][1] == 2

    # sequential k case
    sequential_ctx = ctx.set_parallel_k(value=False)
    sequential_field_slice = OirToNpir().visit(field_access, ctx=sequential_ctx)
    assert sequential_field_slice.k_offset.parallel is False
    assert sequential_field_slice.i_offset.offset.value == -1
    assert parallel_ctx.domain_padding["lower"][0] == 1
    assert parallel_ctx.domain_padding["upper"][1] == 2

    # ctx.domain_padding should be correctly extended when visiting another field access
    OirToNpir().visit(
        oir.FieldAccess(
            name="b",
            offset=common.CartesianOffset(i=-2, j=-1, k=4),
            dtype=common.DataType.FLOAT64,
        ),
        ctx=parallel_ctx,
    )
    assert ctx.domain_padding["lower"][0] == 2
    assert ctx.domain_padding["lower"][1] == 1
    assert ctx.domain_padding["upper"][1] == 2
    assert ctx.domain_padding["upper"][2] == 4
