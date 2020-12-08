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
