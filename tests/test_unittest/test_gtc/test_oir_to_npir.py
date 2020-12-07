from gt4py.gtc import oir, common
from gt4py.gtc.python import npir
from gt4py.gtc.python.oir_to_npir import OirToNpir


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
        vertical_loops=[]
    )
    computation = OirToNpir().visit(stencil)

    assert computation.field_params == ["a"]
    assert computation.params == ["a", "b"]
