from gt4py.next.ffront.type_system_2 import inference, types as types_f
from gt4py.next.type_system_2 import types
import gt4py.next as gtx

IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")


def test_annotation_field():
    result = inference.inferrer.from_annotation(gtx.Field[[IDim, JDim], float])
    assert isinstance(result, types_f.FieldType)
    assert isinstance(result.element_type, types.FloatType)
    assert result.dimensions == {IDim, JDim}
