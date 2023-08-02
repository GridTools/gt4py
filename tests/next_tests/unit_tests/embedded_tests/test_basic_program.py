import gt4py.next as gtx
import numpy as np

IDim = gtx.Dimension("IDim")
IOff = gtx.FieldOffset("IOff", source=IDim, target=(IDim,))


@gtx.field_operator
def fop(
    a: gtx.Field[[IDim], gtx.float64], b: gtx.Field[[IDim], gtx.float64]
) -> gtx.Field[[IDim], gtx.float64]:
    return a(IOff[1]) + b


@gtx.program
def prog(
    a: gtx.Field[[IDim], gtx.float64],
    b: gtx.Field[[IDim], gtx.float64],
    out: gtx.Field[[IDim], gtx.float64],
):
    fop(a, b, out=out)


def test_basic():
    a = gtx.field(np.asarray([0.0, 1.0, 2.0, 3.0]), domain=((IDim, gtx.common.UnitRange(1, 5)),))
    b = gtx.field(np.asarray([0.0, 1.0, 2.0, 3.0]), domain=((IDim, gtx.common.UnitRange(0, 4)),))
    out = gtx.field(np.asarray([0.0, 0.0, 0.0, 0.0]), domain=((IDim, gtx.common.UnitRange(0, 4)),))

    prog(a, b, out, offset_provider={"IOff": IDim})
    assert False, "Add proper check"
