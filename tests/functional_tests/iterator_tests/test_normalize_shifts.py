from functional.iterator import ir
from functional.iterator.transforms.normalize_shifts import NormalizeShifts


# TODO factory
def make_shift(*offsets):
    def impl(*iters):
        return ir.FunCall(
            fun=ir.FunCall(
                fun=ir.SymRef(id="shift"),
                args=[*offsets],
            ),
            args=[
                *iters,
            ],
        )

    return impl


def make_deref(it):
    return ir.FunCall(fun=ir.SymRef(id="deref"), args=[it])


# shift(offset1)(shift(offset2)(it)) -> shift(offset2, offset1)(it)
def test_trivial_shifts():
    testee = make_shift(ir.OffsetLiteral(value="offset1"))(
        make_shift(ir.OffsetLiteral(value="offset2"))(ir.SymRef(id="it"))
    )
    expected = make_shift(ir.OffsetLiteral(value="offset2"), ir.OffsetLiteral(value="offset1"))(
        ir.SymRef(id="it")
    )

    actual = NormalizeShifts().visit(testee)

    assert actual == expected


# shift(offset0)(deref(shift(offset1)(shift(offset2)(it)))) -> shift(offset0)(deref(shift(offset2, offset1)(it)))
def test_shifts_with_sparse_deref():
    testee = make_shift(ir.OffsetLiteral(value="offset0"))(
        make_deref(
            make_shift(ir.OffsetLiteral(value="offset1"))(
                make_shift(ir.OffsetLiteral(value="offset2"))(ir.SymRef(id="it"))
            )
        )
    )
    expected = make_shift(ir.OffsetLiteral(value="offset0"))(
        make_deref(
            make_shift(ir.OffsetLiteral(value="offset2"), ir.OffsetLiteral(value="offset1"))(
                ir.SymRef(id="it")
            )
        )
    )

    actual = NormalizeShifts().visit(testee)

    assert actual == expected
