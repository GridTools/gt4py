from functional.ffront import itir_makers as im
from functional.iterator import ir
from functional.iterator.transforms.propagate_builtins import PropagateBuiltins


def test_deref_proppagation():
    testee = im.deref_(
        im.call_(im.lambda__("inner_it")(im.lift_("stencil")("inner_it")))("outer_it")
    )
    expected = im.call_(im.lambda__("inner_it")(im.deref_(im.lift_("stencil")("inner_it"))))(
        "outer_it"
    )

    actual = PropagateBuiltins.apply(testee)
    assert actual == expected
