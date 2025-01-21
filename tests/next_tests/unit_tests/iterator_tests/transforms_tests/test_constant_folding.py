# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding



def test_constant_folding_plus():
    expected = im.literal_from_value(2)
    testee = im.plus(
        im.literal_from_value(1), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_boolean():
    testee = im.not_(im.literal_from_value(True))
    expected = im.literal_from_value(False)

    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_math_op():
    expected = im.literal_from_value(13)
    testee = im.plus(
        im.literal_from_value(4),
        im.plus(
            im.literal_from_value(7), im.minus(im.literal_from_value(7), im.literal_from_value(5))
        ),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_if():
    expected = im.plus("a", 2)
    testee = im.if_(
        im.literal_from_value(True),
        im.plus(im.ref("a"), im.literal_from_value(2)),
        im.minus(im.literal_from_value(9), im.literal_from_value(5)),
    )
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_minimum():
    testee = im.call("minimum")("a", "a")
    expected = im.ref("a")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal_plus0():
    testee = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(im.literal_from_value(1), im.ref("__out_size_1"))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_literal_minus0():
    testee = im.minus(im.ref("__out_size_1"), im.literal_from_value(1))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.minus(im.literal_from_value(1), im.ref("__out_size_1"))
    expected = im.minus(im.literal_from_value(1), im.ref("__out_size_1"))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_funcall_literal():
    testee = im.plus(im.plus(im.ref("__out_size_1"), im.literal_from_value(1)), im.literal_from_value(1))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(im.literal_from_value(1), im.plus(im.ref("__out_size_1"), im.literal_from_value(1)))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected
def test_constant_folding_maximum_literal_minus():
    testee = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

    testee = im.plus(im.literal_from_value(1), im.ref("__out_size_1"))
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus1():
    testee = im.call("maximum")(
        im.call("maximum")(im.ref("__out_size_1"), im.literal_from_value(1)),
        im.literal_from_value(1),
    )
    expected = im.call("maximum")(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus2():
    testee = im.call("maximum")(
        im.call("maximum")(im.literal_from_value(1), im.ref("__out_size_1")),
        im.literal_from_value(1),
    )
    expected = im.call("maximum")(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus3():
    testee = im.call("maximum")(
        im.call("maximum")(im.literal_from_value(1), im.ref("__out_size_1")),
        im.call("maximum")(im.literal_from_value(1), im.ref("__out_size_1")),
    )
    expected = im.call("maximum")(im.ref("__out_size_1"),im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus4():
    testee = im.call("maximum")(
        im.call("maximum")(im.literal_from_value(1), im.ref("__out_size_1")),
        im.call("maximum")(im.ref("__out_size_1"), im.literal_from_value(1)),
    )
    expected = im.call("maximum")(im.ref("__out_size_1"),im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus5():
    testee = im.call("maximum")(
        im.ref("__out_size_1"), im.call("maximum")(im.literal_from_value(1), im.ref("__out_size_1"))
    )
    expected = im.call("maximum")(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus6():
    testee = im.call("maximum")(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)),
        im.plus(im.ref("__out_size_1"), im.literal_from_value(0)),
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus7():
    testee = im.minus(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)),
        im.plus(im.literal_from_value(1), im.literal_from_value(1)),
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus8():
    testee = im.plus(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)),
        im.plus(im.literal_from_value(1), im.literal_from_value(1)),
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(3))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus9():
    testee = im.call("maximum")(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)),
        im.plus(
            im.plus(im.ref("__out_size_1"), im.literal_from_value(1)), im.literal_from_value(0)
        ),
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus10():
    testee = im.call("maximum")(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)), im.ref("__out_size_1")
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus10a():
    testee =  im.plus(im.ref("__out_size_1"), im.call("maximum")(im.literal_from_value(0), im.literal_from_value(-1)))

    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus11():
    testee = im.call("maximum")(
        im.ref("__out_size_1"), im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus12():
    testee = im.call("maximum")(
        im.ref("__out_size_1"), im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus13():
    testee = im.call("maximum")(
        im.minus(im.ref("__out_size_1"), im.literal_from_value(1)), im.ref("__out_size_1")
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus14():
    testee = im.call("maximum")(
        im.ref("__out_size_1"), im.minus(im.ref("__out_size_1"), im.literal_from_value(1))
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus15():
    testee = im.call("maximum")(
        im.ref("__out_size_1"), im.minus(im.ref("__out_size_1"), im.literal_from_value(-1))
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus16():
    testee = im.call("minimum")(
        im.plus(im.ref("__out_size_1"), im.literal_from_value(1)), im.ref("__out_size_1")
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus17():
    testee = im.call("minimum")(
        im.ref("__out_size_1"), im.plus(im.ref("__out_size_1"), im.literal_from_value(1))
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus18():
    testee = im.call("minimum")(
        im.ref("__out_size_1"), im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus19():
    testee = im.call("minimum")(
        im.minus(im.ref("__out_size_1"), im.literal_from_value(1)), im.ref("__out_size_1")
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus20():
    testee = im.call("minimum")(
        im.ref("__out_size_1"), im.minus(im.ref("__out_size_1"), im.literal_from_value(1))
    )
    expected = im.plus(im.ref("__out_size_1"), im.literal_from_value(-1))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_maximum_literal_plus21():
    testee = im.call("minimum")(
        im.ref("__out_size_1"), im.minus(im.ref("__out_size_1"), im.literal_from_value(-1))
    )
    expected = im.ref("__out_size_1")
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal():
    testee = im.plus(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(3)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_literal_maximum():
    testee = im.call("maximum")(im.literal_from_value(1), im.literal_from_value(2))
    expected = im.literal_from_value(2)
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_complex():
     # 1 - max(max(1, max(1, sym), min(1, sym), sym), 1 + (min(-1, 2) + max(-1, 1 - sym)))
    testee = im.minus(im.literal_from_value(1), im.call("maximum")(im.call("maximum")(im.literal_from_value(1), im.call("maximum")(im.literal_from_value(1), im.ref("sym")),  im.call("minimum")(im.literal_from_value(1), im.ref("sym")), im.ref("sym")), im.plus(im.literal_from_value(1), im.plus(im.call("minimum")(im.literal_from_value(-1), 2 ), im.call("maximum")(im.literal_from_value(-1), im.minus (im.literal_from_value(1), im.ref("sym")))))))
     # 1 - max(max(sym, 1), max(1 - sym, -1))
    expected = im.minus(im.literal_from_value(1),im.call("maximum")(im.call("maximum")(im.ref("sym"), im.literal_from_value(1)), im.call("maximum")(im.minus(im.literal_from_value(1), im.ref("sym")),im.literal_from_value(-1))))
    actual = ConstantFolding.apply(testee)
    assert actual == expected



#( (min(1 - sym, 1 + sym) + (max(max(1 - sym, 1 + sym),1 - sym)  + max(1 - sym, 1 - sym)))))) - 2
 #max(sym, 1 + sym) + (max(1, max(1,  sym)) + (sym - 1 + (1 + (sym + 1) + 1))) - 2

def test_constant_folding_complex_1():
    sym = im.ref("sym")
    # maximum(sym, 1 + sym) + (maximum(1, maximum(1, sym)) + (sym - 1 + (1 + (sym + 1) + 1))) - 2
    testee = im.minus(im.plus(im.call("maximum")(sym, im.plus(im.literal_from_value(1), sym)), im.plus(im.call("maximum")(im.literal_from_value(1), im.call("maximum")(im.literal_from_value(1),sym)),im.plus(im.minus(sym,im.literal_from_value(1)), im.plus(im.plus(im.literal_from_value(1),im.plus(sym,im.literal_from_value(1))),im.literal_from_value(1)) ))) , im.literal_from_value(2))
    # sym + 1 + (maximum(sym, 1) + (sym + sym + 2)) + -2
    expected = im.plus(im.plus(im.plus(sym,1),im.plus( im.call("maximum")(sym,im.literal_from_value(1)),im.plus(im.plus(sym, sym),im.literal_from_value(2)))), im.literal_from_value(-2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_complex_3():
    sym = im.ref("sym")
    # minimum(1 - sym, 1 + sym) + (maximum(maximum(1 - sym, 1 + sym), 1 - sym) + maximum(1 - sym, 1 - sym))
    testee = im.plus(im.call("minimum")(im.minus(im.literal_from_value(1), sym), im.plus(im.literal_from_value(1), sym)), im.plus(im.call("maximum")(im.call("maximum")(im.minus(im.literal_from_value(1), sym), im.plus(im.literal_from_value(1), sym)), im.minus(im.literal_from_value(1), sym)),im.call("maximum")(im.minus(im.literal_from_value(1), sym), im.minus(im.literal_from_value(1), sym))))
    # minimum(1 - sym, sym + 1) + (maximum(sym + 1, 1 - sym) + (1 - sym))
    expected = im.plus(im.call("minimum")(im.minus(im.literal_from_value(1), sym), im.plus(sym, im.literal_from_value(1))), im.plus(im.call("maximum")(im.plus(sym, im.literal_from_value(1)), im.minus(im.literal_from_value(1), sym)),im.minus(im.literal_from_value(1), sym)))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

def test_constant_folding_complex_3a():
    sym = im.ref("sym")
    # maximum(maximum(1 + sym, 1), 1 + sym)
    testee = im.call("maximum")(im.call("maximum")( im.plus(im.literal_from_value(1), sym), 1),   im.plus(im.literal_from_value(1), sym))
    # maximum(1 + sym, 1)
    expected =im.call("maximum")( im.plus(sym, im.literal_from_value(1)), 1)
    actual = ConstantFolding.apply(testee)
    assert actual == expected


def test_constant_folding_complex_2():
    sym = im.ref("sym")
    testee = im.plus(im.plus(sym, im.literal_from_value(-1)),im.plus(sym, im.literal_from_value(3)))
    expected = im.plus(im.plus(sym, sym), im.literal_from_value(2))
    actual = ConstantFolding.apply(testee)
    assert actual == expected

#sym + 1 + (maximum(sym, 1) + (sym + -1 + (sym + 3))) + -2
#maximum(1, sym) + (3 × sym + 1)




def test_constant_folding_complex_4():
    sym = im.ref("sym", "float32")
    testee = im.divides_(im.minus(im.literal_from_value(1), sym), im.minus( im.literal_from_value(2), sym))
    expected = im.divides_(im.minus(im.literal_from_value(1), sym), im.minus( im.literal_from_value(2), sym))
    actual = ConstantFolding.apply(testee)
    assert actual == expected




