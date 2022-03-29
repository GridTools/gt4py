from functional.iterator.builtins import *
from functional.iterator.runtime import *
from functional.iterator.tracing import trace
from functional.iterator.transforms.common_type_deduction import CommonTypeDeduction, unify


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")


def clos(*args):
    return closure(domain(named_range(IDim, 0, 3), named_range(JDim, 0, 5)), *args)


def test_simple():
    def fencil(a, b, c):
        clos(lambda a, b: plus(deref(a), deref(b)), c, [a, b])

    nodes = trace(fencil, [None] * 3)

    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": (0, 0, 0)}
    assert testee == expected


def test_conditional():
    def fencil(a, b, c, d):
        clos(lambda a, b, c: if_(deref(a), deref(b), deref(c)), d, [a, b, c])

    nodes = trace(fencil, [None] * 4)

    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": ("bool", 0, 0, 0)}
    assert testee == expected


def test_lift():
    def fencil(a, b):
        clos(lambda a: deref(lift(lambda a: deref(a))(a)), b, [a])

    nodes = trace(fencil, [None] * 2)

    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": (0, 0)}
    assert testee == expected


def test_scan():
    def fencil(a, b):
        clos(lambda a: scan(lambda acc, a: acc + deref(a), False, 0.0)(a), b, [a])

    nodes = trace(fencil, [None] * 2)
    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": ("float", "float")}
    assert testee == expected


def test_make_tuple():
    def fencil(a, b, c):
        clos(lambda a, b: make_tuple(deref(a), deref(b)), c, [a, b])

    nodes = trace(fencil, [None] * 3)
    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": (0, 1, (0, 1))}
    assert testee == expected


def test_tuple_get():
    def fencil(a, b):
        clos(lambda a: tuple_get(1, deref(a)), b, [a])

    nodes = trace(fencil, [None] * 2)
    testee = CommonTypeDeduction.apply(nodes)
    expected = {"fencil": ((0, 1), 1)}
    assert testee == expected
