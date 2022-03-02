from typing import Tuple

import numpy as np
import pytest

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis, closure, fendef, fundef


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")

# semantics of stencil return that is called from the fencil (after `:` the structure of the output)
# `return a` -> a: field
# `return make_tuple(a)` -> (a,): [field] or (field)
# `return a,b` -> (a,b): [field, field] or (field, field)
# `return make_tuple(a,b)` -> (a,b): [field, field]
# `return make_tuple(a), make_tuple(b)` -> ((a,), (b,)): [(field,), (field,)]
# `return make_tuple(make_tuple(a,b))` -> ((a,b)): [(field,field)]


@fundef
def tuple_output1(inp1, inp2):
    return deref(inp1), deref(inp2)


@fundef
def tuple_output2(inp1, inp2):
    return make_tuple(deref(inp1), deref(inp2))


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_tuple_output(backend, stencil):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = (
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
    )

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    stencil[dom](inp1, inp2, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(inp1, out[0])
        assert np.allclose(inp2, out[1])


def test_tuple_of_field_of_tuple_output(backend):
    backend, validate = backend

    @fundef
    def stencil(inp1, inp2, inp3, inp4):
        return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp3 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp4 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out_np1 = np.zeros(shape, dtype="f8, f8")
    out1 = np_as_located_field(IDim, JDim, KDim)(out_np1)
    out_np2 = np.zeros(shape, dtype="f8, f8")
    out2 = np_as_located_field(IDim, JDim, KDim)(out_np2)
    out = (out1, out2)

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    stencil[dom](inp1, inp2, inp3, inp4, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(inp1, out_np1[:]["f0"])
        assert np.allclose(inp2, out_np1[:]["f1"])
        assert np.allclose(inp3, out_np2[:]["f0"])
        assert np.allclose(inp4, out_np2[:]["f1"])


def test_tuple_of_tuple_of_field_output(backend):
    backend, validate = backend

    @fundef
    def stencil(inp1, inp2, inp3, inp4):
        return make_tuple(deref(inp1), deref(inp2)), make_tuple(deref(inp3), deref(inp4))

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp3 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp4 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = (
        (
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        ),
        (
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
            np_as_located_field(IDim, JDim, KDim)(np.zeros(shape)),
        ),
    )

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    stencil[dom](inp1, inp2, inp3, inp4, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(inp1, out[0][0])
        assert np.allclose(inp2, out[0][1])
        assert np.allclose(inp3, out[1][0])
        assert np.allclose(inp4, out[1][1])


@pytest.mark.parametrize(
    "stencil",
    [tuple_output1, tuple_output2],
)
def test_field_of_tuple_output(backend, stencil):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out_np = np.zeros(shape, dtype="f8, f8")
    out = np_as_located_field(IDim, JDim, KDim)(out_np)

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    stencil[dom](inp1, inp2, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(inp1, out_np[:]["f0"])
        assert np.allclose(inp2, out_np[:]["f1"])


@fundef
def tuple_input(inp):
    inp_deref = deref(inp)
    return tuple_get(0, inp_deref) + tuple_get(1, inp_deref)


def test_tuple_field_input(backend):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()
    inp1 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )
    inp2 = np_as_located_field(IDim, JDim, KDim)(
        rng.normal(size=(shape[0], shape[1], shape[2])),
    )

    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    tuple_input[dom]((inp1, inp2), out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(np.asarray(inp1) + np.asarray(inp2), out)


def test_field_of_tuple_input(backend):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp = np.zeros(shape, dtype="f8, f8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                inp[i, j, k] = (inp1[i, j, k], inp2[i, j, k])

    inp = np_as_located_field(IDim, JDim, KDim)(inp)
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    tuple_input[dom](inp, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(np.asarray(inp1) + np.asarray(inp2), out)


@fundef
def tuple_tuple_input(inp):
    inp_deref = deref(inp)
    return (
        tuple_get(0, tuple_get(0, inp_deref))
        + tuple_get(1, tuple_get(0, inp_deref))
        + tuple_get(0, tuple_get(1, inp_deref))
        + tuple_get(1, tuple_get(1, inp_deref))
    )


def test_tuple_of_field_of_tuple_input(backend):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp1 = rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp2 = rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
    inp = np.zeros(shape, dtype="f8, f8")
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                inp[i, j, k] = (inp1[i, j, k], inp2[i, j, k])

    inp = np_as_located_field(IDim, JDim, KDim)(inp)
    out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    tuple_tuple_input[dom]((inp, inp), out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(2.0 * (np.asarray(inp1) + np.asarray(inp2)), out)


# TODO tuple of tuple currently not supported, needs clean redesign of iterator of tuple
# def test_tuple_of_tuple_of_field_input(backend):
#     backend, validate = backend

#     shape = [5, 7, 9]
#     rng = np.random.default_rng()

#     inp1 = np_as_located_field(IDim, JDim, KDim)(
#         rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
#     )
#     inp2 = np_as_located_field(IDim, JDim, KDim)(
#         rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
#     )
#     inp3 = np_as_located_field(IDim, JDim, KDim)(
#         rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
#     )
#     inp4 = np_as_located_field(IDim, JDim, KDim)(
#         rng.normal(rng.normal(size=(shape[0], shape[1], shape[2])))
#     )

#     out = np_as_located_field(IDim, JDim, KDim)(np.zeros(shape))

#     dom = {
#         IDim: range(0, shape[0]),
#         JDim: range(0, shape[1]),
#         KDim: range(0, shape[2]),
#     }
#     tuple_tuple_input[dom](
#         ((inp1, inp2), (inp3, inp4)), out=out, offset_provider={}, backend=backend
#     )
#     if validate:
#         assert np.allclose(
#             (np.asarray(inp1) + np.asarray(inp2) + np.asarray(inp3) + np.asarray(inp4)), out
#         )
