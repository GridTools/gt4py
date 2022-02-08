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


# TODO test all cases
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
    inp = np.asarray(list(zip(inp1, inp2)))
    print(inp)
    # exit()
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
