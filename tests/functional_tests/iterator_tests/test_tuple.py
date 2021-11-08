import numpy as np

from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis, closure, fendef, fundef


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")
KDim = CartesianAxis("KDim")

# semantics of stencil return that is called from the fencil (after `:` the structure of the output)
# `return a` -> (a,): [field]
# `return make_tuple(a)` -> (a,): [field]
# `return a,b` -> (a,b): [field, field]
# `return make_tuple(a,b)` -> (a,b): [field, field]
# `return make_tuple(a), make_tuple(b)` -> ((a,), (b,)): [(field,), (field,)]
# `return make_tuple(make_tuple(a,b))` -> ((a,b)): [(field,field)]


@fundef
def tuple_output(inp1, inp2):
    return make_tuple(make_tuple(deref(inp1), deref(inp2)))


def test_tuple_output():
    # backend, validate = backend
    backend = None
    validate = True

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

    dom = domain(
        named_range(IDim, 0, shape[0]),
        named_range(JDim, 0, shape[1]),
        named_range(KDim, 0, shape[2]),
    )
    tuple_output[dom](inp1, inp2, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(inp1, out[0])
        assert np.allclose(inp2, out[1])
