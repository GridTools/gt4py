from dataclasses import dataclass

import numpy as np
import pytest

from functional.common import Dimension
from functional.iterator.builtins import deref
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianDomain, UnstructuredDomain, _deduce_domain, fundef


@fundef
def foo(inp):
    return deref(inp)


@dataclass
class DummyConnectivity:
    max_neighbors: int
    has_skip_values: int
    origin_axis: Dimension = Dimension("dummy")

    def mapped_index(_, __) -> int:
        return 0


connectivity = DummyConnectivity(max_neighbors=0, has_skip_values=True)


def test_deduce_domain():
    assert isinstance(_deduce_domain({}, {}), CartesianDomain)
    assert isinstance(_deduce_domain(UnstructuredDomain(), {}), UnstructuredDomain)
    assert isinstance(
        _deduce_domain({}, {"foo": connectivity}),
        UnstructuredDomain,
    )
    assert isinstance(
        _deduce_domain(CartesianDomain([("I", range(1))]), {"foo": connectivity}),
        CartesianDomain,
    )


I = Dimension("I")


def test_embedded_error_on_wrong_domain():
    dom = CartesianDomain([("I", range(1))])

    out = np_as_located_field(I)(
        np.zeros(
            1,
        )
    )
    with pytest.raises(RuntimeError, match="expected `UnstructuredDomain`"):
        foo[dom](
            np_as_located_field(I)(np.zeros((1,))), out=out, offset_provider={"bar": connectivity}
        )
