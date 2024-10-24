# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator.builtins import deref
from gt4py.next.iterator.runtime import CartesianDomain, UnstructuredDomain, _deduce_domain, fundef

from next_tests.unit_tests.conftest import DummyConnectivity


@fundef
def foo(inp):
    return deref(inp)


connectivity = DummyConnectivity(max_neighbors=0, has_skip_values=True)


def test_deduce_domain():
    assert isinstance(_deduce_domain({}, {}), CartesianDomain)
    assert isinstance(_deduce_domain(UnstructuredDomain(), {}), UnstructuredDomain)
    assert isinstance(_deduce_domain({}, {"foo": connectivity}), UnstructuredDomain)
    assert isinstance(
        _deduce_domain(CartesianDomain([("I", range(1))]), {"foo": connectivity}), CartesianDomain
    )


I = gtx.Dimension("I")


def test_embedded_error_on_wrong_domain():
    dom = CartesianDomain([("I", range(1))])

    out = gtx.as_field([I], np.zeros(1))
    with pytest.raises(RuntimeError, match="expected 'UnstructuredDomain'"):
        foo[dom](gtx.as_field([I], np.zeros((1,))), out=out, offset_provider={"bar": connectivity})
