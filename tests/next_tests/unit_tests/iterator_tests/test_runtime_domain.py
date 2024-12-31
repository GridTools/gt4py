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
from gt4py.next import common
from gt4py.next.iterator.builtins import deref
from gt4py.next.iterator.runtime import CartesianDomain, UnstructuredDomain, _deduce_domain, fundef


@fundef
def foo(inp):
    return deref(inp)


connectivity = common.ConnectivityType(
    domain=[gtx.Dimension("dummy_origin"), gtx.Dimension("dummy_neighbor")],
    codomain=gtx.Dimension("dummy_codomain"),
    skip_value=common._DEFAULT_SKIP_VALUE,
    dtype=None,
)

I = gtx.Dimension("I")


def test_deduce_domain():
    assert isinstance(_deduce_domain({}, {}), CartesianDomain)
    assert isinstance(_deduce_domain(UnstructuredDomain(), {}), UnstructuredDomain)
    assert isinstance(_deduce_domain({}, {"foo": connectivity}), UnstructuredDomain)
    assert isinstance(
        _deduce_domain(CartesianDomain([(I, range(1))]), {"foo": connectivity}), CartesianDomain
    )


def test_embedded_error_on_wrong_domain():
    dom = CartesianDomain([(I, range(1))])

    out = gtx.as_field([I], np.zeros(1))
    with pytest.raises(RuntimeError, match="expected 'UnstructuredDomain'"):
        foo[dom](gtx.as_field([I], np.zeros((1,))), out=out, offset_provider={"bar": connectivity})
