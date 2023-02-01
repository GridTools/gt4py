# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass

import numpy as np
import pytest

from gt4py.next.common import Dimension
from gt4py.next.iterator.builtins import deref
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.iterator.runtime import CartesianDomain, UnstructuredDomain, _deduce_domain, fundef

from .conftest import DummyConnectivity


@fundef
def foo(inp):
    return deref(inp)


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
