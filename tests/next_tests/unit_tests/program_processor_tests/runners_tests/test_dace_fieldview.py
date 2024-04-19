# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
"""
Test that ITIR can be lowered to SDFG.

Note: this test module covers the fieldview flavour of ITIR.
"""

from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace_fieldview.gtir_to_sdfg import (
    GtirToSDFG as FieldviewGtirToSDFG,
)
from gt4py.next.type_system import type_specifications as ts

import numpy as np

import pytest

dace = pytest.importorskip("dace")


N = 10
DIM = Dimension("D")
FTYPE = ts.FieldType(dims=[DIM], dtype=ts.ScalarKind.FLOAT64)


def test_gtir_sum2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, 10)
    )
    testee = itir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="z")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b")))
                    )
                )("x", "y"),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        param_types=([FTYPE] * 3),
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, y=b, z=c)
    assert np.allclose(c, (a + b))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, 10)
    )
    testee = itir.Program(
        id="sum_3fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="w"), itir.Sym(id="z")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b")))
                    )
                )(
                    "x",
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b")))
                        )
                    )("y", "w"),
                ),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)
    d = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        param_types=([FTYPE] * 4),
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, y=b, w=c, z=d)
    assert np.allclose(d, (a + b + c))
