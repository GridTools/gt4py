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

from typing import Union
from gt4py.next.common import Connectivity, Dimension
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
FTYPE = ts.FieldType(dims=[DIM], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
FSYMBOLS = dict(
    __w_size_0=N,
    __w_stride_0=1,
    __x_size_0=N,
    __x_stride_0=1,
    __y_size_0=N,
    __y_stride_0=1,
    __z_size_0=N,
    __z_stride_0=1,
    size=N,
)
OFFSET_PROVIDERS: dict[str, Connectivity | Dimension] = {}


def test_gtir_sum2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, "size")
    )
    testee = itir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="z"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
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
        [FTYPE, FTYPE, FTYPE, ts.ScalarType(ts.ScalarKind.INT32)], OFFSET_PROVIDERS
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, y=b, z=c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, "size")
    )
    testee = itir.Program(
        id="sum_2fields",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="z"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )("x", "x"),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    sdfg_genenerator = FieldviewGtirToSDFG(
        [FTYPE, FTYPE, ts.ScalarType(ts.ScalarKind.INT32)], OFFSET_PROVIDERS
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    sdfg(x=a, z=b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, "size")
    )
    testee_fieldview = itir.Program(
        id="sum_3fields",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="y"),
            itir.Sym(id="w"),
            itir.Sym(id="z"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                        domain,
                    )
                )(
                    "x",
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                            domain,
                        )
                    )("y", "w"),
                ),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )
    testee_inlined = itir.Program(
        id="sum_3fields",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="y"),
            itir.Sym(id="w"),
            itir.Sym(id="z"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a", "b", "c")(
                            im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c")))
                        ),
                        domain,
                    )
                )("x", "y", "w"),
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
        [FTYPE, FTYPE, FTYPE, FTYPE, ts.ScalarType(ts.ScalarKind.INT32)], OFFSET_PROVIDERS
    )

    for testee in [testee_fieldview, testee_inlined]:
        sdfg = sdfg_genenerator.visit(testee)
        assert isinstance(sdfg, dace.SDFG)

        sdfg(x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


def test_gtir_select():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=DIM.value), 0, "size")
    )
    testee = itir.Program(
        id="select_2sums",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="y"),
            itir.Sym(id="w"),
            itir.Sym(id="z"),
            itir.Sym(id="cond"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("select")(
                        im.deref("cond"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                domain,
                            )
                        )("x", "y"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                domain,
                            )
                        )("y", "w"),
                    )
                )(),
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
        [
            FTYPE,
            FTYPE,
            FTYPE,
            FTYPE,
            ts.ScalarType(ts.ScalarKind.BOOL),
            ts.ScalarType(ts.ScalarKind.INT32),
        ],
        OFFSET_PROVIDERS,
    )
    sdfg = sdfg_genenerator.visit(testee)

    assert isinstance(sdfg, dace.SDFG)

    for s in [False, True]:
        sdfg(cond=s, x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b) if s else (b + c))
