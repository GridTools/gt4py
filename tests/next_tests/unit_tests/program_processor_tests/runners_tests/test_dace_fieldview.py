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

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import IDim

import numpy as np
import pytest

dace = pytest.importorskip("dace")


N = 10
IFTYPE = ts.FieldType(dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)
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


def test_gtir_copy():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="y"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.deref("a")),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="y"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.empty_like(a)

    arg_types = [IFTYPE, IFTYPE, SIZE_TYPE]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, y=b, **FSYMBOLS)
    assert np.allclose(a, b)


def test_gtir_update():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="gtir_copy",
        function_definitions=[],
        params=[itir.Sym(id="x"), itir.Sym(id="size")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("a")(im.plus(im.deref("a"), 1.0)),
                        domain,
                    )
                )("x"),
                domain=domain,
                target=itir.SymRef(id="x"),
            )
        ],
    )

    a = np.random.rand(N)
    ref = a + 1.0

    arg_types = [IFTYPE, SIZE_TYPE]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, **FSYMBOLS)
    assert np.allclose(a, ref)


def test_gtir_sum2():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
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

    arg_types = [IFTYPE, IFTYPE, IFTYPE, SIZE_TYPE]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, y=b, z=c, **FSYMBOLS)
    assert np.allclose(c, (a + b))


def test_gtir_sum2_sym():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
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

    arg_types = [IFTYPE, IFTYPE, SIZE_TYPE]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    sdfg(x=a, z=b, **FSYMBOLS)
    assert np.allclose(b, (a + a))


def test_gtir_sum3():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    stencil1 = im.call(
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
    )
    stencil2 = im.call(
        im.call("as_fieldop")(
            im.lambda_("a", "b", "c")(
                im.plus(im.deref("a"), im.plus(im.deref("b"), im.deref("c")))
            ),
            domain,
        )
    )("x", "y", "w")

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    arg_types = [IFTYPE, IFTYPE, IFTYPE, IFTYPE, SIZE_TYPE]

    for i, stencil in enumerate([stencil1, stencil2]):
        testee = itir.Program(
            id=f"sum_3fields_{i}",
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
                    expr=stencil,
                    domain=domain,
                    target=itir.SymRef(id="z"),
                )
            ],
        )

        sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

        d = np.empty_like(a)

        sdfg(x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + c))


def test_gtir_select():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
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
            itir.Sym(id="scalar"),
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
                        im.call("select")(
                            im.deref("cond"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                    domain,
                                )
                            )("y", "scalar"),
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("a", "b")(im.plus(im.deref("a"), im.deref("b"))),
                                    domain,
                                )
                            )("w", "scalar"),
                        )
                    )(),
                ),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)
    b = np.random.rand(N)
    c = np.random.rand(N)

    arg_types = [
        IFTYPE,
        IFTYPE,
        IFTYPE,
        IFTYPE,
        ts.ScalarType(ts.ScalarKind.BOOL),
        ts.ScalarType(ts.ScalarKind.FLOAT64),
        SIZE_TYPE,
    ]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    for s in [False, True]:
        d = np.empty_like(a)
        sdfg(cond=np.bool_(s), scalar=1.0, x=a, y=b, w=c, z=d, **FSYMBOLS)
        assert np.allclose(d, (a + b + 1) if s else (a + c + 1))


def test_gtir_select_nested():
    domain = im.call("cartesian_domain")(
        im.call("named_range")(itir.AxisLiteral(value=IDim.value), 0, "size")
    )
    testee = itir.Program(
        id="select_nested",
        function_definitions=[],
        params=[
            itir.Sym(id="x"),
            itir.Sym(id="z"),
            itir.Sym(id="cond_1"),
            itir.Sym(id="cond_2"),
            itir.Sym(id="size"),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("select")(
                        im.deref("cond_1"),
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("a")(im.plus(im.deref("a"), 1)),
                                domain,
                            )
                        )("x"),
                        im.call(
                            im.call("select")(
                                im.deref("cond_2"),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a")(im.plus(im.deref("a"), 2)),
                                        domain,
                                    )
                                )("x"),
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("a")(im.plus(im.deref("a"), 3)),
                                        domain,
                                    )
                                )("x"),
                            )
                        )(),
                    )
                )(),
                domain=domain,
                target=itir.SymRef(id="z"),
            )
        ],
    )

    a = np.random.rand(N)

    arg_types = [
        IFTYPE,
        IFTYPE,
        ts.ScalarType(ts.ScalarKind.BOOL),
        ts.ScalarType(ts.ScalarKind.BOOL),
        SIZE_TYPE,
    ]
    sdfg = dace_backend.build_sdfg_from_gtir(testee, arg_types, {})

    for s1 in [False, True]:
        for s2 in [False, True]:
            b = np.empty_like(a)
            sdfg(cond_1=np.bool_(s1), cond_2=np.bool_(s2), x=a, z=b, **FSYMBOLS)
            assert np.allclose(b, (a + 1.0) if s1 else (a + 2.0) if s2 else (a + 3.0))
