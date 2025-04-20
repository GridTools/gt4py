# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from gt4py.next.iterator.ir_utils import misc, ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import inline_lambdas

from gt4py.eve.pattern_matching import ObjectPattern as P


@pytest.mark.parametrize(
    "expr, expected_projector_matcher, expected_expr",
    [
        (
            im.let("a", "b")(im.make_tuple(im.tuple_get(0, "a"), im.tuple_get(42, "a"))),
            P(
                itir.Lambda,
                expr=P(
                    itir.FunCall,
                    fun=im.ref("make_tuple"),
                    args=[
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(0), P(itir.SymRef)],
                        ),
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(42), P(itir.SymRef)],
                        ),
                    ],
                ),
            ),
            im.ref("b"),
        ),
        (
            im.tuple_get(1, im.ref("a")),
            P(
                itir.Lambda,
                expr=P(
                    itir.FunCall,
                    fun=im.ref("tuple_get"),
                    args=[im.literal_from_value(1), P(itir.SymRef)],
                ),
            ),
            im.ref("a"),
        ),
        (
            im.make_tuple(im.tuple_get(0, im.ref("a")), im.tuple_get(42, im.ref("a"))),
            P(
                itir.Lambda,
                expr=P(
                    itir.FunCall,
                    fun=im.ref("make_tuple"),
                    args=[
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(0), P(itir.SymRef)],
                        ),
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(42), P(itir.SymRef)],
                        ),
                    ],
                ),
            ),
            im.ref("a"),
        ),
        (
            im.make_tuple(
                im.tuple_get(1, im.ref("a")),
                im.make_tuple(im.tuple_get(2, im.ref("a")), im.tuple_get(0, im.ref("a"))),
            ),
            P(
                itir.Lambda,
                expr=P(
                    itir.FunCall,
                    fun=im.ref("make_tuple"),
                    args=[
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(1), P(itir.SymRef)],
                        ),
                        P(
                            itir.FunCall,
                            fun=im.ref("make_tuple"),
                            args=[
                                P(
                                    itir.FunCall,
                                    fun=im.ref("tuple_get"),
                                    args=[im.literal_from_value(2), P(itir.SymRef)],
                                ),
                                P(
                                    itir.FunCall,
                                    fun=im.ref("tuple_get"),
                                    args=[im.literal_from_value(0), P(itir.SymRef)],
                                ),
                            ],
                        ),
                    ],
                ),
            ),
            im.ref("a"),
        ),
        (
            im.tuple_get(3, im.tuple_get(0, im.ref("a"))),
            P(
                itir.Lambda,
                expr=P(
                    itir.FunCall,
                    fun=im.ref("tuple_get"),
                    args=[
                        im.literal_from_value(3),
                        P(
                            itir.FunCall,
                            fun=im.ref("tuple_get"),
                            args=[im.literal_from_value(0), P(itir.SymRef)],
                        ),
                    ],
                ),
            ),
            im.ref("a"),
        ),
        (
            im.plus(im.ref("a"), im.ref("b")),
            None,
            im.plus(im.ref("a"), im.ref("b")),
        ),
        (
            im.call("as_fieldop")(im.ref("a")),
            None,
            im.call("as_fieldop")(im.ref("a")),
        ),
    ],
)
def test_extract_projector(expr, expected_projector_matcher, expected_expr):
    actual_projector, actual_expr = misc.extract_projector(expr)
    print(actual_projector)
    assert actual_expr == expected_expr

    if expected_projector_matcher is None:
        assert actual_projector is None
    else:
        assert expected_projector_matcher.match(actual_projector, raise_exception=True)
