# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import common
import pytest
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im, domain_utils
from gt4py.next.iterator.transforms import concat_where_transforms, inline_lambdas, infer_domain, collapse_tuple
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.type_system import type_specifications as it_ts

int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
field_type = ts.FieldType(dims=[IDim], dtype=int_type)

def test_trivial():
    cond = im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 1)})
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 2)})
    symbolic_domain = domain_utils.SymbolicDomain.from_expr(domain)

    testee = im.concat_where(
        cond,
        im.make_tuple(im.ref("a", field_type), im.ref("c", field_type)),
        im.make_tuple(im.ref("b", field_type), im.ref("d", field_type))
    )
    testee, _ = infer_domain.infer_expr(
        testee,
        (symbolic_domain, symbolic_domain),
        keep_existing_domains=True,
        offset_provider={},
    )

    expected = im.make_tuple(
        im.concat_where(cond, "a", "b"),
        im.concat_where(cond, "c", "d")
    )

    actual = concat_where_transforms.expand_tuple(
        testee, offset_provider_type={}, allow_undeclared_symbols=True)

    actual = collapse_tuple.CollapseTuple.apply(actual, allow_undeclared_symbols=True, within_stencil=False)

    assert actual == expected