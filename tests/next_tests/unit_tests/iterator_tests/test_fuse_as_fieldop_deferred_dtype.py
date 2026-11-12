# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""`FuseAsFieldOp` must tolerate fieldop arguments with unconstrained (deferred) types.

Reproduces a failure reported from an icon4py tracer-advection stencil: a boolean
scalar passed into an `as_fieldop` lambda that consumes it only in a way that does
not constrain the type (e.g. forwarded through arithmetic/reduce constructs) keeps
type `DeferredType(constraint=None)` after (re-)inference. The inliner's argument
predicate used to call `extract_dtype` unconditionally and raised
`ValueError: Can not unambiguosly extract data type from 'DeferredType(constraint=None)'`
instead of simply keeping the argument un-inlined.
"""

import pytest

from gt4py import next as gtx
from gt4py.next import utils as gtx_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import fuse_as_fieldop
from gt4py.next.type_system import type_specifications as ts


def test_fuse_as_fieldop_with_deferred_arg_dtype():
    idim = gtx.Dimension("IDim")
    field_type = ts.FieldType(dims=[idim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    domain = im.domain("cartesian_domain", {idim: (0, 1)})

    # inner `as_fieldop` consumes an undeclared symbol, so its result type remains
    # `DeferredType(constraint=None)` after (re-)inference
    inner_as_fieldop = im.op_as_fieldop("plus", domain)(
        im.ref("inp1", field_type),
        im.ref("undeclared_scalar"),
    )
    nested_as_fieldop = im.op_as_fieldop("plus", domain)(
        inner_as_fieldop,
        im.ref("inp3", field_type),
    )

    # must not raise `ValueError: Can not unambiguosly extract data type from
    # 'DeferredType(constraint=None)'`; the deferred-typed argument is kept un-inlined
    result = fuse_as_fieldop.FuseAsFieldOp.apply(
        nested_as_fieldop,
        offset_provider_type={},
        allow_undeclared_symbols=True,
        uids=gtx_utils.IDGeneratorPool(),
    )

    assert isinstance(result, itir.Expr)
    # the deferred-typed argument was not inlined into the outer stencil
    assert "undeclared_scalar" in str(result)


@pytest.mark.parametrize("allow_undeclared_symbols", [True])
def test_fuse_as_fieldop_strict_mode_still_fuses_concrete_args(allow_undeclared_symbols: bool):
    idim = gtx.Dimension("IDim")
    field_type = ts.FieldType(dims=[idim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    domain = im.domain("cartesian_domain", {idim: (0, 1)})

    nested_as_fieldop = im.op_as_fieldop("plus", domain)(
        im.op_as_fieldop("multiplies", domain)(
            im.ref("inp1", field_type), im.ref("inp2", field_type)
        ),
        im.ref("inp3", field_type),
    )

    result = fuse_as_fieldop.FuseAsFieldOp.apply(
        nested_as_fieldop,
        offset_provider_type={},
        allow_undeclared_symbols=allow_undeclared_symbols,
        uids=gtx_utils.IDGeneratorPool(),
    )

    # regular fusion behavior is unaffected by the fix
    assert "undeclared" not in str(result)
    assert str(result).count("as_fieldop") == 1
