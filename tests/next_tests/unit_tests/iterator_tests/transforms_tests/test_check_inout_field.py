# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pytest
from next_tests.toy_connectivity import e2v_conn

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.check_inout_field import CheckInOutField
from gt4py.next.type_system import type_specifications as ts

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
offset_provider = {"IOff": IDim}
i_field_type = ts.FieldType(dims=[IDim], dtype=float_type)
cartesian_domain = im.call("cartesian_domain")(
    im.call("named_range")(itir.AxisLiteral(value="IDim"), 0, 5),
    itir.AxisLiteral(value="JDim"),
    0,
    7,
)


def program_factory(
    params: list[itir.Sym],
    body: list[itir.SetAt],
    declarations: Optional[list[itir.Temporary]] = None,
) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=params,
        declarations=declarations or [],
        body=body,
    )


def test_check_inout_no_offset():
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.ref("deref"))(im.ref("inout")),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    # Should not raise
    assert ir == CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_zero_offset():
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 0)("x"))))(
                    im.ref("inout")
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    # Should not raise
    assert ir == CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_e2v_zero_offset():
    offset_provider = {"E2V": e2v_conn}  # override
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("E2V", 0)("x"))))(
                    im.ref("inout")
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_offset():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                    im.ref("inout")
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_shift_different_field():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 0)("x")), im.deref(im.shift("IOff", 1)("y"))
                        )
                    )
                )(im.ref("inout"), im.ref("in")),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    assert ir == CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                    im.as_fieldop(im.ref("deref"))(im.ref("inout"))
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_two_fields():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 1)("x")), im.deref(im.shift("IOff", 0)("y"))
                        )
                    )
                )(im.as_fieldop(im.ref("deref"))(im.ref("inout")), im.ref("in")),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_shift_different_field():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 0)("x")), im.deref(im.shift("IOff", 1)("y"))
                        )
                    )
                )(im.as_fieldop(im.ref("deref"))(im.ref("inout")), im.ref("in")),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    assert ir == CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_shifted():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 0)("x")), im.deref(im.shift("IOff", 0)("y"))
                        )
                    )
                )(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.ref("inout")
                    ),
                    im.ref("in"),
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_nested_shifted():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 0)("x")), im.deref(im.shift("IOff", 0)("y"))
                        )
                    )
                )(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 0)("x"))))(
                        im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                            im.ref("inout")
                        )
                    ),
                    im.ref("in"),
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_nested_shift_different_arg():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.plus(
                            im.deref(im.shift("IOff", 0)("x")), im.deref(im.shift("IOff", 0)("y"))
                        )
                    )
                )(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 0)("x"))))(
                        im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                            im.ref("in")
                        )
                    ),
                    im.ref("inout"),
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    assert ir == CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple():
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.ref("inout")
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("in")),
                ),
                domain=cartesian_domain,
                target=im.make_tuple(im.ref("inout"), im.ref("in")),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target {inout, in} is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_get():
    ir = program_factory(
        params=[
            im.sym("inout", ts.TupleType(types=[i_field_type] * 2)),
            im.sym("in", i_field_type),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.tuple_get(0, im.ref("inout"))
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("in")),
                ),
                domain=cartesian_domain,
                target=im.make_tuple(im.ref("inout")),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target {inout} is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)
