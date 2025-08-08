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
cartesian_domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 5)})


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
    # inout ← (⇑deref)(inout)
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
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
    # inout ← (⇑(λ(x) → ·⟪IOffₒ, 0ₒ⟫(x)))(inout)
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
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
    # inout ← (⇑(λ(x) → ·⟪E2Vₒ, 0ₒ⟫(x)))(inout)
    offset_provider = {"E2V": e2v_conn}  # override
    ir = program_factory(
        params=[im.sym("inout", i_field_type)],
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
    # inout ← (⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout)
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
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
    # inout ← (⇑(λ(x, y) → ·⟪IOffₒ, 0ₒ⟫(x) + ·⟪IOffₒ, 1ₒ⟫(y)))(inout, in);
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
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


def test_check_inout_in_as_fieldop_arg():
    # inout ← (⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))((⇑deref)(inout))
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
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

    with pytest.raises(
        ValueError,
        match=r"Unexpected as_fieldop argument \(⇑deref\)\(inout\). Expected `make_tuple`, `tuple_get` or `SymRef`. Please run temporary extraction first.",
    ):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_two_fields():
    # inout ← (⇑(λ(x, y) → ·⟪IOffₒ, 1ₒ⟫(x) + ·⟪IOffₒ, 0ₒ⟫(y)))((⇑deref)(inout), in)
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x")(
                        im.plus(
                            im.deref(im.shift("IOff", 1)("x")), im.deref(im.shift("IOff", 0)("x"))
                        )
                    )
                )(im.make_tuple(im.ref("inout"), im.ref("in"))),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_arg_tuple():
    # inout ← (⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x) + ·⟪IOffₒ, 0ₒ⟫(x)))({inout, in})
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x")(
                        im.plus(
                            im.deref(im.shift("IOff", 1)("x")), im.deref(im.shift("IOff", 0)("x"))
                        )
                    )
                )(im.make_tuple(im.ref("inout"), im.ref("in"))),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_in_make_tuple_as_fieldop_in_arg():
    # inout ← (⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x) + ·⟪IOffₒ, 0ₒ⟫(x)))({(⇑deref)(inout), in})
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x")(
                        im.plus(
                            im.deref(im.shift("IOff", 1)("x")), im.deref(im.shift("IOff", 0)("x"))
                        )
                    )
                )(im.make_tuple(im.as_fieldop(im.ref("deref"))(im.ref("inout")), im.ref("in"))),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"Unexpected as_fieldop argument \{\(⇑deref\)\(inout\), in\}. Expected `make_tuple`, `tuple_get` or `SymRef`. Please run temporary extraction first.",
    ):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple():
    # {inout, inout2} ← {(⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout[0]),  (⇑deref)(inout2)}
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("inout2", i_field_type)],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.ref("inout")
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("inout2")),
                ),
                domain=cartesian_domain,
                target=im.make_tuple(im.ref("inout"), im.ref("inout2")),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target {inout, inout2} is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_as_fieldop():
    # {inout, out} ← (⇑(λ(x, y) → {·⟪IOffₒ, 1ₒ⟫(x), ·y}))(inout, in)
    ir = program_factory(
        params=[
            im.sym("inout", i_field_type),
            im.sym("in", i_field_type),
            im.sym("out", i_field_type),
        ],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(
                    im.lambda_("x", "y")(
                        im.make_tuple(im.deref(im.shift("IOff", 1)("x")), im.deref("y"))
                    )
                )(im.ref("inout"), im.ref("in")),
                domain=cartesian_domain,
                target=im.make_tuple(im.ref("inout"), im.ref("out")),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target {inout, out} is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_get_make_tuple():
    # inout ← {(⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout[0]), as_fieldop(...)}[0]
    ir = program_factory(
        params=[im.sym("inout", i_field_type), im.sym("in", i_field_type)],
        body=[
            itir.SetAt(
                expr=im.tuple_get(
                    0,
                    im.make_tuple(
                        im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                            im.ref("inout")
                        ),
                        im.as_fieldop(im.ref("deref"))(im.ref("in")),
                    ),
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_get():
    # inout ← {(⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout[0]), (⇑deref)(in)}
    ir = program_factory(
        params=[
            im.sym("inout", ts.TupleType(types=[i_field_type] * 2)),
            im.sym("in", i_field_type),
        ],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.tuple_get(0, im.ref("inout"))
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("in")),
                ),
                domain=cartesian_domain,
                target=im.ref("inout"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_tuple_get():
    # {inout[0], inout2} ← {(⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout[0]),  (⇑deref)(inout2)}
    ir = program_factory(
        params=[
            im.sym("inout", ts.TupleType(types=[i_field_type] * 2)),
            im.sym("inout2", i_field_type),
        ],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.tuple_get(0, im.ref("inout"))
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("inout2")),
                ),
                domain=cartesian_domain,
                target=im.make_tuple(im.tuple_get(0, im.ref("inout")), im.ref("inout2")),
            ),
        ],
    )

    with pytest.raises(
        ValueError, match="The target {inout\[0\], inout2} is also read with an offset."
    ):
        CheckInOutField.apply(ir, offset_provider=offset_provider)


def test_check_inout_tuple_get_tuple():
    # inout[0] ← {(⇑(λ(x) → ·⟪IOffₒ, 1ₒ⟫(x)))(inout[0][0]), (⇑deref)(in)}
    ir = program_factory(
        params=[
            im.sym("inout", ts.TupleType(types=[ts.TupleType(types=[i_field_type] * 2)] * 2)),
            im.sym("in", i_field_type),
        ],
        body=[
            itir.SetAt(
                expr=im.make_tuple(
                    im.as_fieldop(im.lambda_("x")(im.deref(im.shift("IOff", 1)("x"))))(
                        im.tuple_get(0, im.tuple_get(0, im.ref("inout")))
                    ),
                    im.as_fieldop(im.ref("deref"))(im.ref("in")),
                ),
                domain=cartesian_domain,
                target=im.tuple_get(0, im.ref("inout")),
            ),
        ],
    )

    with pytest.raises(ValueError, match="The target inout\[0\] is also read with an offset."):
        CheckInOutField.apply(ir, offset_provider=offset_provider)
