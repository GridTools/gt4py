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

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.infer_domain import InferDomain
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, AUTO_DOMAIN
import pytest
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next import common
from gt4py.next.type_system import type_specifications as ts

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)


@pytest.fixture
def offset_provider():
    offset_provider = {
        "Ioff": Dimension("IDim", DimensionKind.HORIZONTAL),
        "Joff": Dimension("JDim", DimensionKind.HORIZONTAL),
        "Koff": Dimension("KDim", DimensionKind.VERTICAL),
    }

    return offset_provider


def test_forward_difference_x(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    testee = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field"))

    domain = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
    )

    expected = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("in_field"))
    expected_domains = {
        "in_field": SymbolicDomain.from_expr(
            im.cartesian_domain(
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)}
            )
        )
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    assert actual_call == expected
    assert actual_domains == expected_domains


def test_laplace(offset_provider):
    stencil = im.lambda_("arg0")(
        im.plus(
            im.plus(
                im.plus(
                    im.plus(
                        im.multiplies_(-4.0, im.deref("arg0")),
                        im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
                    ),
                    im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg0")),
                ),
                im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg0")),
            ),
            im.deref(im.shift(itir.SymbolRef("Joff"), -1)("arg0")),
        )
    )

    testee = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field"))

    domain = im.cartesian_domain({"IDim": (0, 11), "JDim": (0, 7)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("in_field"))
    expected_domains = {
        "in_field": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (-1, 12), "JDim": (-1, 8)})
        )
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    assert actual_call == expected
    assert actual_domains == expected_domains


def test_shift_x_y_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg1")),
        )
    )
    testee = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field1"), im.ref("in_field2"))

    domain = im.cartesian_domain({"IDim": (0, 11), "JDim": (0, 7)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (-1, 10), "JDim": (0, 7)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 11), "JDim": (1, 8)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    assert actual_call == expected
    assert actual_domains == expected_domains


def test_shift_x_y_z_three_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1", "arg2")(
        im.plus(
            im.plus(
                im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
                im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg1")),
            ),
            im.deref(im.shift(itir.SymbolRef("Koff"), -1)("arg2")),
        )
    )
    testee = im.call(im.call("as_fieldop")(stencil))(
        im.ref("in_field1"), im.ref("in_field2"), im.ref("in_field3")
    )

    domain = im.cartesian_domain(
        {
            "IDim": (0, 11),
            "JDim": (0, 7),
            common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
        }
    )

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"), im.ref("in_field2"), im.ref("in_field3")
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain(
                {
                    "IDim": (1, 12),
                    "JDim": (0, 7),
                    common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
                }
            )
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain(
                {
                    "IDim": (0, 11),
                    "JDim": (1, 8),
                    common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
                }
            )
        ),
        "in_field3": SymbolicDomain.from_expr(
            im.cartesian_domain(
                {
                    "IDim": (0, 11),
                    "JDim": (0, 7),
                    common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (-1, 2),
                }
            )
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    assert actual_call == expected
    assert actual_domains == expected_domains


def test_nested_stencils(offset_provider):
    inner_stencil = im.lambda_("arg0_tmp", "arg1_tmp")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0_tmp")),
            im.deref(im.shift(itir.SymbolRef("Joff"), -1)("arg1_tmp")),
        )
    )
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Joff"), -1)("arg1")),
        )
    )

    tmp = im.call(im.call("as_fieldop")(inner_stencil))(im.ref("in_field1"), im.ref("in_field2"))

    testee = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field1"), tmp)

    domain_inner = im.cartesian_domain({"IDim": (0, 11), "JDim": (-1, 6)})
    domain = im.cartesian_domain({"IDim": (0, 11), "JDim": (0, 7)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"),
        im.call(im.call("as_fieldop")(inner_stencil, domain_inner))(
            im.ref("in_field1"), im.ref("in_field2")
        ),
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (1, 12), "JDim": (-1, 7)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 11), "JDim": (-2, 5)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    assert actual_call == expected
    assert actual_domains == expected_domains


@pytest.mark.parametrize("iterations", [3, 5])
def test_nested_stencils_n_times(offset_provider, iterations):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg1")),
        )
    )
    assert iterations >= 2

    current_tmp = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field1"), im.ref("in_field2"))
    current_domain = im.cartesian_domain(
        {"IDim": (0, 11), "JDim": (iterations - 1, 7 + iterations - 1)}
    )
    current_expected = im.call(im.call("as_fieldop")(stencil, current_domain))(
        im.ref("in_field1"), im.ref("in_field2")
    )

    for n in range(1, iterations):
        previous_tmp = current_tmp
        previous_expected = current_expected

        current_tmp = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field1"), previous_tmp)
        current_domain = im.cartesian_domain(
            {"IDim": (0, 11), "JDim": (iterations - 1 - n, 7 + iterations - 1 - n)}
        )
        current_expected = im.call(im.call("as_fieldop")(stencil, current_domain))(
            im.ref("in_field1"), previous_expected
        )

    testee = current_tmp

    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (1, 12), "JDim": (0, 7 + iterations - 1)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 11), "JDim": (iterations, 7 + iterations)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, SymbolicDomain.from_expr(current_domain), offset_provider
    )

    assert actual_call == current_expected
    assert actual_domains == expected_domains


def test_program(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    as_fieldop_tmp = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field"))
    as_fieldop = im.call(im.call("as_fieldop")(stencil))(im.ref("tmp"))

    domain_tmp = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)}
    )
    domain = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
    )

    params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]
    for param in params[:2]:
        param.dtype = ("float64", False)
        param.kind = "Iterator"

    testee = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=AUTO_DOMAIN, dtype=float_type)],
        body=[
            itir.SetAt(expr=as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
            itir.SetAt(expr=as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    expected_as_fieldop_tmp = im.call(im.call("as_fieldop")(stencil, domain_tmp))(
        im.ref("in_field")
    )
    expected_as_fieldop = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("tmp"))

    expected = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_tmp, dtype=float_type)],
        body=[
            itir.SetAt(expr=expected_as_fieldop_tmp, domain=domain_tmp, target=im.ref("tmp")),
            itir.SetAt(expr=expected_as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    actual_program = InferDomain.infer_program(testee, offset_provider)

    assert actual_program == expected


def test_program_two_tmps(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    as_fieldop_tmp1 = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field"))
    as_fieldop_tmp2 = im.call(im.call("as_fieldop")(stencil))(im.ref("tmp1"))
    as_fieldop = im.call(im.call("as_fieldop")(stencil))(im.ref("tmp2"))

    domain = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
    )
    domain_tmp1 = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 13)}
    )
    domain_tmp2 = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)}
    )

    params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]
    for param in params[:2]:
        param.dtype = ("float64", False)
        param.kind = "Iterator"

    testee = itir.Program(
        id="forward_diff_with_two_tmps",
        function_definitions=[],
        params=params,
        declarations=[
            itir.Temporary(id="tmp1", domain=AUTO_DOMAIN, dtype=float_type),
            itir.Temporary(id="tmp2", domain=AUTO_DOMAIN, dtype=float_type),
        ],
        body=[
            itir.SetAt(expr=as_fieldop_tmp1, domain=AUTO_DOMAIN, target=im.ref("tmp1")),
            itir.SetAt(expr=as_fieldop_tmp2, domain=AUTO_DOMAIN, target=im.ref("tmp2")),
            itir.SetAt(expr=as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    expected_as_fieldop_tmp1 = im.call(im.call("as_fieldop")(stencil, domain_tmp1))(
        im.ref("in_field")
    )
    expected_as_fieldop_tmp2 = im.call(im.call("as_fieldop")(stencil, domain_tmp2))(im.ref("tmp1"))
    expected_as_fieldop = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("tmp2"))

    expected = itir.Program(
        id="forward_diff_with_two_tmps",
        function_definitions=[],
        params=params,
        declarations=[
            itir.Temporary(id="tmp1", domain=domain_tmp1, dtype=float_type),
            itir.Temporary(id="tmp2", domain=domain_tmp2, dtype=float_type),
        ],
        body=[
            itir.SetAt(expr=expected_as_fieldop_tmp1, domain=domain_tmp1, target=im.ref("tmp1")),
            itir.SetAt(expr=expected_as_fieldop_tmp2, domain=domain_tmp2, target=im.ref("tmp2")),
            itir.SetAt(expr=expected_as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    actual_program = InferDomain.infer_program(testee, offset_provider)

    assert actual_program == expected


@pytest.mark.xfail(raises=ValueError)
def test_program_ValueError(offset_provider):
    with pytest.raises(ValueError, match=r"Temporaries can only be used once within a program."):
        stencil = im.lambda_("arg0")(
            im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
        )

        as_fieldop_tmp = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field"))
        as_fieldop = im.call(im.call("as_fieldop")(stencil))(im.ref("tmp"))

        domain = im.cartesian_domain(
            {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
        )

        params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]
        for param in params[:2]:
            param.dtype = ("float64", False)
            param.kind = "Iterator"

        testee = InferDomain.infer_program(
            itir.Program(
                id="forward_diff_with_tmp",
                function_definitions=[],
                params=params,
                declarations=[itir.Temporary(id="tmp", domain=AUTO_DOMAIN, dtype=float_type)],
                body=[
                    itir.SetAt(expr=as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
                    itir.SetAt(expr=as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
                    itir.SetAt(expr=as_fieldop, domain=domain, target=im.ref("out_field")),
                ],
            ),
            offset_provider,
        )


def test_program_tree_tmps_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg1"))
    )
    stencil_tmp = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    stencil_tmp_minus = im.lambda_("arg0", "arg1")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg0")), im.deref("arg1"))
    )

    as_fieldop_tmp1 = im.call(im.call("as_fieldop")(stencil))(
        im.ref("in_field1"), im.ref("in_field2")
    )
    as_fieldop_tmp2 = im.call(im.call("as_fieldop")(stencil_tmp))(im.ref("tmp1"))
    as_fieldop_out1 = im.call(im.call("as_fieldop")(stencil_tmp))(im.ref("tmp2"))
    as_fieldop_tmp3 = im.call(im.call("as_fieldop")(stencil))(im.ref("tmp1"), im.ref("in_field2"))
    as_fieldop_out2 = im.call(im.call("as_fieldop")(stencil_tmp_minus))(
        im.ref("tmp2"), im.ref("tmp3")
    )

    domain_out = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
    )
    domain_tmp3 = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)}
    )
    domain_tmp2 = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 12)}
    )
    domain_tmp1 = im.cartesian_domain(
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 13)}
    )
    params = [
        im.sym(name)
        for name in ["in_field1", "in_field2", "out_field1", "out_field2", "_gtmp_auto_domain"]
    ]
    for param in params[:3]:
        param.dtype = ("float64", False)
        param.kind = "Iterator"

    testee = itir.Program(
        id="differences_three_tmps_two_inputs",
        function_definitions=[],
        params=params,
        declarations=[
            itir.Temporary(id="tmp1", domain=AUTO_DOMAIN, dtype=float_type),
            itir.Temporary(id="tmp2", domain=AUTO_DOMAIN, dtype=float_type),
            itir.Temporary(id="tmp3", domain=AUTO_DOMAIN, dtype=float_type),
        ],
        body=[
            itir.SetAt(expr=as_fieldop_tmp1, domain=AUTO_DOMAIN, target=im.ref("tmp1")),
            itir.SetAt(expr=as_fieldop_tmp2, domain=AUTO_DOMAIN, target=im.ref("tmp2")),
            itir.SetAt(expr=as_fieldop_out1, domain=domain_out, target=im.ref("out_field1")),
            itir.SetAt(expr=as_fieldop_tmp3, domain=AUTO_DOMAIN, target=im.ref("tmp3")),
            itir.SetAt(expr=as_fieldop_out2, domain=domain_out, target=im.ref("out_field2")),
        ],
    )

    expected_as_fieldop_tmp1 = im.call(im.call("as_fieldop")(stencil, domain_tmp1))(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_as_fieldop_tmp2 = im.call(im.call("as_fieldop")(stencil_tmp, domain_tmp2))(
        im.ref("tmp1")
    )
    expected_as_fieldop_out1 = im.call(im.call("as_fieldop")(stencil_tmp, domain_out))(
        im.ref("tmp2")
    )
    expected_as_fieldop_tmp3 = im.call(im.call("as_fieldop")(stencil, domain_tmp3))(
        im.ref("tmp1"), im.ref("in_field2")
    )
    expected_as_fieldop_out2 = im.call(im.call("as_fieldop")(stencil_tmp_minus, domain_out))(
        im.ref("tmp2"), im.ref("tmp3")
    )

    expected = itir.Program(
        id="differences_three_tmps_two_inputs",
        function_definitions=[],
        params=params,
        declarations=[
            itir.Temporary(id="tmp1", domain=domain_tmp1, dtype=float_type),
            itir.Temporary(id="tmp2", domain=domain_tmp2, dtype=float_type),
            itir.Temporary(id="tmp3", domain=domain_tmp3, dtype=float_type),
        ],
        body=[
            itir.SetAt(expr=expected_as_fieldop_tmp1, domain=domain_tmp1, target=im.ref("tmp1")),
            itir.SetAt(expr=expected_as_fieldop_tmp2, domain=domain_tmp2, target=im.ref("tmp2")),
            itir.SetAt(
                expr=expected_as_fieldop_out1, domain=domain_out, target=im.ref("out_field1")
            ),
            itir.SetAt(expr=expected_as_fieldop_tmp3, domain=domain_tmp3, target=im.ref("tmp3")),
            itir.SetAt(
                expr=expected_as_fieldop_out2, domain=domain_out, target=im.ref("out_field2")
            ),
        ],
    )

    actual_program = InferDomain.infer_program(testee, offset_provider)

    assert actual_program == expected
