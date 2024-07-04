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
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain
import pytest
from gt4py.next.common import Dimension, DimensionKind


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

    domain = im.cartesian_domain({"IDim": (0, 0)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("in_field"))
    expected_domains = {"in_field": SymbolicDomain.from_expr(im.cartesian_domain({"IDim": (0, 1)}))}

    actual_call, actual_domains = InferDomain.infer_as_fieldop(testee, domain, offset_provider)

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

    domain = im.cartesian_domain({"IDim": (0, 0), "JDim": (0, 0)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(im.ref("in_field"))
    expected_domains = {
        "in_field": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (-1, 1), "JDim": (-1, 1)})
        )
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(testee, domain, offset_provider)

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

    domain = im.cartesian_domain({"IDim": (0, 0), "JDim": (0, 0)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (-1, -1), "JDim": (0, 0)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 0), "JDim": (1, 1)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(testee, domain, offset_provider)

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

    domain = im.cartesian_domain({"IDim": (0, 0), "JDim": (0, 0), "KDim": (0, 0)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"), im.ref("in_field2"), im.ref("in_field3")
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (1, 1), "JDim": (0, 0), "KDim": (0, 0)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 0), "JDim": (1, 1), "KDim": (0, 0)})
        ),
        "in_field3": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 0), "JDim": (0, 0), "KDim": (-1, -1)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(testee, domain, offset_provider)

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

    domain_inner = im.cartesian_domain({"IDim": (0, 0), "JDim": (-1, -1)})
    domain = im.cartesian_domain({"IDim": (0, 0), "JDim": (0, 0)})

    expected = im.call(im.call("as_fieldop")(stencil, domain))(
        im.ref("in_field1"),
        im.call(im.call("as_fieldop")(inner_stencil, domain_inner))(
            im.ref("in_field1"), im.ref("in_field2")
        ),
    )
    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (1, 1), "JDim": (-1, 0)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 0), "JDim": (-2, -2)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(testee, domain, offset_provider)

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
    current_domain = im.cartesian_domain({"IDim": (0, 0), "JDim": (iterations - 1, iterations - 1)})
    current_expected = im.call(im.call("as_fieldop")(stencil, current_domain))(
        im.ref("in_field1"), im.ref("in_field2")
    )

    for n in range(1, iterations):
        previous_tmp = current_tmp
        previous_expected = current_expected

        current_tmp = im.call(im.call("as_fieldop")(stencil))(im.ref("in_field1"), previous_tmp)
        current_domain = im.cartesian_domain(
            {"IDim": (0, 0), "JDim": (iterations - 1 - n, iterations - 1 - n)}
        )
        current_expected = im.call(im.call("as_fieldop")(stencil, current_domain))(
            im.ref("in_field1"), previous_expected
        )

    testee = current_tmp

    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (1, 1), "JDim": (0, iterations - 1)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.cartesian_domain({"IDim": (0, 0), "JDim": (iterations, iterations)})
        ),
    }

    actual_call, actual_domains = InferDomain.infer_as_fieldop(
        testee, current_domain, offset_provider
    )

    assert actual_call == current_expected
    assert actual_domains == expected_domains
