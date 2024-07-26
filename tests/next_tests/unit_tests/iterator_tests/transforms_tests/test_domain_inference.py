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

# TODO(SF-N): test scan operator

import copy
import numpy as np
from typing import List
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.infer_domain import infer_as_fieldop, infer_program
from gt4py.next.iterator.transforms.global_tmps import SymbolicDomain, AUTO_DOMAIN
import pytest
from gt4py.eve.extended_typing import Dict
from gt4py.next.common import Dimension, DimensionKind
from gt4py.next import common, NeighborTableOffsetProvider
from gt4py.next.type_system import type_specifications as ts
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding

float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)


@pytest.fixture
def offset_provider():
    offset_provider = {
        "Ioff": Dimension("IDim", DimensionKind.HORIZONTAL),
        "Joff": Dimension("JDim", DimensionKind.HORIZONTAL),
        "Koff": Dimension("KDim", DimensionKind.VERTICAL),
    }

    return offset_provider


@pytest.fixture
def unstructured_offset_provider():
    offset_provider = {
        "E2V": NeighborTableOffsetProvider(
            np.array([[0, 1]], dtype=np.int32),
            Dimension("Edge", DimensionKind.HORIZONTAL),
            Dimension("Vertex", DimensionKind.HORIZONTAL),
            2,
        )
    }

    return offset_provider


def run_test_as_fieldop(
    stencil,
    domain,
    expected_domain_dict,
    offset_provider,
    *refs,
    domain_type=common.GridType.CARTESIAN,
):
    testee = im.as_fieldop(stencil)(*refs)
    expected = im.as_fieldop(stencil, domain)(*refs)

    actual_call, actual_domains = infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    folded_domains = domains_dict_constant_folding(actual_domains)
    expected_domains = {
        ref: SymbolicDomain.from_expr(im.domain(domain_type, d))
        for ref, d in expected_domain_dict.items()
    }

    assert actual_call == expected
    assert folded_domains == expected_domains


def run_test_program(testee, expected, offset_provider):
    actual_program = infer_program(testee, offset_provider)

    folded_program = program_constant_folding(actual_program)
    assert folded_program == expected


def create_params(names: List[str]) -> List[im.sym]:
    params = [im.sym(name) for name in names]
    for param in params[:2]:
        param.dtype = ("float64", False)
        param.kind = "Iterator"
    return params


def domains_dict_constant_folding(domains: Dict[str, SymbolicDomain]) -> Dict[str, SymbolicDomain]:
    return {k: domain_constant_folding(v) for k, v in domains.items()}


def domain_constant_folding(domain: SymbolicDomain | itir.FunCall) -> SymbolicDomain | itir.FunCall:
    new_domain = (
        SymbolicDomain.from_expr(copy.deepcopy(domain))
        if isinstance(domain, itir.FunCall)
        else copy.deepcopy(domain)
    )
    for dim in new_domain.ranges:
        range_dim = new_domain.ranges[dim]
        range_dim.start, range_dim.stop = (
            ConstantFolding.apply(range_dim.start),
            ConstantFolding.apply(range_dim.stop),
        )
    return SymbolicDomain.as_expr(new_domain) if isinstance(domain, itir.FunCall) else new_domain


def as_fieldop_domains_constant_folding(as_fieldop_call: itir.FunCall) -> itir.FunCall:
    def fold_fun_args(fun):
        if isinstance(fun, itir.FunCall) and cpm.is_call_to(fun, "cartesian_domain"):
            fun.args = [
                ConstantFolding.apply(dim) if isinstance(dim, itir.FunCall) else dim
                for dim in fun.args
            ]

    new_call = copy.deepcopy(as_fieldop_call)
    for fun in new_call.fun.args:
        fold_fun_args(fun)

    new_call.args = [
        as_fieldop_domains_constant_folding(arg) if isinstance(arg, itir.FunCall) else arg
        for arg in new_call.args
    ]

    return new_call


def program_constant_folding(program_call: itir.Program) -> itir.Program:
    new_call = copy.deepcopy(program_call)
    for j, tmp in enumerate(new_call.declarations):
        tmp.domain = domain_constant_folding(tmp.domain)
    for j, set_at in enumerate(new_call.body):
        set_at.expr = as_fieldop_domains_constant_folding(set_at.expr)
        set_at.domain = domain_constant_folding(set_at.domain)
    return new_call


def test_forward_difference_x(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11)})
    expected_domains_dict = {
        "in_field": {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)}
    }
    run_test_as_fieldop(stencil, domain, expected_domains_dict, offset_provider, im.ref("in_field"))


def test_unstructured_shift(unstructured_offset_provider):
    stencil = im.lambda_("arg0")(im.deref(im.shift(itir.SymbolRef("E2V"), 1)("arg0")))
    domain = im.domain(
        common.GridType.UNSTRUCTURED,
        {common.Dimension(value="Edge", kind=common.DimensionKind.HORIZONTAL): (0, 1)},
    )
    expected_domains_dict = {
        "in_field": {common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL): (0, 2)}
    }

    run_test_as_fieldop(
        stencil,
        domain,
        expected_domains_dict,
        unstructured_offset_provider,
        im.ref("in_field"),
        domain_type=common.GridType.UNSTRUCTURED,
    )


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
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})
    expected_domains_dict = {"in_field": {"IDim": (-1, 12), "JDim": (-1, 8)}}

    run_test_as_fieldop(stencil, domain, expected_domains_dict, offset_provider, im.ref("in_field"))


def test_shift_x_y_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})
    expected_domains_dict = {
        "in_field1": {"IDim": (-1, 10), "JDim": (0, 7)},
        "in_field2": {"IDim": (0, 11), "JDim": (1, 8)},
    }
    run_test_as_fieldop(
        stencil,
        domain,
        expected_domains_dict,
        offset_provider,
        im.ref("in_field1"),
        im.ref("in_field2"),
    )


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
    domain_dict = {
        "IDim": (0, 11),
        "JDim": (0, 7),
        common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
    }
    expected_domain_dict = {
        "in_field1": {
            "IDim": (1, 12),
            "JDim": (0, 7),
            common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
        },
        "in_field2": {
            "IDim": (0, 11),
            "JDim": (1, 8),
            common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (0, 3),
        },
        "in_field3": {
            "IDim": (0, 11),
            "JDim": (0, 7),
            common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL): (-1, 2),
        },
    }
    run_test_as_fieldop(
        stencil,
        im.domain(common.GridType.CARTESIAN, domain_dict),
        expected_domain_dict,
        offset_provider,
        im.ref("in_field1"),
        im.ref("in_field2"),
        im.ref("in_field3"),
    )


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
    tmp = im.as_fieldop(inner_stencil)(im.ref("in_field1"), im.ref("in_field2"))
    testee = im.as_fieldop(stencil)(im.ref("in_field1"), tmp)

    domain_inner = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (-1, 6)})
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})

    expected_inner = im.as_fieldop(inner_stencil, domain_inner)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), expected_inner)

    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.domain(common.GridType.CARTESIAN, {"IDim": (1, 12), "JDim": (-1, 7)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (-2, 5)})
        ),
    }
    actual_call, actual_domains = infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )
    folded_domains = domains_dict_constant_folding(actual_domains)
    folded_call = as_fieldop_domains_constant_folding(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


@pytest.mark.parametrize("iterations", [3, 5])
def test_nested_stencils_n_times(offset_provider, iterations):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Joff"), 1)("arg1")),
        )
    )
    assert iterations >= 2

    current_tmp = im.as_fieldop(stencil)(im.ref("in_field1"), im.ref("in_field2"))
    current_domain = im.domain(
        common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (iterations - 1, 7 + iterations - 1)}
    )
    current_expected = im.as_fieldop(stencil, current_domain)(
        im.ref("in_field1"), im.ref("in_field2")
    )

    for n in range(1, iterations):
        previous_tmp = current_tmp
        previous_expected = current_expected

        current_tmp = im.as_fieldop(stencil)(im.ref("in_field1"), previous_tmp)
        current_domain = im.domain(
            common.GridType.CARTESIAN,
            {"IDim": (0, 11), "JDim": (iterations - 1 - n, 7 + iterations - 1 - n)},
        )
        current_expected = im.as_fieldop(stencil, current_domain)(
            im.ref("in_field1"), previous_expected
        )

    testee = current_tmp

    expected_domains = {
        "in_field1": SymbolicDomain.from_expr(
            im.domain(common.GridType.CARTESIAN, {"IDim": (1, 12), "JDim": (0, 7 + iterations - 1)})
        ),
        "in_field2": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (iterations, 7 + iterations)}
            )
        ),
    }

    actual_call, actual_domains = infer_as_fieldop(
        testee, SymbolicDomain.from_expr(current_domain), offset_provider
    )

    folded_domains = domains_dict_constant_folding(actual_domains)
    folded_call = as_fieldop_domains_constant_folding(actual_call)
    assert folded_call == current_expected
    assert folded_domains == expected_domains


def test_program(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    as_fieldop_tmp = im.as_fieldop(stencil)(im.ref("in_field"))
    as_fieldop = im.as_fieldop(stencil)(im.ref("tmp"))

    domain_tmp = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)},
    )
    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )

    params = create_params(["in_field", "out_field", "_gtmp_auto_domain"])

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

    expected_as_fieldop_tmp = im.as_fieldop(stencil, domain_tmp)(im.ref("in_field"))
    expected_as_fieldop = im.as_fieldop(stencil, domain)(im.ref("tmp"))

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

    run_test_program(testee, expected, offset_provider)


def test_program_two_tmps(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    as_fieldop_tmp1 = im.as_fieldop(stencil)(im.ref("in_field"))
    as_fieldop_tmp2 = im.as_fieldop(stencil)(im.ref("tmp1"))
    as_fieldop = im.as_fieldop(stencil)(im.ref("tmp2"))

    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_tmp1 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 13)},
    )
    domain_tmp2 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)},
    )

    params = create_params(["in_field", "out_field", "_gtmp_auto_domain"])

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

    expected_as_fieldop_tmp1 = im.as_fieldop(stencil, domain_tmp1)(im.ref("in_field"))
    expected_as_fieldop_tmp2 = im.as_fieldop(stencil, domain_tmp2)(im.ref("tmp1"))
    expected_as_fieldop = im.as_fieldop(stencil, domain)(im.ref("tmp2"))

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

    run_test_program(testee, expected, offset_provider)


@pytest.mark.xfail(raises=ValueError)
def test_program_ValueError(offset_provider):
    with pytest.raises(ValueError, match=r"Temporaries can only be used once within a program."):
        stencil = im.lambda_("arg0")(
            im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
        )

        as_fieldop_tmp = im.as_fieldop(stencil)(im.ref("in_field"))
        as_fieldop = im.as_fieldop(stencil)(im.ref("tmp"))

        domain = im.domain(
            common.GridType.CARTESIAN,
            {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
        )

        params = create_params(["in_field", "out_field", "_gtmp_auto_domain"])

        testee = infer_program(
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

    as_fieldop_tmp1 = im.as_fieldop(stencil)(im.ref("in_field1"), im.ref("in_field2"))
    as_fieldop_tmp2 = im.as_fieldop(stencil_tmp)(im.ref("tmp1"))
    as_fieldop_out1 = im.as_fieldop(stencil_tmp)(im.ref("tmp2"))
    as_fieldop_tmp3 = im.as_fieldop(stencil)(im.ref("tmp1"), im.ref("in_field2"))
    as_fieldop_out2 = im.as_fieldop(stencil_tmp_minus)(im.ref("tmp2"), im.ref("tmp3"))

    domain_out = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_tmp3 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_tmp2 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 12)},
    )
    domain_tmp1 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 13)},
    )
    params = create_params(
        ["in_field1", "in_field2", "out_field1", "out_field2", "_gtmp_auto_domain"]
    )

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

    expected_as_fieldop_tmp1 = im.as_fieldop(stencil, domain_tmp1)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_as_fieldop_tmp2 = im.as_fieldop(stencil_tmp, domain_tmp2)(im.ref("tmp1"))
    expected_as_fieldop_out1 = im.as_fieldop(stencil_tmp, domain_out)(im.ref("tmp2"))
    expected_as_fieldop_tmp3 = im.as_fieldop(stencil, domain_tmp3)(
        im.ref("tmp1"), im.ref("in_field2")
    )
    expected_as_fieldop_out2 = im.as_fieldop(stencil_tmp_minus, domain_out)(
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

    run_test_program(testee, expected, offset_provider)

#TODO: expected_expr