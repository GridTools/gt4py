# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(SF-N): test scan operator

import numpy as np
from typing import Iterable

from gt4py import eve
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.infer_domain import (
    infer_as_fieldop,
    infer_program,
    infer_cond,
    infer_let,
)
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
    stencil: itir.Lambda,
    domain: itir.FunCall,
    expected_domain_dict: Dict[str, Dict[str | Dimension, tuple[itir.Expr, itir.Expr]]],
    offset_provider: Dict[str, Dimension],
    *,
    refs: Iterable[itir.SymRef] = None,
    domain_type: str = common.GridType.CARTESIAN,
) -> None:
    if refs is None:
        refs = [f"in_field{i+1}" for i in range(0, len(stencil.params))]

    testee = im.as_fieldop(stencil)(*refs)
    expected = im.as_fieldop(stencil, domain)(*refs)

    actual_call, actual_domains = infer_as_fieldop(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    folded_domains = constant_fold_accessed_domains(actual_domains)
    expected_domains = {
        ref: SymbolicDomain.from_expr(im.domain(domain_type, d))
        for ref, d in expected_domain_dict.items()
    }

    assert actual_call == expected
    assert folded_domains == expected_domains


def run_test_program(
    testee: itir.Program, expected: itir.Program, offset_provider: dict[str, Dimension]
) -> None:
    actual_program = infer_program(testee, offset_provider)

    folded_program = constant_fold_domain_exprs(actual_program)
    assert folded_program == expected


def run_test_let(
    testee: itir.FunCall,
    expected: itir.FunCall,
    domain: itir.FunCall,
    expected_domains: Dict[str, SymbolicDomain],
    offset_provider: dict[str, Dimension],
):
    actual_program, actual_domains = infer_let(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    actual_call, actual_domains = infer_let(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )
    folded_call = constant_fold_domain_exprs(actual_call)
    folded_domains = constant_fold_accessed_domains(actual_domains)

    assert folded_call == expected
    assert folded_domains == expected_domains


class _ConstantFoldDomainsExprs(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
            return ConstantFolding.apply(node)
        return self.generic_visit(node)


def constant_fold_domain_exprs(arg: itir.Node) -> itir.Node:
    return _ConstantFoldDomainsExprs().visit(arg)


def constant_fold_accessed_domains(domains: Dict[str, SymbolicDomain]) -> Dict[str, SymbolicDomain]:
    return {
        k: SymbolicDomain.from_expr(constant_fold_domain_exprs(v.as_expr()))
        for k, v in domains.items()
    }


def test_forward_difference_x(offset_provider):
    stencil = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11)})
    expected_accessed_domains = {
        "in_field1": {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)}
    }
    run_test_as_fieldop(stencil, domain, expected_accessed_domains, offset_provider)


def test_multi_length_shift(offset_provider):
    stencil = im.lambda_("arg0")(
        im.deref(
            im.call(
                im.call("shift")(
                    im.ensure_offset("Ioff"),
                    im.ensure_offset(1),
                    im.ensure_offset("Ioff"),
                    im.ensure_offset(2),
                )
            )("arg0")
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11)})
    expected_accessed_domains = {
        "in_field1": {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (3, 14)}
    }
    run_test_as_fieldop(stencil, domain, expected_accessed_domains, offset_provider)


def test_unused_input(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(im.deref("arg0"))

    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11)})
    expected_accessed_domains = {
        "in_field1": {"IDim": (0, 11)},
    }
    run_test_as_fieldop(
        stencil,
        domain,
        expected_accessed_domains,
        offset_provider,
    )


def test_unstructured_shift(unstructured_offset_provider):
    stencil = im.lambda_("arg0")(im.deref(im.shift("E2V", 1)("arg0")))
    domain = im.domain(
        common.GridType.UNSTRUCTURED,
        {common.Dimension(value="Edge", kind=common.DimensionKind.HORIZONTAL): (0, 1)},
    )
    expected_accessed_domains = {
        "in_field1": {
            common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL): (0, 2)
        }
    }

    run_test_as_fieldop(
        stencil,
        domain,
        expected_accessed_domains,
        unstructured_offset_provider,
        domain_type=common.GridType.UNSTRUCTURED,
    )


def test_laplace(offset_provider):
    stencil = im.lambda_("arg0")(
        im.plus(
            im.plus(
                im.plus(
                    im.plus(
                        im.multiplies_(-4.0, im.deref("arg0")),
                        im.deref(im.shift("Ioff", 1)("arg0")),
                    ),
                    im.deref(im.shift("Joff", 1)("arg0")),
                ),
                im.deref(im.shift("Ioff", -1)("arg0")),
            ),
            im.deref(im.shift("Joff", -1)("arg0")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})
    expected_accessed_domains = {"in_field1": {"IDim": (-1, 12), "JDim": (-1, 8)}}

    run_test_as_fieldop(stencil, domain, expected_accessed_domains, offset_provider)


def test_shift_x_y_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", -1)("arg0")),
            im.deref(im.shift("Joff", 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})
    expected_accessed_domains = {
        "in_field1": {"IDim": (-1, 10), "JDim": (0, 7)},
        "in_field2": {"IDim": (0, 11), "JDim": (1, 8)},
    }
    run_test_as_fieldop(
        stencil,
        domain,
        expected_accessed_domains,
        offset_provider,
    )


def test_shift_x_y_two_inputs_literal(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", -1)("arg0")),
            im.deref(im.shift("Joff", 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (0, 7)})
    expected_accessed_domains = {
        "in_field1": {"IDim": (-1, 10), "JDim": (0, 7)},
    }
    run_test_as_fieldop(
        stencil,
        domain,
        expected_accessed_domains,
        offset_provider,
        refs=(im.ref("in_field1"), 2),
    )


def test_shift_x_y_z_three_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1", "arg2")(
        im.plus(
            im.plus(
                im.deref(im.shift("Ioff", 1)("arg0")),
                im.deref(im.shift("Joff", 1)("arg1")),
            ),
            im.deref(im.shift("Koff", -1)("arg2")),
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
    )


def test_nested_stencils(offset_provider):
    inner_stencil = im.lambda_("arg0_tmp", "arg1_tmp")(
        im.plus(
            im.deref(im.shift("Ioff", 1)("arg0_tmp")),
            im.deref(im.shift("Joff", -1)("arg1_tmp")),
        )
    )
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", 1)("arg0")),
            im.deref(im.shift("Joff", -1)("arg1")),
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
    folded_domains = constant_fold_accessed_domains(actual_domains)
    folded_call = constant_fold_domain_exprs(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


@pytest.mark.parametrize("iterations", [3, 5])
def test_nested_stencils_n_times(offset_provider, iterations):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", 1)("arg0")),
            im.deref(im.shift("Joff", 1)("arg1")),
        )
    )
    assert iterations >= 2

    domain = im.domain(
        common.GridType.CARTESIAN, {"IDim": (0, 11), "JDim": (iterations - 1, 7 + iterations - 1)}
    )
    testee = im.as_fieldop(stencil)(im.ref("in_field1"), im.ref("in_field2"))
    expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), im.ref("in_field2"))

    for n in range(1, iterations):
        domain = im.domain(
            common.GridType.CARTESIAN,
            {"IDim": (0, 11), "JDim": (iterations - 1 - n, 7 + iterations - 1 - n)},
        )
        testee = im.as_fieldop(stencil)(im.ref("in_field1"), testee)
        expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), expected)

    testee = testee

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
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    folded_domains = constant_fold_accessed_domains(actual_domains)
    folded_call = constant_fold_domain_exprs(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


def test_program(offset_provider):
    stencil = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))

    applied_as_fieldop_tmp = im.as_fieldop(stencil)(im.ref("in_field"))
    applied_as_fieldop = im.as_fieldop(stencil)(im.ref("tmp"))

    domain_tmp = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)},
    )
    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )

    params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]

    testee = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=AUTO_DOMAIN, dtype=float_type)],
        body=[
            itir.SetAt(expr=applied_as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
            itir.SetAt(expr=applied_as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    expected_expr_tmp = im.as_fieldop(stencil, domain_tmp)(im.ref("in_field"))
    expected_epxr = im.as_fieldop(stencil, domain)(im.ref("tmp"))

    expected = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_tmp, dtype=float_type)],
        body=[
            itir.SetAt(expr=expected_expr_tmp, domain=domain_tmp, target=im.ref("tmp")),
            itir.SetAt(expr=expected_epxr, domain=domain, target=im.ref("out_field")),
        ],
    )

    run_test_program(testee, expected, offset_provider)


def test_program_two_tmps(offset_provider):
    stencil = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))

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

    params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]

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

    expected_expr_tmp1 = im.as_fieldop(stencil, domain_tmp1)(im.ref("in_field"))
    expected_expr_tmp2 = im.as_fieldop(stencil, domain_tmp2)(im.ref("tmp1"))
    expected_expr = im.as_fieldop(stencil, domain)(im.ref("tmp2"))

    expected = itir.Program(
        id="forward_diff_with_two_tmps",
        function_definitions=[],
        params=params,
        declarations=[
            itir.Temporary(id="tmp1", domain=domain_tmp1, dtype=float_type),
            itir.Temporary(id="tmp2", domain=domain_tmp2, dtype=float_type),
        ],
        body=[
            itir.SetAt(expr=expected_expr_tmp1, domain=domain_tmp1, target=im.ref("tmp1")),
            itir.SetAt(expr=expected_expr_tmp2, domain=domain_tmp2, target=im.ref("tmp2")),
            itir.SetAt(expr=expected_expr, domain=domain, target=im.ref("out_field")),
        ],
    )

    run_test_program(testee, expected, offset_provider)


@pytest.mark.xfail(raises=ValueError)
def test_program_ValueError(offset_provider):
    with pytest.raises(ValueError, match=r"Temporaries can only be used once within a program."):
        stencil = im.lambda_("arg0")(im.deref("arg0"))

        as_fieldop_tmp = im.as_fieldop(stencil)(im.ref("in_field"))
        as_fieldop = im.as_fieldop(stencil)(im.ref("tmp"))

        domain = im.domain(
            common.GridType.CARTESIAN,
            {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
        )

        params = [im.sym(name) for name in ["in_field", "out_field", "_gtmp_auto_domain"]]

        infer_program(
            itir.Program(
                id="forward_diff_with_tmp",
                function_definitions=[],
                params=params,
                declarations=[itir.Temporary(id="tmp", domain=AUTO_DOMAIN, dtype=float_type)],
                body=[
                    # target occurs twice here which is prohibited
                    itir.SetAt(expr=as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
                    itir.SetAt(expr=as_fieldop_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
                    itir.SetAt(expr=as_fieldop, domain=domain, target=im.ref("out_field")),
                ],
            ),
            offset_provider,
        )


def test_program_tree_tmps_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg1"))
    )
    stencil_tmp = im.lambda_("arg0")(
        im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0"))
    )
    stencil_tmp_minus = im.lambda_("arg0", "arg1")(
        im.minus(im.deref(im.shift("Ioff", -1)("arg0")), im.deref("arg1"))
    )

    as_fieldop_tmp1 = im.as_fieldop(stencil)(im.ref("in_field1"), im.ref("in_field2"))
    as_fieldop_tmp2 = im.as_fieldop(stencil_tmp)(im.ref("tmp1"))
    as_fieldop_out1 = im.as_fieldop(stencil_tmp)(im.ref("tmp2"))
    as_fieldop_tmp3 = im.as_fieldop(stencil)(im.ref("tmp1"), im.ref("in_field2"))
    as_fieldop_out2 = im.as_fieldop(stencil_tmp_minus)(im.ref("tmp2"), im.ref("tmp3"))

    domain_tmp1 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 13)},
    )
    domain_tmp2 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 12)},
    )
    domain_tmp3 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_out = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    params = [
        im.sym(name)
        for name in ["in_field1", "in_field2", "out_field1", "out_field2", "_gtmp_auto_domain"]
    ]

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

    expected_expr_tmp1 = im.as_fieldop(stencil, domain_tmp1)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_expr_tmp2 = im.as_fieldop(stencil_tmp, domain_tmp2)(im.ref("tmp1"))
    expected_expr_out1 = im.as_fieldop(stencil_tmp, domain_out)(im.ref("tmp2"))
    expected_expr_tmp3 = im.as_fieldop(stencil, domain_tmp3)(im.ref("tmp1"), im.ref("in_field2"))
    expected_expr_out2 = im.as_fieldop(stencil_tmp_minus, domain_out)(
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
            itir.SetAt(expr=expected_expr_tmp1, domain=domain_tmp1, target=im.ref("tmp1")),
            itir.SetAt(expr=expected_expr_tmp2, domain=domain_tmp2, target=im.ref("tmp2")),
            itir.SetAt(expr=expected_expr_out1, domain=domain_out, target=im.ref("out_field1")),
            itir.SetAt(expr=expected_expr_tmp3, domain=domain_tmp3, target=im.ref("tmp3")),
            itir.SetAt(expr=expected_expr_out2, domain=domain_out, target=im.ref("out_field2")),
        ],
    )

    run_test_program(testee, expected, offset_provider)


def test_cond(offset_provider):
    stencil1 = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    field_1 = im.as_fieldop(stencil1)(im.ref("in_field1"))
    inner_stencil2 = im.lambda_("arg0_tmp", "arg1_tmp")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0_tmp")),
            im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg1_tmp")),
        )
    )
    stencil2 = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")),
            im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg1")),
        )
    )
    tmp2 = im.as_fieldop(inner_stencil2)(im.ref("in_field1"), im.ref("in_field2"))
    field_2 = im.as_fieldop(stencil2)(im.ref("in_field2"), tmp2)

    cond = im.deref("cond_")

    testee = im.cond(cond, field_1, field_2)

    domain = im.domain(common.GridType.CARTESIAN, {"IDim": (0, 11)})
    domain_tmp = im.domain(common.GridType.CARTESIAN, {"IDim": (-1, 10)})
    expected_domains_dict = {
        "in_field1": {
            common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 12)
        },
        "in_field2": {
            common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-2, 12)
        },
    }
    expected_tmp2 = im.as_fieldop(inner_stencil2, domain_tmp)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_field_1 = im.as_fieldop(stencil1, domain)(im.ref("in_field1"))
    expected_field_2 = im.as_fieldop(stencil2, domain)(im.ref("in_field2"), expected_tmp2)

    expected = im.cond(cond, expected_field_1, expected_field_2)

    actual_call, actual_domains = infer_cond(
        testee, SymbolicDomain.from_expr(domain), offset_provider
    )

    folded_domains = constant_fold_accessed_domains(actual_domains)
    expected_domains = {
        ref: SymbolicDomain.from_expr(im.domain(common.GridType.CARTESIAN, d))
        for ref, d in expected_domains_dict.items()
    }
    folded_call = constant_fold_domain_exprs(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


def test_simple_let(offset_provider):

    testee = im.let("a", 1)("b")
    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    expected = im.let("a", 1)("b")

    expected_domains = {
        "b": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
            )
        )
    }
    run_test_let(testee, expected, domain, expected_domains, offset_provider)


def test_let(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    testee = im.let(
        "inner",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("squared_shift"),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            )
        )("inner")
    )
    testee2 = im.as_fieldop(
        im.lambda_("it")(
            im.multiplies_(
                im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
            )
        )
    )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("squared_shift"))
    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_squared = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 10)},
    )
    expected = im.let(
        "inner",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_squared)(
            "squared_shift"
        ),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            ),
            domain,
        )("inner")
    )
    expected2 = im.as_fieldop(
        im.lambda_("it")(
            im.multiplies_(
                im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
            )
        ),
        domain,
    )(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_squared)(
            "squared_shift"
        )
    )
    expected_domains = {
        "squared_shift": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
            )
        )
    }
    run_test_let(testee, expected, domain, expected_domains, offset_provider)

    actual_call2, actual_domains2 = infer_as_fieldop(
        testee2, SymbolicDomain.from_expr(domain), offset_provider
    )
    folded_domains2 = constant_fold_accessed_domains(actual_domains2)
    folded_call2 = constant_fold_domain_exprs(actual_call2)
    assert folded_call2 == expected2
    assert expected_domains == folded_domains2


def test_let_two_inputs(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    testee = im.let(
        ("inner1", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("shift1")),
        ("inner2", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", -1)("it"))))("shift2")),
    )(
        im.as_fieldop(
            im.lambda_("it1", "it2")(
                im.multiplies_(
                    im.deref("it1"), im.deref("it2")
                )
            )
        )("inner1", "inner2")
    )
    domain = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_m110 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 10)},
    )
    domain_112 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (1, 12)},
    )
    expected = im.let(
        (
            "inner1",
            im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_m110)(
                "shift1"
            ),
        ),
        (
            "inner2",
            im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", -1)("it"))), domain_112)(
                "shift2"
            ),
        ),
    )(
        im.as_fieldop(
            im.lambda_("it1", "it2")(
                im.multiplies_(
                    im.deref("it1"), im.deref("it2")
                )
            ),
            domain,
        )("inner1", "inner2")
    )
    expected_domains = {
        "shift1": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
            )
        ),
        "shift2": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
            )
        ),
    }
    run_test_let(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_fun_expr(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    testee = im.let(
        "pow1_p1", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("outer")
    )(
        im.let(
            "pow2_m1",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                )
            )("pow1_p1"),
        )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))))("pow2_m1"))
    )

    domain_011 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_112 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (1, 12)},
    )
    domain_213 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (2, 13)},
    )

    expected = im.let(
        "pow1_p1",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_112)("outer"),
    )(
        im.let(
            "pow2_m1",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                ),
                domain_213,
            )("pow1_p1"),
        )(
            im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))), domain_011)(
                "pow2_m1"
            )
        )
    )
    expected_domains = {
        "outer": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (2, 13)},
            )
        )
    }
    run_test_let(testee, expected, domain_011, expected_domains, offset_provider)


def test_nested_let_fun_expr_shaddowing(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    testee = im.let("a", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("outer"))(
        im.let(
            "a",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                )
            )("a"),
        )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))))("a"))
    )

    # (λ(a) → (λ(a) → as_fieldop(λ(it) → ·⟪Ioffₒ, 2 ₒ⟫(it), c⟨ IDimₕ: [0, 11) ⟩)(a))(
    #     as_fieldop(λ(it, it2) → ·⟪Ioffₒ, -1ₒ⟫(it) × ·⟪Ioffₒ, -1 ₒ⟫(it2), c⟨ IDimₕ: [2, 13) ⟩)(a, outer)
    # ))(as_fieldop(λ(it) → ·⟪Ioffₒ, 1ₒ⟫(it), c⟨ IDimₕ: [1, 12) ⟩)(outer))

    domain_011 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_112 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (1, 12)},
    )
    domain_213 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (2, 13)},
    )

    expected = im.let(
        "a", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_112)("outer")
    )(
        im.let(
            "a",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                ),
                domain_213,
            )("a"),
        )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))), domain_011)("a"))
    )
    expected_domains = {
        "a": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (2, 13)},
            )
        )
    }
    run_test_let(testee, expected, domain_011, expected_domains, offset_provider)


def test_double_nested_let_fun_expr(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )

    testee = im.let(
        "pow1_p1", im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("outer")
    )(
        im.let(
            "pow2_m1",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                )
            )("pow1_p1"),
        )(
            im.let(
                "pow4_m1",
                im.as_fieldop(
                    im.lambda_("it")(
                        im.multiplies_(
                            im.deref(im.shift("Ioff", -1)("it")),
                            im.deref(im.shift("Ioff", -1)("it")),
                        )
                    )
                )("pow2_m1"),
            )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))))("pow4_m1"))
        )
    )
    im.let("power_1", im.plus("x", 1))(
        im.let("power_2", im.multiplies_("power_1", "power_1"))(
            im.let("power_4", im.multiplies_("power_2", "power_2"))(im.plus("power_4", "power_1"))
        )
    )
    domain_011 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_112 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (1, 12)},
    )
    domain_213 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (2, 13)},
    )

    expected = im.let(
        "pow1_p1",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_011)("outer"),
    )(
        im.let(
            "pow2_m1",
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                ),
                domain_112,
            )("pow1_p1"),
        )(
            im.let(
                "pow4_m1",
                im.as_fieldop(
                    im.lambda_("it")(
                        im.multiplies_(
                            im.deref(im.shift("Ioff", -1)("it")),
                            im.deref(im.shift("Ioff", -1)("it")),
                        )
                    ),
                    domain_213,
                )("pow2_m1"),
            )(
                im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 2)("it"))), domain_011)(
                    "pow4_m1"
                )
            )
        )
    )

    expected_domains = {
        "outer": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (1, 12)},
            )
        )
    }

    run_test_let(testee, expected, domain_011, expected_domains, offset_provider)


def test_nested_let_args(offset_provider):
    stencil = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), 1)("arg0")), im.deref("arg0"))
    )
    testee = im.let(
        "inner",
        im.let(
            "inner_arg",
            im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))))("squared_shift"),
        )(
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                )
            )("inner_arg")
        ),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            )
        )("inner")
    )

    domain_011 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_m110 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 10)},
    )
    domain_m29 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-2, 9)},
    )
    expected = im.let(
        "inner",
        im.let(
            "inner_arg",
            im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", 1)("it"))), domain_m29)(
                "squared_shift"
            ),
        )(
            im.as_fieldop(
                im.lambda_("it")(
                    im.multiplies_(
                        im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                    )
                ),
                domain_m110,
            )("inner_arg")
        ),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            ),
            domain_011,
        )("inner")
    )

    expected_domains = {
        "squared_shift": SymbolicDomain.from_expr(
            im.domain(
                common.GridType.CARTESIAN,
                {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 10)},
            )
        )
    }
    run_test_let(testee, expected, domain_011, expected_domains, offset_provider)


def test_program_let(offset_provider):
    stencil_tmp = im.lambda_("arg0")(
        im.minus(im.deref(im.shift(itir.SymbolRef("Ioff"), -1)("arg0")), im.deref("arg0"))
    )

    let_tmp = im.let(
        "inner",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", -1)("it"))))("squared_shift"),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            )
        )("inner")
    )

    as_fieldop = im.as_fieldop(stencil_tmp)(im.ref("tmp"))

    domain_m210 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-2, 10)},
    )
    domain_011 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (0, 11)},
    )
    domain_m111 = im.domain(
        common.GridType.CARTESIAN,
        {common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL): (-1, 11)},
    )

    params = [
        im.sym(name)
        for name in ["in_field", "out_field", "new_field", "squared_shift", "_gtmp_auto_domain"]
    ]
    for param in params[:2]:
        param.dtype = ("float64", False)
        param.kind = "Iterator"

    testee = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=AUTO_DOMAIN, dtype=float_type)],
        body=[
            itir.SetAt(expr=let_tmp, domain=AUTO_DOMAIN, target=im.ref("tmp")),
            itir.SetAt(expr=as_fieldop, domain=domain_011, target=im.ref("out_field")),
        ],
    )

    expected_let = im.let(
        "inner",
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("Ioff", -1)("it"))), domain_m210)(
            "squared_shift"
        ),
    )(
        im.as_fieldop(
            im.lambda_("it")(
                im.multiplies_(
                    im.deref(im.shift("Ioff", -1)("it")), im.deref(im.shift("Ioff", -1)("it"))
                )
            ),
            domain_m111,
        )("inner")
    )

    expected_as_fieldop = im.as_fieldop(stencil_tmp, domain_011)(im.ref("tmp"))

    expected = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_m111, dtype=float_type)],
        body=[
            itir.SetAt(expr=expected_let, domain=domain_m111, target=im.ref("tmp")),
            itir.SetAt(expr=expected_as_fieldop, domain=domain_011, target=im.ref("out_field")),
        ],
    )

    run_test_program(testee, expected, offset_provider)
