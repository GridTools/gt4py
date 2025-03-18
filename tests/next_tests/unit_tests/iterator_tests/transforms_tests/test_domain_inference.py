# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# TODO(SF-N): test scan operator

from typing import Iterable, Literal, Optional, Union

import numpy as np
import pytest

from gt4py import eve
from gt4py.next import common, constructors, utils
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.type_system import type_specifications as ts


float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
JDim = common.Dimension(value="JDim", kind=common.DimensionKind.HORIZONTAL)
KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
Edge = common.Dimension(value="Edge", kind=common.DimensionKind.HORIZONTAL)
E2VDim = common.Dimension(value="E2V", kind=common.DimensionKind.LOCAL)


@pytest.fixture
def offset_provider():
    return {"Ioff": IDim, "Joff": JDim, "Koff": KDim}


@pytest.fixture
def unstructured_offset_provider():
    return {
        "E2V": constructors.as_connectivity(
            domain={Edge: 1, E2VDim: 2},
            codomain=Vertex,
            data=np.array([[0, 1]], dtype=np.int32),
        )
    }


def premap_field(
    field: itir.Expr, dim: str, offset: int, domain: Optional[itir.FunCall] = None
) -> itir.Expr:
    return im.as_fieldop(im.lambda_("it")(im.deref(im.shift(dim, offset)("it"))), domain)(field)


def setup_test_as_fieldop(
    stencil: itir.Lambda | Literal["deref"],
    domain: itir.FunCall,
    *,
    refs: Iterable[itir.SymRef] = None,
) -> tuple[itir.FunCall, itir.FunCall]:
    if refs is None:
        assert isinstance(stencil, itir.Lambda)
        refs = [f"in_field{i+1}" for i in range(0, len(stencil.params))]

    testee = im.as_fieldop(stencil)(*refs)
    expected = im.as_fieldop(stencil, domain)(*refs)
    return testee, expected


def run_test_program(
    testee: itir.Program, expected: itir.Program, offset_provider: common.OffsetProvider
) -> None:
    actual_program = infer_domain.infer_program(testee, offset_provider=offset_provider)

    folded_program = constant_fold_domain_exprs(actual_program)
    assert folded_program == expected


def run_test_expr(
    testee: itir.FunCall,
    expected: itir.FunCall,
    domain: itir.FunCall,
    expected_domains: dict[str, itir.Expr | dict[str | Dimension, tuple[itir.Expr, itir.Expr]]],
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
    allow_uninferred: bool = False,
):
    actual_call, actual_domains = infer_domain.infer_expr(
        testee,
        domain_utils.SymbolicDomain.from_expr(domain),
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
        allow_uninferred=allow_uninferred,
    )
    folded_call = constant_fold_domain_exprs(actual_call)
    folded_domains = constant_fold_accessed_domains(actual_domains) if actual_domains else None

    grid_type = str(domain.fun.id)

    def canonicalize_domain(d):
        if isinstance(d, dict):
            return im.domain(grid_type, d)
        elif isinstance(d, (itir.FunCall, infer_domain.DomainAccessDescriptor)):
            return d
        raise AssertionError()

    expected_domains = {ref: canonicalize_domain(d) for ref, d in expected_domains.items()}

    assert folded_call == expected
    assert folded_domains == expected_domains


class _ConstantFoldDomainsExprs(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall):
        if cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain")):
            return ConstantFolding.apply(node)
        return self.generic_visit(node)


def constant_fold_domain_exprs(arg: itir.Node) -> itir.Node:
    return _ConstantFoldDomainsExprs().visit(arg)


def constant_fold_accessed_domains(
    domains: infer_domain.AccessedDomains,
) -> infer_domain.AccessedDomains:
    def fold_domain(
        domain: domain_utils.SymbolicDomain | Literal[infer_domain.DomainAccessDescriptor.NEVER],
    ):
        if isinstance(domain, infer_domain.DomainAccessDescriptor):
            return domain
        return constant_fold_domain_exprs(domain.as_expr())

    return {k: utils.tree_map(fold_domain)(v) for k, v in domains.items()}


def translate_domain(
    domain: itir.FunCall,
    shifts: dict[str, tuple[itir.Expr, itir.Expr]],
    offset_provider: common.OffsetProvider,
) -> domain_utils.SymbolicDomain:
    shift_tuples = [
        (
            im.ensure_offset(d),
            im.ensure_offset(r),
        )
        for d, r in shifts.items()
    ]

    shift_list = [item for sublist in shift_tuples for item in sublist]

    translated_domain_expr = domain_utils.SymbolicDomain.from_expr(domain).translate(
        shift_list, offset_provider=offset_provider
    )

    return constant_fold_domain_exprs(translated_domain_expr.as_expr())


def test_forward_difference_x(offset_provider):
    stencil = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected_domains = {"in_field1": {IDim: (0, 12)}}
    testee, expected = setup_test_as_fieldop(stencil, domain)
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_deref(offset_provider):
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected_domains = {"in_field": {IDim: (0, 11)}}
    testee, expected = setup_test_as_fieldop("deref", domain, refs=["in_field"])
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


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
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected_domains = {"in_field1": {IDim: (3, 14)}}
    testee, expected = setup_test_as_fieldop(stencil, domain)
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_unstructured_shift(unstructured_offset_provider):
    stencil = im.lambda_("arg0")(im.deref(im.shift("E2V", 1)("arg0")))
    domain = im.domain(common.GridType.UNSTRUCTURED, {Edge: (0, 1)})
    expected_domains = {"in_field1": {Vertex: (0, 2)}}

    testee, expected = setup_test_as_fieldop(
        stencil,
        domain,
    )
    run_test_expr(testee, expected, domain, expected_domains, unstructured_offset_provider)


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
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (1, 11), JDim: (1, 7)})
    expected_domains = {"in_field1": {IDim: (0, 12), JDim: (0, 8)}}

    testee, expected = setup_test_as_fieldop(stencil, domain)
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_shift_x_y_two_inputs(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", -1)("arg0")),
            im.deref(im.shift("Joff", 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (1, 11), JDim: (0, 7)})
    expected_domains = {
        "in_field1": {IDim: (0, 10), JDim: (0, 7)},
        "in_field2": {IDim: (1, 11), JDim: (1, 8)},
    }
    testee, expected = setup_test_as_fieldop(
        stencil,
        domain,
    )
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_shift_x_y_two_inputs_literal(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", -1)("arg0")),
            im.deref(im.shift("Joff", 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (1, 11), JDim: (0, 7)})
    expected_domains = {
        "in_field1": {IDim: (0, 10), JDim: (0, 7)},
    }
    testee, expected = setup_test_as_fieldop(
        stencil,
        domain,
        refs=(im.ref("in_field1"), 2),
    )
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


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
    domain_dict = {IDim: (0, 11), JDim: (0, 7), KDim: (1, 3)}
    expected_domains = {
        "in_field1": {IDim: (1, 12), JDim: (0, 7), KDim: (1, 3)},
        "in_field2": {IDim: (0, 11), JDim: (1, 8), KDim: (1, 3)},
        "in_field3": {IDim: (0, 11), JDim: (0, 7), KDim: (0, 2)},
    }
    testee, expected = setup_test_as_fieldop(
        stencil,
        im.domain(common.GridType.CARTESIAN, domain_dict),
    )
    run_test_expr(
        testee,
        expected,
        im.domain(common.GridType.CARTESIAN, domain_dict),
        expected_domains,
        offset_provider,
    )


def test_two_params_same_arg(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref("arg0"),
            im.deref(im.shift("Ioff", 1)("arg1")),
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected_domains = {
        "in_field": {IDim: (0, 12)},
    }
    testee, expected = setup_test_as_fieldop(stencil, domain, refs=["in_field", "in_field"])
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


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

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11), JDim: (1, 7)})
    domain_inner = translate_domain(domain, {"Ioff": 0, "Joff": -1}, offset_provider)

    expected_inner = im.as_fieldop(inner_stencil, domain_inner)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), expected_inner)

    expected_domains = {
        "in_field1": im.domain(common.GridType.CARTESIAN, {IDim: (1, 12), JDim: (0, 7)}),
        "in_field2": translate_domain(domain, {"Ioff": 0, "Joff": -2}, offset_provider),
    }
    actual_call, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
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
        common.GridType.CARTESIAN, {IDim: (0, 11), JDim: (iterations - 1, 7 + iterations - 1)}
    )
    testee = im.as_fieldop(stencil)(im.ref("in_field1"), im.ref("in_field2"))
    expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), im.ref("in_field2"))

    for n in range(1, iterations):
        domain = im.domain(
            common.GridType.CARTESIAN,
            {IDim: (0, 11), JDim: (iterations - 1 - n, 7 + iterations - 1 - n)},
        )
        testee = im.as_fieldop(stencil)(im.ref("in_field1"), testee)
        expected = im.as_fieldop(stencil, domain)(im.ref("in_field1"), expected)

    testee = testee

    expected_domains = {
        "in_field1": im.domain(
            common.GridType.CARTESIAN, {IDim: (1, 12), JDim: (0, 7 + iterations - 1)}
        ),
        "in_field2": im.domain(
            common.GridType.CARTESIAN, {IDim: (0, 11), JDim: (iterations, 7 + iterations)}
        ),
    }

    actual_call, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    folded_domains = constant_fold_accessed_domains(actual_domains)
    folded_call = constant_fold_domain_exprs(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


def test_unused_input(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(im.deref("arg0"))

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected_domains = {
        "in_field1": {IDim: (0, 11)},
        "in_field2": infer_domain.DomainAccessDescriptor.NEVER,
    }
    testee, expected = setup_test_as_fieldop(
        stencil,
        domain,
    )
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_let_unused_field(offset_provider):
    testee = im.let("a", "c")("b")
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.let("a", "c")("b")
    expected_domains = {"b": {IDim: (0, 11)}, "c": infer_domain.DomainAccessDescriptor.NEVER}

    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_program(offset_provider):
    stencil = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))

    applied_as_fieldop_tmp = im.as_fieldop(stencil)(im.ref("in_field"))
    applied_as_fieldop = im.as_fieldop(stencil)(im.ref("tmp"))

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_tmp = im.domain(common.GridType.CARTESIAN, {IDim: (0, 12)})

    params = [im.sym(name) for name in ["in_field", "out_field"]]

    testee = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_tmp, dtype=float_type)],
        body=[
            itir.SetAt(expr=applied_as_fieldop_tmp, domain=domain_tmp, target=im.ref("tmp")),
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


def test_program_make_tuple(offset_provider):
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    params = [im.sym(name) for name in ["in_field", "out_field"]]

    testee = itir.Program(
        id="make_tuple_prog",
        function_definitions=[],
        params=params,
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.make_tuple(im.as_fieldop("deref")("in_field"), "in_field"),
                domain=domain,
                target=im.ref("out_field"),
            ),
        ],
    )

    expected = itir.Program(
        id="make_tuple_prog",
        function_definitions=[],
        params=params,
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.make_tuple(im.as_fieldop("deref", domain)("in_field"), "in_field"),
                domain=domain,
                target=im.ref("out_field"),
            ),
        ],
    )

    run_test_program(testee, expected, offset_provider)


def test_cond(offset_provider):
    stencil1 = im.lambda_("arg0")(im.minus(im.deref(im.shift("Ioff", 1)("arg0")), im.deref("arg0")))
    field_1 = im.as_fieldop(stencil1)(im.ref("in_field1"))
    tmp_stencil2 = im.lambda_("arg0_tmp", "arg1_tmp")(
        im.plus(
            im.deref(im.shift("Ioff", 1)("arg0_tmp")),
            im.deref(im.shift("Ioff", -1)("arg1_tmp")),
        )
    )
    stencil2 = im.lambda_("arg0", "arg1")(
        im.plus(
            im.deref(im.shift("Ioff", 1)("arg0")),
            im.deref(im.shift("Ioff", -1)("arg1")),
        )
    )
    tmp2 = im.as_fieldop(tmp_stencil2)(im.ref("in_field1"), im.ref("in_field2"))
    field_2 = im.as_fieldop(stencil2)(im.ref("in_field2"), tmp2)

    cond = im.deref("cond_")

    testee = im.if_(cond, field_1, field_2)

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (2, 13)})
    domain_tmp = translate_domain(domain, {"Ioff": -1}, offset_provider)
    expected_domains_dict = {"in_field1": {IDim: (2, 14)}, "in_field2": {IDim: (0, 14)}}
    expected_tmp2 = im.as_fieldop(tmp_stencil2, domain_tmp)(
        im.ref("in_field1"), im.ref("in_field2")
    )
    expected_field_1 = im.as_fieldop(stencil1, domain)(im.ref("in_field1"))
    expected_field_2 = im.as_fieldop(stencil2, domain)(im.ref("in_field2"), expected_tmp2)

    expected = im.if_(cond, expected_field_1, expected_field_2)

    actual_call, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    folded_domains = constant_fold_accessed_domains(actual_domains)
    expected_domains = {
        ref: im.domain(common.GridType.CARTESIAN, d) for ref, d in expected_domains_dict.items()
    }
    folded_call = constant_fold_domain_exprs(actual_call)
    assert folded_call == expected
    assert folded_domains == expected_domains


def test_let_scalar_expr(offset_provider):
    testee = im.let("a", 1)(im.op_as_fieldop(im.plus)("a", "b"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.let("a", 1)(im.op_as_fieldop(im.plus, domain)("a", "b"))
    expected_domains = {"b": {IDim: (0, 11)}}

    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_simple_let(offset_provider):
    testee = im.let("a", premap_field("in_field", "Ioff", 1))("a")
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.let("a", premap_field("in_field", "Ioff", 1, domain))("a")

    expected_domains = {"in_field": translate_domain(domain, {"Ioff": 1}, offset_provider)}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_simple_let2(offset_provider):
    testee = im.let("a", "in_field")(premap_field("a", "Ioff", 1))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.let("a", "in_field")(premap_field("a", "Ioff", 1, domain))

    expected_domains = {"in_field": translate_domain(domain, {"Ioff": 1}, offset_provider)}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_let(offset_provider):
    testee = im.let(
        "a",
        premap_field("in_field", "Ioff", 1),
    )(premap_field("a", "Ioff", 1))
    testee2 = premap_field(premap_field("in_field", "Ioff", 1), "Ioff", 1)
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_a = translate_domain(domain, {"Ioff": 1}, offset_provider)
    expected = im.let(
        "a",
        premap_field("in_field", "Ioff", 1, domain_a),
    )(premap_field("a", "Ioff", 1, domain))
    expected2 = premap_field(premap_field("in_field", "Ioff", 1, domain_a), "Ioff", 1, domain)
    expected_domains = {"in_field": translate_domain(domain, {"Ioff": 2}, offset_provider)}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)

    expected_domains_sym = {"in_field": translate_domain(domain, {"Ioff": 2}, offset_provider)}
    actual_call2, actual_domains2 = infer_domain.infer_expr(
        testee2, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )
    folded_domains2 = constant_fold_accessed_domains(actual_domains2)
    folded_call2 = constant_fold_domain_exprs(actual_call2)
    assert folded_call2 == expected2
    assert expected_domains_sym == folded_domains2


def test_let_two_inputs(offset_provider):
    multiply_stencil = im.lambda_("it1", "it2")(im.multiplies_(im.deref("it1"), im.deref("it2")))

    testee = im.let(
        ("inner1", premap_field("in_field1", "Ioff", 1)),
        ("inner2", premap_field("in_field2", "Ioff", -1)),
    )(im.as_fieldop(multiply_stencil)("inner1", "inner2"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_p1 = translate_domain(domain, {"Ioff": 1}, offset_provider)
    domain_m1 = translate_domain(domain, {"Ioff": -1}, offset_provider)
    expected = im.let(
        ("inner1", premap_field("in_field1", "Ioff", 1, domain)),
        ("inner2", premap_field("in_field2", "Ioff", -1, domain)),
    )(im.as_fieldop(multiply_stencil, domain)("inner1", "inner2"))
    expected_domains = {
        "in_field1": domain_p1,
        "in_field2": domain_m1,
    }
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_in_body(offset_provider):
    testee = im.let("inner1", premap_field("outer", "Ioff", 1))(
        im.let("inner2", premap_field("inner1", "Ioff", 1))(premap_field("inner2", "Ioff", 1))
    )

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_p1 = translate_domain(domain, {"Ioff": 1}, offset_provider)
    domain_p2 = translate_domain(domain, {"Ioff": 2}, offset_provider)
    domain_p3 = translate_domain(domain, {"Ioff": 3}, offset_provider)

    expected = im.let(
        "inner1",
        premap_field("outer", "Ioff", 1, domain_p2),
    )(
        im.let("inner2", premap_field("inner1", "Ioff", 1, domain_p1))(
            premap_field("inner2", "Ioff", 1, domain)
        )
    )
    expected_domains = {"outer": domain_p3}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_arg(offset_provider):
    testee = im.let("a", "in_field")(
        im.as_fieldop(
            im.lambda_("it1", "it2")(
                im.multiplies_(im.deref("it1"), im.deref(im.shift("Ioff", 1)("it2")))
            )
        )("a", "in_field")
    )

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})

    expected = im.let("a", "in_field")(
        im.as_fieldop(
            im.lambda_("it1", "it2")(
                im.multiplies_(im.deref("it1"), im.deref(im.shift("Ioff", 1)("it2")))
            ),
            domain,
        )("a", "in_field")
    )
    expected_domains = {"in_field": {IDim: (0, 12)}}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_arg_shadowed(offset_provider):
    testee = im.let("a", premap_field("in_field", "Ioff", 3))(
        im.let("a", premap_field("a", "Ioff", 2))(premap_field("a", "Ioff", 1))
    )

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_p1 = translate_domain(domain, {"Ioff": 1}, offset_provider)
    domain_p3 = translate_domain(domain, {"Ioff": 3}, offset_provider)
    domain_p6 = translate_domain(domain, {"Ioff": 6}, offset_provider)

    expected = im.let(
        "a",
        premap_field("in_field", "Ioff", 3, domain_p3),
    )(im.let("a", premap_field("a", "Ioff", 2, domain_p1))(premap_field("a", "Ioff", 1, domain)))
    expected_domains = {"in_field": domain_p6}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_arg_shadowed2(offset_provider):
    # test that if we shadow `in_field1` its accessed domain is not affected by the accesses
    # on the shadowed field
    testee = im.as_fieldop(
        im.lambda_("it1", "it2")(im.multiplies_(im.deref("it1"), im.deref("it2")))
    )(
        premap_field("in_field1", "Ioff", 1),  # only here we access `in_field1`
        im.let("in_field1", "in_field2")("in_field1"),  # here we actually access `in_field2`
    )

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_p1 = translate_domain(domain, {"Ioff": 1}, offset_provider)

    expected = im.as_fieldop(
        im.lambda_("it1", "it2")(im.multiplies_(im.deref("it1"), im.deref("it2"))), domain
    )(
        premap_field("in_field1", "Ioff", 1, domain),
        im.let("in_field1", "in_field2")("in_field1"),
    )
    expected_domains = {"in_field1": domain_p1, "in_field2": domain}
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_double_nested_let_fun_expr(offset_provider):
    testee = im.let("inner1", premap_field("outer", "Ioff", 1))(
        im.let("inner2", premap_field("inner1", "Ioff", -1))(
            im.let("inner3", premap_field("inner2", "Ioff", -1))(premap_field("inner3", "Ioff", 3))
        )
    )
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain_p1 = translate_domain(domain, {"Ioff": 1}, offset_provider)
    domain_p2 = translate_domain(domain, {"Ioff": 2}, offset_provider)
    domain_p3 = translate_domain(domain, {"Ioff": 3}, offset_provider)

    expected = im.let("inner1", premap_field("outer", "Ioff", 1, domain_p1))(
        im.let("inner2", premap_field("inner1", "Ioff", -1, domain_p2))(
            im.let("inner3", premap_field("inner2", "Ioff", -1, domain_p3))(
                premap_field("inner3", "Ioff", 3, domain)
            )
        )
    )

    expected_domains = {"outer": domain_p2}

    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_nested_let_args(offset_provider):
    testee = im.let(
        "inner",
        im.let("inner_arg", premap_field("outer", "Ioff", 1))(
            premap_field("inner_arg", "Ioff", -1)
        ),
    )(premap_field("inner", "Ioff", -1))

    domain = im.domain(common.GridType.CARTESIAN, {IDim: (2, 11)})
    domain_m1 = translate_domain(domain, {"Ioff": -1}, offset_provider)
    domain_m2 = translate_domain(domain, {"Ioff": -2}, offset_provider)

    expected = im.let(
        "inner",
        im.let("inner_arg", premap_field("outer", "Ioff", 1, domain_m2))(
            premap_field("inner_arg", "Ioff", -1, domain_m1)
        ),
    )(premap_field("inner", "Ioff", -1, domain))

    expected_domains = {"outer": domain_m1}

    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_program_let(offset_provider):
    stencil_tmp = im.lambda_("arg0")(
        im.minus(im.deref(im.shift("Ioff", -1)("arg0")), im.deref("arg0"))
    )
    let_tmp = im.let("inner", premap_field("outer", "Ioff", -1))(premap_field("inner", "Ioff", -1))
    as_fieldop = im.as_fieldop(stencil_tmp)(im.ref("tmp"))

    domain_lm2_rm1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (2, 11)})
    domain_lm1 = im.domain(common.GridType.CARTESIAN, {IDim: (1, 11)})

    params = [im.sym(name) for name in ["in_field", "out_field", "outer"]]

    testee = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_lm1, dtype=float_type)],
        body=[
            itir.SetAt(expr=let_tmp, domain=domain_lm1, target=im.ref("tmp")),
            itir.SetAt(expr=as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    expected_let = im.let("inner", premap_field("outer", "Ioff", -1, domain_lm2_rm1))(
        premap_field("inner", "Ioff", -1, domain_lm1)
    )
    expected_as_fieldop = im.as_fieldop(stencil_tmp, domain)(im.ref("tmp"))

    expected = itir.Program(
        id="forward_diff_with_tmp",
        function_definitions=[],
        params=params,
        declarations=[itir.Temporary(id="tmp", domain=domain_lm1, dtype=float_type)],
        body=[
            itir.SetAt(expr=expected_let, domain=domain_lm1, target=im.ref("tmp")),
            itir.SetAt(expr=expected_as_fieldop, domain=domain, target=im.ref("out_field")),
        ],
    )

    run_test_program(testee, expected, offset_provider)


def test_make_tuple(offset_provider):
    testee = im.make_tuple(im.as_fieldop("deref")("in_field1"), im.as_fieldop("deref")("in_field2"))
    domain1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain2 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 13)})
    expected = im.make_tuple(
        im.as_fieldop("deref", domain1)("in_field1"), im.as_fieldop("deref", domain2)("in_field2")
    )
    expected_domains_dict = {"in_field1": {IDim: (0, 11)}, "in_field2": {IDim: (0, 13)}}
    expected_domains = {
        ref: im.domain(common.GridType.CARTESIAN, d) for ref, d in expected_domains_dict.items()
    }

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        (
            domain_utils.SymbolicDomain.from_expr(domain1),
            domain_utils.SymbolicDomain.from_expr(domain2),
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_tuple_get_1_make_tuple(offset_provider):
    testee = im.tuple_get(1, im.make_tuple(im.ref("a"), im.ref("b"), im.ref("c")))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.tuple_get(1, im.make_tuple(im.ref("a"), im.ref("b"), im.ref("c")))
    expected_domains = {
        "a": infer_domain.DomainAccessDescriptor.NEVER,
        "b": im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)}),
        "c": infer_domain.DomainAccessDescriptor.NEVER,
    }

    actual, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_tuple_get_1_nested_make_tuple(offset_provider):
    testee = im.tuple_get(1, im.make_tuple(im.ref("a"), im.make_tuple(im.ref("b"), im.ref("c"))))
    domain1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain2 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 12)})
    expected = im.tuple_get(1, im.make_tuple(im.ref("a"), im.make_tuple(im.ref("b"), im.ref("c"))))
    expected_domains = {"a": infer_domain.DomainAccessDescriptor.NEVER, "b": domain1, "c": domain2}

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        (
            domain_utils.SymbolicDomain.from_expr(domain1),
            domain_utils.SymbolicDomain.from_expr(domain2),
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_tuple_get_let_arg_make_tuple(offset_provider):
    testee = im.tuple_get(1, im.let("a", im.make_tuple(im.ref("b"), im.ref("c")))("d"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.tuple_get(1, im.let("a", im.make_tuple(im.ref("b"), im.ref("c")))("d"))
    expected_domains = {
        "b": infer_domain.DomainAccessDescriptor.NEVER,
        "c": infer_domain.DomainAccessDescriptor.NEVER,
        "d": (infer_domain.DomainAccessDescriptor.NEVER, domain),
    }

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        domain_utils.SymbolicDomain.from_expr(
            im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_tuple_get_let_make_tuple(offset_provider):
    testee = im.tuple_get(1, im.let("a", "b")(im.make_tuple(im.ref("c"), im.ref("d"))))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.tuple_get(1, im.let("a", "b")(im.make_tuple(im.ref("c"), im.ref("d"))))
    expected_domains = {
        "c": infer_domain.DomainAccessDescriptor.NEVER,
        "d": domain,
        "b": infer_domain.DomainAccessDescriptor.NEVER,
    }

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        domain_utils.SymbolicDomain.from_expr(domain),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_nested_make_tuple(offset_provider):
    testee = im.make_tuple(im.make_tuple(im.ref("a"), im.ref("b")), im.ref("c"))
    domain1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain2_1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 12)})
    domain2_2 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 13)})
    domain3 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 14)})
    expected = im.make_tuple(im.make_tuple(im.ref("a"), im.ref("b")), im.ref("c"))
    expected_domains = {"a": domain1, "b": (domain2_1, domain2_2), "c": domain3}

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        (
            (
                domain_utils.SymbolicDomain.from_expr(domain1),
                (
                    domain_utils.SymbolicDomain.from_expr(domain2_1),
                    domain_utils.SymbolicDomain.from_expr(domain2_2),
                ),
            ),
            domain_utils.SymbolicDomain.from_expr(domain3),
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_tuple_get_1(offset_provider):
    testee = im.tuple_get(1, im.ref("a"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.tuple_get(1, im.ref("a"))
    expected_domains = {"a": (infer_domain.DomainAccessDescriptor.NEVER, domain)}

    actual, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_domain_tuple(offset_provider):
    testee = im.ref("a")
    domain1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain2 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 12)})
    expected = im.ref("a")
    expected_domains = {"a": (domain1, domain2)}

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        (
            domain_utils.SymbolicDomain.from_expr(domain1),
            domain_utils.SymbolicDomain.from_expr(domain2),
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_as_fieldop_tuple_get(offset_provider):
    testee = im.op_as_fieldop(im.plus)(im.tuple_get(0, im.ref("a")), im.tuple_get(1, im.ref("a")))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.op_as_fieldop(im.plus, domain)(
        im.tuple_get(0, im.ref("a")), im.tuple_get(1, im.ref("a"))
    )
    expected_domains = {"a": (domain, domain)}

    actual, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_make_tuple_2tuple_get(offset_provider):
    testee = im.make_tuple(im.tuple_get(0, im.ref("a")), im.tuple_get(1, im.ref("a")))
    domain1 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    domain2 = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.make_tuple(im.tuple_get(0, im.ref("a")), im.tuple_get(1, im.ref("a")))
    expected_domains = {"a": (domain1, domain2)}

    actual, actual_domains = infer_domain.infer_expr(
        testee,
        (
            domain_utils.SymbolicDomain.from_expr(domain1),
            domain_utils.SymbolicDomain.from_expr(domain2),
        ),
        offset_provider=offset_provider,
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_make_tuple_non_tuple_domain(offset_provider):
    testee = im.make_tuple(im.as_fieldop("deref")("in_field1"), im.as_fieldop("deref")("in_field2"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})

    expected = im.make_tuple(
        im.as_fieldop("deref", domain)("in_field1"), im.as_fieldop("deref", domain)("in_field2")
    )
    expected_domains = {"in_field1": domain, "in_field2": domain}

    actual, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )

    assert expected == actual
    assert expected_domains == constant_fold_accessed_domains(actual_domains)


def test_arithmetic_builtin(offset_provider):
    testee = im.plus(im.ref("alpha"), im.ref("beta"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    expected = im.plus(im.ref("alpha"), im.ref("beta"))
    expected_domains = {}

    actual_call, actual_domains = infer_domain.infer_expr(
        testee, domain_utils.SymbolicDomain.from_expr(domain), offset_provider=offset_provider
    )
    folded_call = constant_fold_domain_exprs(actual_call)

    assert folded_call == expected
    assert actual_domains == expected_domains


def test_scan(offset_provider):
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 11)})
    testee = im.as_fieldop(
        im.scan(im.lambda_("init", "it")(im.deref(im.shift("Ioff", 1)("it"))), True, 0.0)
    )("a")
    expected = im.as_fieldop(
        im.scan(im.lambda_("init", "it")(im.deref(im.shift("Ioff", 1)("it"))), True, 0.0),
        domain,
    )("a")

    run_test_expr(
        testee,
        expected,
        domain,
        {"a": im.domain(common.GridType.CARTESIAN, {IDim: (1, 12)})},
        offset_provider,
    )


def test_symbolic_domain_sizes(unstructured_offset_provider):
    stencil = im.lambda_("arg0")(im.deref(im.shift("E2V", 1)("arg0")))
    domain = im.domain(common.GridType.UNSTRUCTURED, {Edge: (0, 1)})
    symbolic_domain_sizes = {"Vertex": "num_vertices"}

    testee, expected = setup_test_as_fieldop(
        stencil,
        domain,
    )
    run_test_expr(
        testee,
        expected,
        domain,
        {"in_field1": {Vertex: (0, im.ref("num_vertices"))}},
        unstructured_offset_provider,
        symbolic_domain_sizes,
    )


def test_unknown_domain(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(im.deref(im.shift("Ioff", im.deref("arg1"))("arg0")))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    expected_domains = {
        "in_field1": infer_domain.DomainAccessDescriptor.UNKNOWN,
        "in_field2": {IDim: (0, 10)},
    }
    testee, expected = setup_test_as_fieldop(stencil, domain)
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_never_accessed_domain(offset_provider):
    stencil = im.lambda_("arg0", "arg1")(im.deref("arg0"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    expected_domains = {
        "in_field1": {IDim: (0, 10)},
        "in_field2": infer_domain.DomainAccessDescriptor.NEVER,
    }
    testee, expected = setup_test_as_fieldop(stencil, domain)
    run_test_expr(testee, expected, domain, expected_domains, offset_provider)


def test_never_accessed_domain_tuple(offset_provider):
    testee = im.tuple_get(0, im.make_tuple("in_field1", "in_field2"))
    domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    expected_domains = {
        "in_field1": {IDim: (0, 10)},
        "in_field2": infer_domain.DomainAccessDescriptor.NEVER,
    }
    run_test_expr(testee, testee, domain, expected_domains, offset_provider)
