# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import pytest

from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import pass_manager
from gt4py.next.iterator.transforms import scan_fusion
from gt4py.next.type_system import type_specifications as ts


def make_shadowed_program(scope):
    k = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
    field_type = ts.FieldType(dims=[k], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64))
    domain = im.domain("cartesian_domain", {k: (0, 5)})
    scan = im.scan(
        im.lambda_("carry", "a", "b")(im.plus("carry", im.plus(im.deref("a"), im.deref("b")))),
        True,
        0.0,
    )
    wrapper = ir.FunctionDefinition(
        id="scan_wrapper",
        params=[im.sym("a"), im.sym("b")],
        expr=im.as_fieldop(scan)("a", "b"),
    )
    call = im.call("scan_wrapper")("gamma", "gamma")
    if scope == "nested_let":
        body = im.let("gamma", "values")(im.let("values", "wind")(call))
    elif scope == "lambda":
        # This lambda is not applied directly, so the lambda visitor handles it.
        choose = im.if_(True, im.lambda_("values")(call), im.lambda_("values")(call))
        body = im.let("gamma", "values")(im.call(choose)("wind"))
    elif scope == "parallel_let":
        body = im.let(("gamma", "values"), ("values", "wind"))(call)
    else:
        # A binder inside the replacement destination must not capture its names either.
        argument = im.let("values", "wind")("gamma")
        body = im.let("gamma", "values")(im.call("scan_wrapper")(argument, argument))
    stage = ir.FunctionDefinition(
        id="coefficient_stage", params=[im.sym("values"), im.sym("wind")], expr=body
    )
    program = ir.Program(
        id="shadowed_scan_input",
        function_definitions=[wrapper, stage],
        declarations=[],
        params=[im.sym(name, field_type) for name in ("x", "w", "out")],
        body=[
            ir.SetAt(
                expr=im.call("coefficient_stage")("x", "w"), domain=domain, target=im.ref("out")
            )
        ],
    )
    return program


@pytest.mark.parametrize("scope", ["nested_let", "lambda", "parallel_let", "argument_lambda"])
def test_producer_substitution_preserves_variable_scope(scope):
    program = make_shadowed_program(scope)
    before = copy.deepcopy(program)
    prepared = scan_fusion.normalize_scan_producers(program, offset_provider_type={})
    # Resolve lexical scopes independently through the standard pipeline. All four
    # inputs must still read x, not w, after preparation.
    expected = pass_manager.apply_fieldview_transforms(before, offset_provider={})
    actual = pass_manager.apply_fieldview_transforms(prepared, offset_provider={})
    assert actual == expected
    assert program == before
