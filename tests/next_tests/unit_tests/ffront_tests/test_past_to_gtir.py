# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import re

import pytest

import gt4py.eve as eve
import gt4py.next as gtx
from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next import errors
from gt4py.next.ffront.func_to_past import ProgramParser
from gt4py.next.ffront.past_to_itir import ProgramLowering
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.type_system import type_specifications as ts

from next_tests.past_common_fixtures import (
    IDim,
    copy_program_def,
    copy_restrict_program_def,
    float64,
    identity_def,
    invalid_call_sig_program_def,
)


@pytest.fixture
def gtir_identity_fundef():
    return itir.FunctionDefinition(
        id="identity",
        params=[itir.Sym(id="x")],
        expr=im.as_fieldop("deref")("x"),
    )


def test_copy_lowering(copy_program_def, gtir_identity_fundef):
    past_node = ProgramParser.apply_to_function(copy_program_def)
    itir_node = ProgramLowering.apply(
        past_node,
        function_definitions=[gtir_identity_fundef],
        grid_type=gtx.GridType.CARTESIAN,
    )
    set_at_pattern = P(
        itir.SetAt,
        domain=P(
            itir.FunCall,
            fun=P(itir.SymRef, id=eve.SymbolRef("cartesian_domain")),
            args=[
                P(
                    itir.FunCall,
                    fun=P(itir.SymRef, id=eve.SymbolRef("named_range")),
                    args=[
                        P(itir.AxisLiteral, value="IDim"),
                        P(itir.Literal, value="0", type=ts.ScalarType(kind=ts.ScalarKind.INT32)),
                        P(itir.SymRef, id=eve.SymbolRef("__out_size_0")),
                    ],
                )
            ],
        ),
        expr=P(
            itir.FunCall,
            fun=P(itir.SymRef, id=eve.SymbolRef("identity")),
            args=[P(itir.SymRef, id=eve.SymbolRef("in_field"))],
        ),
        target=P(itir.SymRef, id=eve.SymbolRef("out")),
    )
    program_pattern = P(
        itir.Program,
        id=eve.SymbolName("copy_program"),
        params=[
            P(itir.Sym, id=eve.SymbolName("in_field")),
            P(itir.Sym, id=eve.SymbolName("out")),
            P(itir.Sym, id=eve.SymbolName("__in_field_size_0")),
            P(itir.Sym, id=eve.SymbolName("__out_size_0")),
        ],
        body=[set_at_pattern],
    )

    program_pattern.match(itir_node, raise_exception=True)


def test_copy_restrict_lowering(copy_restrict_program_def, gtir_identity_fundef):
    past_node = ProgramParser.apply_to_function(copy_restrict_program_def)
    itir_node = ProgramLowering.apply(
        past_node,
        function_definitions=[gtir_identity_fundef],
        grid_type=gtx.GridType.CARTESIAN,
    )
    set_at_pattern = P(
        itir.SetAt,
        domain=P(
            itir.FunCall,
            fun=P(itir.SymRef, id=eve.SymbolRef("cartesian_domain")),
            args=[
                P(
                    itir.FunCall,
                    fun=P(itir.SymRef, id=eve.SymbolRef("named_range")),
                    args=[
                        P(itir.AxisLiteral, value="IDim"),
                        P(
                            itir.Literal,
                            value="1",
                            type=ts.ScalarType(
                                kind=getattr(ts.ScalarKind, itir.INTEGER_INDEX_BUILTIN.upper())
                            ),
                        ),
                        P(
                            itir.Literal,
                            value="2",
                            type=ts.ScalarType(
                                kind=getattr(ts.ScalarKind, itir.INTEGER_INDEX_BUILTIN.upper())
                            ),
                        ),
                    ],
                )
            ],
        ),
    )
    program_pattern = P(
        itir.Program,
        id=eve.SymbolName("copy_restrict_program"),
        params=[
            P(itir.Sym, id=eve.SymbolName("in_field")),
            P(itir.Sym, id=eve.SymbolName("out")),
            P(itir.Sym, id=eve.SymbolName("__in_field_size_0")),
            P(itir.Sym, id=eve.SymbolName("__out_size_0")),
        ],
        body=[set_at_pattern],
    )

    program_pattern.match(itir_node, raise_exception=True)


def test_tuple_constructed_in_out_with_slicing(make_tuple_op):
    def tuple_program(
        inp: gtx.Field[[IDim], float64],
        out1: gtx.Field[[IDim], float64],
        out2: gtx.Field[[IDim], float64],
    ):
        make_tuple_op(inp, out=(out1[1:], out2[1:]))

    parsed = ProgramParser.apply_to_function(tuple_program)
    ProgramLowering.apply(parsed, function_definitions=[], grid_type=gtx.GridType.CARTESIAN)


@pytest.mark.xfail(
    reason="slicing is only allowed if all fields are sliced in the same way."
)  # see ADR 10
def test_tuple_constructed_in_out_with_slicing(make_tuple_op):
    def tuple_program(
        inp: gtx.Field[[IDim], float64],
        out1: gtx.Field[[IDim], float64],
        out2: gtx.Field[[IDim], float64],
    ):
        make_tuple_op(inp, out=(out1[1:], out2))

    parsed = ProgramParser.apply_to_function(tuple_program)
    ProgramLowering.apply(parsed, function_definitions=[], grid_type=gtx.GridType.CARTESIAN)


@pytest.mark.xfail
def test_inout_prohibited(identity_def):
    identity = gtx.field_operator(identity_def)

    def inout_field_program(inout_field: gtx.Field[[IDim], "float64"]):
        identity(inout_field, out=inout_field)

    with pytest.raises(
        ValueError, match=(r"Call to function with field as input and output not allowed.")
    ):
        ProgramLowering.apply(
            ProgramParser.apply_to_function(inout_field_program),
            function_definitions=[],
            grid_type=gtx.GridType.CARTESIAN,
        )


def test_invalid_call_sig_program(invalid_call_sig_program_def):
    with pytest.raises(errors.DSLError) as exc_info:
        ProgramLowering.apply(
            ProgramParser.apply_to_function(invalid_call_sig_program_def),
            function_definitions=[],
            grid_type=gtx.GridType.CARTESIAN,
        )

    assert exc_info.match("Invalid call to 'identity'")
    # TODO(tehrengruber): re-enable again when call signature check doesn't return
    #  immediately after missing `out` argument
    # assert (
    #    re.search(
    #        "Function takes 1 arguments, but 2 were given.", exc_info.value.__cause__.args[0]
    #    )
    #    is not None
    # )
    assert (
        re.search(r"Missing required keyword argument 'out'", exc_info.value.__cause__.args[0])
        is not None
    )
