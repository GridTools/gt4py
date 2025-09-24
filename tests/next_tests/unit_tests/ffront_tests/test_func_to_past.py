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
from gt4py.next import errors, float64
from gt4py.next.ffront import program_ast as past
from gt4py.next.ffront.func_to_past import ProgramParser
from gt4py.next.type_system import type_specifications as ts

from next_tests.past_common_fixtures import (
    IDim,
    copy_program_def,
    copy_restrict_program_def,
    double_copy_program_def,
    identity_def,
    make_tuple_op,
)


def test_tuple_constructed_in_out(make_tuple_op):
    def tuple_program(
        inp: gtx.Field[[IDim], float64],
        out1: gtx.Field[[IDim], float64],
        out2: gtx.Field[[IDim], float64],
    ):
        make_tuple_op(inp, out=(out1, out2))

    _ = ProgramParser.apply_to_function(tuple_program)


def test_copy_parsing(copy_program_def):
    past_node = ProgramParser.apply_to_function(copy_program_def)

    field_type = ts.FieldType(
        dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("copy_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out"), type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("in_field"))],
                kwargs={"out": P(past.Name, id=past.SymbolRef("out"))},
            )
        ],
        location=P(past.SourceLocation),
    )
    assert pattern_node.match(past_node, raise_exception=True)


def test_double_copy_parsing(double_copy_program_def):
    past_node = ProgramParser.apply_to_function(double_copy_program_def)

    field_type = ts.FieldType(
        dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("double_copy_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("intermediate_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out"), type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("in_field"))],
                kwargs={"out": P(past.Name, id=past.SymbolRef("intermediate_field"))},
            ),
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("intermediate_field"))],
                kwargs={"out": P(past.Name, id=past.SymbolRef("out"))},
            ),
        ],
    )
    assert pattern_node.match(past_node, raise_exception=True)


def test_undefined_field_program(identity_def):
    identity = gtx.field_operator(identity_def)

    def undefined_field_program(in_field: gtx.Field[[IDim], "float64"]):
        identity(in_field, out=out_field)  # noqa: F821 [undefined-name]

    with pytest.raises(errors.DSLError, match=(r"Undeclared or untyped symbol 'out_field'.")):
        ProgramParser.apply_to_function(undefined_field_program)


def test_copy_restrict_parsing(copy_restrict_program_def):
    past_node = ProgramParser.apply_to_function(copy_restrict_program_def)

    field_type = ts.FieldType(
        dims=[IDim], dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None)
    )
    slice_pattern_node = P(
        past.Slice, lower=P(past.Constant, value=1), upper=P(past.Constant, value=2)
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("copy_restrict_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out"), type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("in_field"))],
                kwargs={
                    "out": P(
                        past.Subscript,
                        value=P(past.Name, id=past.SymbolRef("out")),
                        slice_=slice_pattern_node,
                    )
                },
            )
        ],
    )

    pattern_node.match(past_node, raise_exception=True)


def test_domain_exception_1(identity_def):
    domain_format_1 = gtx.field_operator(identity_def)

    def domain_format_1_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_1(in_field, out=in_field, domain=(0, 2))

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_1_program)

    assert exc_info.match("Invalid call to 'domain_format_1'")

    assert (
        re.search("Tuple domain requires tuple output", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_2(identity_def):
    domain_format_2 = gtx.field_operator(identity_def)

    def domain_format_2_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_2(in_field, out=in_field, domain={IDim: (0, 1, 2)})

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_2_program)

    assert exc_info.match("Invalid call to 'domain_format_2'")

    assert (
        re.search("Only 2 values allowed in domain range", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_3(identity_def):
    domain_format_3 = gtx.field_operator(identity_def)

    def domain_format_3_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_3(in_field, domain={IDim: (0, 2)})

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_3_program)

    assert exc_info.match("Invalid call to 'domain_format_3'")

    assert (
        re.search(r"Missing required keyword argument\ 'out'", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_4(identity_def):
    domain_format_4 = gtx.field_operator(identity_def)

    def domain_format_4_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_4(
            in_field, out=(in_field[0:1], (in_field[0:1], in_field[0:1])), domain={IDim: (0, 1)}
        )

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_4_program)

    assert exc_info.match("Invalid call to 'domain_format_4'")

    assert (
        re.search("Either only domain or slicing allowed", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_5(identity_def):
    domain_format_5 = gtx.field_operator(identity_def)

    def domain_format_5_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_5(in_field, out=in_field, domain={IDim: ("1.0", 9.0)})

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_5_program)

    assert exc_info.match("Invalid call to 'domain_format_5'")

    assert (
        re.search("Only integer values allowed in domain range", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_6(identity_def):
    domain_format_6 = gtx.field_operator(identity_def)

    def domain_format_6_program(in_field: gtx.Field[[IDim], float64]):
        domain_format_6(in_field, out=in_field, domain={})

    with pytest.raises(errors.DSLError) as exc_info:
        ProgramParser.apply_to_function(domain_format_6_program)

    assert exc_info.match("Invalid call to 'domain_format_6'")

    assert re.search("Empty domain not allowed.", exc_info.value.__cause__.args[0]) is not None
