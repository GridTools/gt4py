# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

import re

import pytest

import gt4py.eve as eve
from gt4py.eve.pattern_matching import ObjectPattern as P
from gt4py.next.common import Field, GTTypeError
from gt4py.next.ffront import program_ast as past, type_specifications as ts
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import float64
from gt4py.next.ffront.func_to_past import ProgramParser
from gt4py.next.ffront.past_passes.type_deduction import ProgramTypeError
from gt4py.next.type_system import type_specifications as ts

from .past_common_fixtures import (
    IDim,
    copy_program_def,
    copy_restrict_program_def,
    double_copy_program_def,
    float64,
    identity_def,
    invalid_call_sig_program_def,
    make_tuple_op,
)


def test_tuple_constructed_in_out(make_tuple_op):
    def tuple_program(
        inp: Field[[IDim], float64], out1: Field[[IDim], float64], out2: Field[[IDim], float64]
    ):
        make_tuple_op(inp, out=(out1, out2))

    _ = ProgramParser.apply_to_function(tuple_program)


def test_copy_parsing(copy_program_def):
    past_node = ProgramParser.apply_to_function(copy_program_def)

    field_type = ts.FieldType(
        dims=[IDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("copy_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out_field"), type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("in_field"))],
                kwargs={"out": P(past.Name, id=past.SymbolRef("out_field"))},
            )
        ],
        location=P(past.SourceLocation),
    )
    assert pattern_node.match(past_node, raise_exception=True)


def test_double_copy_parsing(double_copy_program_def):
    past_node = ProgramParser.apply_to_function(double_copy_program_def)

    field_type = ts.FieldType(
        dims=[IDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("double_copy_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("intermediate_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out_field"), type=field_type),
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
                kwargs={"out": P(past.Name, id=past.SymbolRef("out_field"))},
            ),
        ],
    )
    assert pattern_node.match(past_node, raise_exception=True)


def test_undefined_field_program(identity_def):
    identity = field_operator(identity_def)

    def undefined_field_program(in_field: Field[[IDim], "float64"]):
        identity(in_field, out=out_field)

    with pytest.raises(
        ProgramTypeError,
        match=(r"Undeclared or untyped symbol `out_field`."),
    ):
        ProgramParser.apply_to_function(undefined_field_program)


def test_copy_restrict_parsing(copy_restrict_program_def):
    past_node = ProgramParser.apply_to_function(copy_restrict_program_def)

    field_type = ts.FieldType(
        dims=[IDim],
        dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT64, shape=None),
    )
    slice_pattern_node = P(
        past.Slice, lower=P(past.Constant, value=1), upper=P(past.Constant, value=2)
    )
    pattern_node = P(
        past.Program,
        id=eve.SymbolName("copy_restrict_program"),
        params=[
            P(past.Symbol, id=eve.SymbolName("in_field"), type=field_type),
            P(past.Symbol, id=eve.SymbolName("out_field"), type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id=past.SymbolRef("identity")),
                args=[P(past.Name, id=past.SymbolRef("in_field"))],
                kwargs={
                    "out": P(
                        past.Subscript,
                        value=P(past.Name, id=past.SymbolRef("out_field")),
                        slice_=slice_pattern_node,
                    )
                },
            )
        ],
    )

    pattern_node.match(past_node, raise_exception=True)


def test_domain_exception_1(identity_def):
    domain_format_1 = field_operator(identity_def)

    def domain_format_1_program(in_field: Field[[IDim], float64]):
        domain_format_1(in_field, out=in_field, domain=(0, 2))

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_1_program)

    assert exc_info.match("Invalid call to `domain_format_1`")

    assert (
        re.search("Only Dictionaries allowed in domain", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_2(identity_def):
    domain_format_2 = field_operator(identity_def)

    def domain_format_2_program(in_field: Field[[IDim], float64]):
        domain_format_2(in_field, out=in_field, domain={IDim: (0, 1, 2)})

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_2_program)

    assert exc_info.match("Invalid call to `domain_format_2`")

    assert (
        re.search("Only 2 values allowed in domain range", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_3(identity_def):
    domain_format_3 = field_operator(identity_def)

    def domain_format_3_program(in_field: Field[[IDim], float64]):
        domain_format_3(in_field, domain={IDim: (0, 2)})

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_3_program)

    assert exc_info.match("Invalid call to `domain_format_3`")

    assert (
        re.search("Missing required keyword argument\(s\) `out`.", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_4(identity_def):
    domain_format_4 = field_operator(identity_def)

    def domain_format_4_program(in_field: Field[[IDim], float64]):
        domain_format_4(
            in_field, out=(in_field[0:1], (in_field[0:1], in_field[0:1])), domain={IDim: (0, 1)}
        )

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_4_program)

    assert exc_info.match("Invalid call to `domain_format_4`")

    assert (
        re.search("Either only domain or slicing allowed", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_5(identity_def):
    domain_format_5 = field_operator(identity_def)

    def domain_format_5_program(in_field: Field[[IDim], float64]):
        domain_format_5(in_field, out=in_field, domain={IDim: ("1.0", 9.0)})

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_5_program)

    assert exc_info.match("Invalid call to `domain_format_5`")

    assert (
        re.search("Only integer values allowed in domain range", exc_info.value.__cause__.args[0])
        is not None
    )


def test_domain_exception_6(identity_def):
    domain_format_6 = field_operator(identity_def)

    def domain_format_6_program(in_field: Field[[IDim], float64]):
        domain_format_6(in_field, out=in_field, domain={})

    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramParser.apply_to_function(domain_format_6_program)

    assert exc_info.match("Invalid call to `domain_format_6`")

    assert re.search("Empty domain not allowed.", exc_info.value.__cause__.args[0]) is not None
