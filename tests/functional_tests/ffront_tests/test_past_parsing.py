# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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
from typing import Tuple

import pytest

import eve
from eve.pattern_matching import ObjectPattern as P
from functional.common import Field, GTTypeError
from functional.ffront import common_types, program_ast as past
from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import Dimension, float64
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeError

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

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
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

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
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

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
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
