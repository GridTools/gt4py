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
# TODO(tehrengruber): All field operators and programs should be executable
#  as is at some point. Adopt tests to also run on the regular python objects.
import re

import numpy as np
import pytest

from eve.pattern_matching import ObjectPattern as P
from functional.common import Field, GTTypeError
from functional.ffront import common_types
from functional.ffront import program_ast as past
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import FieldOffset
from functional.ffront.func_to_past import ProgramParser
from functional.ffront.past_passes.type_deduction import ProgramTypeError
from functional.ffront.past_to_itir import ProgramLowering
from functional.iterator import ir as itir
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis


float64 = float
IDim = CartesianAxis("IDim")
Ioff = FieldOffset("Ioff", source=IDim, target=[IDim])


# TODO(tehrengruber): Improve test structure. Identity needs to be decorated
#  in order to be used inside a program. This is unfortunate as a bug inside
#  the decorator may result in failing tests before the actual test is run.
#  A better way would be to first test everything field operator related,
#  including the decorator and then continue with program tests that then
#  can safely use the field operator decorator inside the fixtures.
@pytest.fixture
def identity_def():
    def identity(in_field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return in_field

    return identity


@pytest.fixture
def copy_program_def(identity_def):
    identity = field_operator(identity_def)

    def copy_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
        identity(in_field, out=out_field)

    return copy_program


@pytest.fixture
def double_copy_program_def(identity_def):
    identity = field_operator(identity_def)

    def double_copy_program(
        in_field: Field[[IDim], "float64"],
        intermediate_field: Field[[IDim], "float64"],
        out_field: Field[[IDim], "float64"],
    ):
        identity(in_field, out=intermediate_field)
        identity(intermediate_field, out=out_field)

    return double_copy_program


@pytest.fixture
def copy_restrict_program_def(identity_def):
    identity = field_operator(identity_def)

    def copy_restrict_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out=out_field[1:2])

    return copy_restrict_program


@pytest.fixture
def invalid_call_sig_program_def(identity_def):
    identity = field_operator(identity_def)

    def invalid_call_sig_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out_field)

    return invalid_call_sig_program


@pytest.fixture
def invalid_out_slice_dims_program_def(identity_def):
    identity = field_operator(identity_def)

    def invalid_out_slice_dims_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        identity(in_field, out=out_field[1:2, 3:4])

    return invalid_out_slice_dims_program


@pytest.fixture
def itir_identity_fundef():
    return itir.FunctionDefinition(
        id="identity",
        params=[itir.Sym(id="x")],
        expr=itir.FunCall(fun=itir.SymRef(id="deref"), args=[itir.SymRef(id="x")]),
    )


def test_identity_fo_execution(identity_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    identity = field_operator(identity_def)

    identity(in_field, out=out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_copy_parsing(copy_program_def):
    past_node = ProgramParser.apply_to_function(copy_program_def)

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )
    pattern_node = P(
        past.Program,
        id="copy_program",
        params=[
            P(past.Symbol, id="in_field", type=field_type),
            P(past.Symbol, id="out_field", type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id="identity"),
                args=[P(past.Name, id="in_field")],
                kwargs={"out": P(past.Name, id="out_field")},
            )
        ],
        location=P(past.SourceLocation, line=58, source=__file__),
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
        id="double_copy_program",
        params=[
            P(past.Symbol, id="in_field", type=field_type),
            P(past.Symbol, id="intermediate_field", type=field_type),
            P(past.Symbol, id="out_field", type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id="identity"),
                args=[P(past.Name, id="in_field")],
                kwargs={"out": P(past.Name, id="intermediate_field")},
            ),
            P(
                past.Call,
                func=P(past.Name, id="identity"),
                args=[P(past.Name, id="intermediate_field")],
                kwargs={"out": P(past.Name, id="out_field")},
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


@pytest.mark.xfail
def test_inout_prohibited(identity_def):
    identity = field_operator(identity_def)

    def inout_field_program(inout_field: Field[[IDim], "float64"]):
        identity(inout_field, out=inout_field)

    with pytest.raises(
        GTTypeError,
        match=(r"Call to function with field as input and output not allowed."),
    ):
        ProgramLowering.apply(ProgramParser.apply_to_function(inout_field_program))


def test_invalid_call_sig_program(invalid_call_sig_program_def):
    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramLowering.apply(ProgramParser.apply_to_function(invalid_call_sig_program_def))

    assert exc_info.match("Invalid call to `identity`")
    # TODO(tehrengruber): find a better way to test this
    assert (
        re.search(
            "Function takes 1 arguments, but 2 were given.", exc_info.value.__context__.args[0]
        )
        is not None
    )
    assert (
        re.search(
            "Missing required keyword argument\(s\) `out`", exc_info.value.__context__.args[0]
        )
        is not None
    )


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
        id="copy_restrict_program",
        params=[
            P(past.Symbol, id="in_field", type=field_type),
            P(past.Symbol, id="out_field", type=field_type),
        ],
        body=[
            P(
                past.Call,
                func=P(past.Name, id="identity"),
                args=[P(past.Name, id="in_field")],
                kwargs={
                    "out": P(
                        past.Subscript,
                        value=P(past.Name, id="out_field"),
                        slice_=slice_pattern_node,
                    )
                },
            )
        ],
    )

    pattern_node.match(past_node, raise_exception=True)


def test_copy_lowering(copy_program_def, itir_identity_fundef):
    past_node = ProgramParser.apply_to_function(copy_program_def)
    itir_node = ProgramLowering.apply(past_node, function_definitions=[itir_identity_fundef])
    closure_pattern = P(
        itir.StencilClosure,
        domain=P(
            itir.FunCall,
            fun=P(itir.SymRef, id="domain"),
            args=[
                P(
                    itir.FunCall,
                    fun=P(itir.SymRef, id="named_range"),
                    args=[
                        P(itir.AxisLiteral, value="IDim"),
                        P(itir.IntLiteral, value=0),
                        P(itir.SymRef, id="__out_field_size_0"),
                    ],
                )
            ],
        ),
        stencil=P(itir.SymRef, id="identity"),
        inputs=[P(itir.SymRef, id="in_field")],
        output=P(itir.SymRef, id="out_field"),
    )
    fencil_pattern = P(
        itir.FencilDefinition,
        id="copy_program",
        params=[
            P(itir.Sym, id="in_field"),
            P(itir.Sym, id="out_field"),
            P(itir.Sym, id="__in_field_size_0"),
            P(itir.Sym, id="__out_field_size_0"),
        ],
        closures=[closure_pattern],
    )

    fencil_pattern.match(itir_node, raise_exception=True)


def test_copy_restrict_lowering(copy_restrict_program_def, itir_identity_fundef):
    past_node = ProgramParser.apply_to_function(copy_restrict_program_def)
    itir_node = ProgramLowering.apply(past_node, function_definitions=[itir_identity_fundef])
    closure_pattern = P(
        itir.StencilClosure,
        domain=P(
            itir.FunCall,
            fun=P(itir.SymRef, id="domain"),
            args=[
                P(
                    itir.FunCall,
                    fun=P(itir.SymRef, id="named_range"),
                    args=[
                        P(itir.AxisLiteral, value="IDim"),
                        P(itir.IntLiteral, value=1),
                        P(itir.IntLiteral, value=2),
                    ],
                )
            ],
        ),
    )
    fencil_pattern = P(
        itir.FencilDefinition,
        id="copy_restrict_program",
        params=[
            P(itir.Sym, id="in_field"),
            P(itir.Sym, id="out_field"),
            P(itir.Sym, id="__in_field_size_0"),
            P(itir.Sym, id="__out_field_size_0"),
        ],
        closures=[closure_pattern],
    )

    fencil_pattern.match(itir_node, raise_exception=True)


def test_shift_by_one_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.arange(0, size, 1))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(
        np.array([i + 1 if i in range(0, size - 1) else 0 for i in range(0, size)])
    )

    @field_operator
    def shift_by_one(in_field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return in_field(Ioff[1])

    # direct call to field operator
    # TODO(tehrengruber): slicing located fields not supported currently
    # shift_by_one(in_field, out=out_field[:-1], offset_provider={"Ioff": IDim})

    @program
    def shift_by_one_program(
        in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
    ):
        shift_by_one(in_field, out=out_field[:-1])

    shift_by_one_program(in_field, out_field, offset_provider={"Ioff": IDim})

    assert np.allclose(out_field, out_field_ref)


def test_copy_execution(copy_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    copy_program = program(copy_program_def)

    copy_program(in_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_double_copy_execution(double_copy_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    intermediate_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    double_copy_program = program(double_copy_program_def)

    double_copy_program(in_field, intermediate_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_copy_restricted_execution(copy_restrict_program_def):
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(
        np.array([1 if i in range(1, 2) else 0 for i in range(0, size)])
    )
    copy_restrict_program = program(copy_restrict_program_def)

    copy_restrict_program(in_field, out_field, offset_provider={})

    assert np.allclose(out_field_ref, out_field)


def test_calling_fo_from_fo_execution(identity_def):
    size = 10
    in_field = np_as_located_field(IDim)(2 * np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(2 * 2 * 2 * np.ones((size)))

    @field_operator
    def pow_two(field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return field * field

    @field_operator
    def pow_three(field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
        return field * pow_two(field)

    @program
    def fo_from_fo_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
        pow_three(in_field, out=out_field)

    fo_from_fo_program(in_field, out_field, offset_provider={})

    assert np.allclose(out_field, out_field_ref)
