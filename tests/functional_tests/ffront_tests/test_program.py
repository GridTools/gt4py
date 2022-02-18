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
import numpy as np
import pytest

from eve import pattern_matching
from functional.common import Field, GTTypeError
from functional.ffront import common_types
from functional.ffront import program_ast as past
from functional.ffront.decorator import field_operator, program
from functional.ffront.func_to_past import ProgramParser, ProgramSyntaxError
from functional.ffront.past_to_itir import ProgramLowering
from functional.iterator import ir as itir
from functional.iterator.embedded import index_field, np_as_located_field
from functional.iterator.runtime import CartesianAxis, offset


float64 = float
IDim = CartesianAxis("IDim")
Ioff = offset("Ioff")

past_ = pattern_matching.ModuleWrapper(past)
itir_ = pattern_matching.ModuleWrapper(itir)


@field_operator
def identity(in_field: Field[[IDim], "float64"]) -> Field[[IDim], "float64"]:
    return in_field


@program
def copy_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
    identity(in_field, out=out_field)


@program
def double_copy_program(
    in_field: Field[[IDim], "float64"],
    intermediate_field: Field[[IDim], "float64"],
    out_field: Field[[IDim], "float64"],
):
    identity(in_field, out=intermediate_field)
    identity(intermediate_field, out=out_field)


@program
def copy_restrict_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
    identity(in_field, out=out_field[1:2])


def invalid_call_sig_program(
    in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]
):
    identity(in_field, in_field, out=out_field)


def invalid_slice_program(in_field: Field[[IDim], "float64"], out_field: Field[[IDim], "float64"]):
    identity(in_field, out=out_field[1:2, 3:4])


def test_copy_parsing():
    past_node = ProgramParser.apply_to_function(copy_program.definition)

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )
    pattern_node = past_.Program(
        id="copy_program",
        params=[
            past_.Symbol(id="in_field", type=field_type),
            past_.Symbol(id="out_field", type=field_type),
        ],
        body=[
            past_.Call(
                func=past_.Name(id="identity"),
                args=[past_.Name(id="in_field")],
                kwargs={"out": past_.Name(id="out_field")},
            )
        ],
        location=past_.SourceLocation(line=43, source=__file__),
    )
    assert pattern_node.matches(past_node, raise_=True)


def test_double_copy_parsing():
    past_node = ProgramParser.apply_to_function(double_copy_program.definition)

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )
    pattern_node = past_.Program(
        id="double_copy_program",
        params=[
            past_.Symbol(id="in_field", type=field_type),
            past_.Symbol(id="intermediate_field", type=field_type),
            past_.Symbol(id="out_field", type=field_type),
        ],
        body=[
            past_.Call(
                func=past_.Name(id="identity"),
                args=[past_.Name(id="in_field")],
                kwargs={"out": past_.Name(id="intermediate_field")},
            ),
            past_.Call(
                func=past_.Name(id="identity"),
                args=[past_.Name(id="intermediate_field")],
                kwargs={"out": past_.Name(id="out_field")},
            ),
        ],
    )
    assert pattern_node.matches(past_node, raise_=True)


def test_undefined_field_program():
    def undefined_field_program(in_field: Field[[IDim], "float64"]):
        identity(in_field, out=out_field)

    with pytest.raises(
        ProgramSyntaxError,
        match=(r"Invalid Program Syntax: Missing symbol definitions: {'out_field'}"),
    ):
        ProgramParser.apply_to_function(undefined_field_program)


@pytest.mark.xfail
def test_inout_prohibited():
    def inout_field_program(inout_field: Field[[IDim], "float64"]):
        identity(inout_field, out=inout_field)

    with pytest.raises(
        GTTypeError,
        match=(r"Call to function with field as input and output not allowed."),
    ):
        ProgramLowering.apply(ProgramParser.apply_to_function(inout_field_program))


def invalid_call_sig_program():
    with pytest.raises(
        GTTypeError,
    ) as exc_info:
        ProgramLowering.apply(ProgramParser.apply_to_function(invalid_call_sig_program))

    assert exc_info.match("Invalid call to `identity`")
    assert exc_info.match("Function takes 1 arguments, but 2 were given.")
    assert exc_info.match("Missing required keyword argument\(s\) `out`")


def test_copy_restrict_parsing():
    past_node = ProgramParser.apply_to_function(copy_restrict_program.definition)

    field_type = common_types.FieldType(
        dims=[IDim],
        dtype=common_types.ScalarType(kind=common_types.ScalarKind.FLOAT64, shape=None),
    )
    slice_pattern_node = past_.Slice(lower=past_.Constant(value=1), upper=past_.Constant(value=2))
    pattern_node = past_.Program(
        id="copy_restrict_program",
        params=[
            past_.Symbol(id="in_field", type=field_type),
            past_.Symbol(id="out_field", type=field_type),
        ],
        body=[
            past_.Call(
                func=past_.Name(id="identity"),
                args=[past_.Name(id="in_field")],
                kwargs={
                    "out": past_.Subscript(
                        value=past_.Name(id="out_field"), slice_=slice_pattern_node
                    )
                },
            )
        ],
    )

    pattern_node.matches(past_node, raise_=True)


def test_copy_lowering():
    past_node = ProgramParser.apply_to_function(copy_restrict_program.definition)
    itir_node = ProgramLowering.apply(past_node)
    closure_pattern = itir_.StencilClosure(
        domain=itir_.FunCall(
            fun=itir_.SymRef(id="domain"),
            args=[
                itir_.FunCall(
                    fun=itir_.SymRef(id="named_range"),
                    args=[
                        itir_.AxisLiteral(value="IDim"),
                        itir_.IntLiteral(value=1),
                        itir_.IntLiteral(value=2),
                    ],
                )
            ],
        )
    )
    fencil_pattern = itir_.FencilDefinition(
        id="copy_restrict_program",
        params=[
            itir_.Sym(id="in_field"),
            itir_.Sym(id="out_field"),
            itir_.Sym(id="__in_field_size_0"),
            itir_.Sym(id="__out_field_size_0"),
        ],
        closures=[closure_pattern],
    )

    fencil_pattern.matches(itir_node, raise_=True)


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


def test_copy_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))

    copy_program(in_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_double_copy_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    intermediate_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))

    double_copy_program(in_field, intermediate_field, out_field, offset_provider={})

    assert np.allclose(in_field, out_field)


def test_copy_restricted_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))
    out_field_ref = np_as_located_field(IDim)(
        np.array([1 if i in range(1, 2) else 0 for i in range(0, size)])
    )

    copy_restrict_program(in_field, out_field, offset_provider={})

    assert np.allclose(out_field_ref, out_field)


def test_identity_fo_execution():
    size = 10
    in_field = np_as_located_field(IDim)(np.ones((size)))
    out_field = np_as_located_field(IDim)(np.zeros((size)))

    identity(in_field, out=out_field, offset_provider={})

    assert np.allclose(in_field, out_field)
