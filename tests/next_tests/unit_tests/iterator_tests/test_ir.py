# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.iterator import ir
from gt4py.next.otf import arguments, definitions, stages
from gt4py.next.type_system import type_specifications as ts
from gt4py import eve


def test_noninstantiable():
    with pytest.raises(TypeError, match="non-instantiable"):
        ir.Node()
    with pytest.raises(TypeError, match="non-instantiable"):
        ir.Expr()


def test_str():
    testee = ir.Lambda(params=[ir.Sym(id="x")], expr=ir.SymRef(id="x"))
    expected = "λ(x) → x"
    actual = str(testee)
    assert actual == expected


def test_fingerprint():
    loc1 = eve.SourceLocation(filename="loc1", line=1, column=1)
    loc2 = eve.SourceLocation(filename="loc2", line=1, column=1)
    node1 = ir.SymRef(id="abc", location=loc1)
    node2 = ir.SymRef(id="abc", location=loc2)
    node3 = ir.SymRef(id="abcd", location=loc1)
    assert node1.fingerprint() == node2.fingerprint()
    assert node1.fingerprint() != node3.fingerprint()


def test_fingerprint_nested():
    def node_maker(fun: str, filename: str):
        loc = eve.SourceLocation(filename=filename, line=1, column=1)
        return ir.FunCall(
            fun=ir.SymRef(id=fun, location=loc),
            args=[ir.SymRef(id="arg", location=loc)],
            location=loc,
        )

    node1 = node_maker("f1", "loc1")
    node2 = node_maker("f1", "loc2")
    node3 = node_maker("f3", "loc1")
    assert node1.fingerprint() == node2.fingerprint()
    assert node1.fingerprint() != node3.fingerprint()


def test_different_precisions():
    program = ir.Program(
        id="test_program",
        function_definitions=[],
        params=[ir.Sym(id="arg")],
        declarations=[],
        body=[],
    )

    compilable_single = definitions.CompilableProgramDef(
        data=program,
        args=arguments.CompileTimeArgs(
            args=(ts.ScalarType(kind=ts.ScalarKind.FLOAT32),),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={},
        ),
    )
    compilable_double = definitions.CompilableProgramDef(
        data=program,
        args=arguments.CompileTimeArgs(
            args=(ts.ScalarType(kind=ts.ScalarKind.FLOAT64),),
            kwargs={},
            offset_provider={},
            column_axis=None,
            argument_descriptor_contexts={},
        ),
    )

    hash_single = stages.fingerprint_compilable_program(compilable_single)
    hash_double = stages.fingerprint_compilable_program(compilable_double)

    assert hash_single != hash_double
