# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
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

from typing import Dict

from gtc import common
from gtc.common import DataType
from gtc.gtir import FieldDecl, Literal, ParAssignStmt, Stencil
from gtc.passes.gtir_dtype_resolver import _GTIRPropagateDtypeToAccess, resolve_dtype

from .gtir_utils import FieldAccessBuilder, StencilBuilder, VerticalLoopBuilder


A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_propagate_dtype_to_FieldAccess():
    name = "foo"
    decl = FieldDecl(name=name, dtype=A_ARITHMETIC_TYPE)

    testee = FieldAccessBuilder(name).build()

    result = _GTIRPropagateDtypeToAccess().visit(testee, symtable={name: decl})

    assert result.dtype == A_ARITHMETIC_TYPE


def get_nodes_with_name(stencil: Stencil, name: str):
    return stencil.iter_tree().if_hasattr("name").filter(lambda node: node.name == name).to_list()


def resolve_dtype_and_validate(testee: Stencil, expected_dtypes: Dict[str, common.DataType]):
    # ensure consistency (input is not already fully resolved)
    for name, _dtype in expected_dtypes.items():
        nodes = get_nodes_with_name(testee, name)
        assert len(nodes) > 0
        assert any([node.dtype is None for node in nodes])

    result: Stencil = resolve_dtype(testee)

    for name, dtype in expected_dtypes.items():
        nodes = get_nodes_with_name(result, name)
        assert len(nodes) > 0
        assert all([node.dtype == dtype for node in nodes])


def test_resolve_dtype_to_FieldAccess():
    testee = (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_par_assign_stmt(
            ParAssignStmt(
                left=FieldAccessBuilder("field").dtype(None).build(),
                right=FieldAccessBuilder("field").dtype(None).build(),
            )
        )
        .build()
    )
    resolve_dtype_and_validate(
        testee,
        {"field": A_ARITHMETIC_TYPE},
    )


def test_resolve_AUTO_from_literal_to_temporary():
    testee = (
        StencilBuilder()
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp").dtype(None).build(),
                    right=Literal(value="0", dtype=A_ARITHMETIC_TYPE),
                )
            )
            .build()
        )
        .build()
    )
    resolve_dtype_and_validate(testee, {"tmp": A_ARITHMETIC_TYPE})


def test_resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary():
    testee = (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp").dtype(None).build(),
                    right=FieldAccessBuilder("field").dtype(None).build(),
                )
            )
            .build()
        )
        .build()
    )
    resolve_dtype_and_validate(testee, {"field": A_ARITHMETIC_TYPE, "tmp": A_ARITHMETIC_TYPE})


def test_resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary_to_FieldAccess_to_temporary():
    testee = (
        StencilBuilder()
        .add_param(FieldDecl(name="field", dtype=A_ARITHMETIC_TYPE))
        .add_vertical_loop(
            VerticalLoopBuilder()
            .add_temporary("tmp1", DataType.AUTO)
            .add_temporary("tmp2", DataType.AUTO)
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp1").dtype(None).build(),
                    right=FieldAccessBuilder("field").dtype(None).build(),
                )
            )
            .add_stmt(
                ParAssignStmt(
                    left=FieldAccessBuilder("tmp2").dtype(None).build(),
                    right=FieldAccessBuilder("tmp1").dtype(None).build(),
                ),
            )
            .build()
        )
        .build()
    )
    resolve_dtype_and_validate(
        testee, {"field": A_ARITHMETIC_TYPE, "tmp1": A_ARITHMETIC_TYPE, "tmp2": A_ARITHMETIC_TYPE}
    )
