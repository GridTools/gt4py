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
from gtc.gtir import FieldDecl, Stencil
from gtc.passes.gtir_dtype_resolver import _GTIRPropagateDtypeToAccess, resolve_dtype

from .gtir_utils import (
    FieldAccessFactory,
    FieldDeclFactory,
    LiteralFactory,
    ParAssignStmtFactory,
    StencilFactory,
    VerticalLoopFactory,
)


A_ARITHMETIC_TYPE = common.DataType.FLOAT32


def test_propagate_dtype_to_FieldAccess():
    name = "foo"
    decl = FieldDecl(name=name, dtype=A_ARITHMETIC_TYPE)

    testee = FieldAccessFactory(name=name)

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
    testee = StencilFactory(
        params=[FieldDeclFactory(name="field", dtype=A_ARITHMETIC_TYPE)],
        vertical_loops__0__body__0=ParAssignStmtFactory(
            left__name="field", left__dtype=None, right__name="field", right__dtype=None
        ),
    )
    resolve_dtype_and_validate(
        testee,
        {"field": A_ARITHMETIC_TYPE},
    )


def test_resolve_AUTO_from_literal_to_temporary():
    testee = StencilFactory(
        vertical_loops__0=VerticalLoopFactory(
            temporaries=[FieldDeclFactory(name="tmp", dtype=DataType.AUTO)],
            body=[
                ParAssignStmtFactory(
                    left__name="tmp",
                    left__dtype=None,
                    right=LiteralFactory(value="0", dtype=A_ARITHMETIC_TYPE),
                )
            ],
        )
    )
    resolve_dtype_and_validate(testee, {"tmp": A_ARITHMETIC_TYPE})


def test_resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary():
    testee = StencilFactory(
        params=[FieldDeclFactory(name="field", dtype=A_ARITHMETIC_TYPE)],
        vertical_loops__0=VerticalLoopFactory(
            temporaries=[FieldDeclFactory(name="tmp", dtype=DataType.AUTO)],
            body=[
                ParAssignStmtFactory(
                    left__name="tmp", left__dtype=None, right__name="field", right__dtype=None
                )
            ],
        ),
    )
    resolve_dtype_and_validate(testee, {"field": A_ARITHMETIC_TYPE, "tmp": A_ARITHMETIC_TYPE})


def test_resolve_AUTO_from_FieldDecl_to_FieldAccess_to_temporary_to_FieldAccess_to_temporary():
    testee = StencilFactory(
        params=[FieldDeclFactory(name="field", dtype=A_ARITHMETIC_TYPE)],
        vertical_loops__0=VerticalLoopFactory(
            temporaries=[
                FieldDeclFactory(name="tmp1", dtype=DataType.AUTO),
                FieldDeclFactory(name="tmp2", dtype=DataType.AUTO),
            ],
            body=[
                ParAssignStmtFactory(
                    left__name="tmp1", left__dtype=None, right__name="field", right__dtype=None
                ),
                ParAssignStmtFactory(
                    left__name="tmp2", left__dtype=None, right__name="tmp1", right__dtype=None
                ),
            ],
        ),
    )
    resolve_dtype_and_validate(
        testee, {"field": A_ARITHMETIC_TYPE, "tmp1": A_ARITHMETIC_TYPE, "tmp2": A_ARITHMETIC_TYPE}
    )
