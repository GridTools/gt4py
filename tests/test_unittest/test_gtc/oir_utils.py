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

from typing import List, Optional, Tuple, Union

from gtc.common import CartesianOffset, DataType, ExprKind, LoopOrder
from gtc.oir import (
    AssignStmt,
    AxisBound,
    CacheDecl,
    Decl,
    Expr,
    FieldAccess,
    FieldDecl,
    HorizontalExecution,
    Interval,
    ScalarAccess,
    ScalarDecl,
    Stencil,
    Stmt,
    SymbolName,
    Temporary,
    VerticalLoop,
)


class AssignStmtBuilder:
    def __init__(self, left_name=None, right_name=None, right_offset=None) -> None:
        self._left = FieldAccessBuilder(left_name).build() if left_name else None
        self._right = (
            FieldAccessBuilder(right_name, offset=right_offset).build() if right_name else None
        )

    def left(self, access: Union[ScalarAccess, FieldAccess]) -> "AssignStmtBuilder":
        self._left = access
        return self

    def right(self, expr: Expr) -> "AssignStmtBuilder":
        self._right = expr
        return self

    def build(self) -> AssignStmt:
        return AssignStmt(left=self._left, right=self._right)


class CartesianOffsetBuilder:
    def __init__(self, i=None, j=None, k=None) -> None:
        self._i = i if i else 0
        self._j = j if j else 0
        self._k = k if k else 0

    def build(self) -> CartesianOffset:
        return CartesianOffset(i=self._i, j=self._j, k=self._k)


class FieldAccessBuilder:
    def __init__(self, name: str, offset: Optional[Tuple[int, int, int]] = None) -> None:
        self._name = name
        self._offset = CartesianOffsetBuilder(*(offset if offset else [])).build()
        self._kind = ExprKind.FIELD
        self._dtype = DataType.FLOAT32

    def offset(self, offset: CartesianOffset) -> "FieldAccessBuilder":
        self._offset = offset
        return self

    def dtype(self, dtype: DataType) -> "FieldAccessBuilder":
        self._dtype = dtype
        return self

    def build(self) -> FieldAccess:
        return FieldAccess(name=self._name, offset=self._offset, dtype=self._dtype, kind=self._kind)


class TemporaryBuilder:
    def __init__(self, name: SymbolName = None, dtype: DataType = None) -> None:
        self._name = name
        self._dtype = DataType.FLOAT32 if dtype is None else dtype

    def build(self) -> Temporary:
        return Temporary(name=self._name, dtype=self._dtype)


class FieldDeclBuilder:
    def __init__(self, name: SymbolName = None, dtype: DataType = None) -> None:
        self._name = name
        self._dtype = DataType.FLOAT32 if dtype is None else dtype

    def build(self) -> FieldDecl:
        return FieldDecl(name=self._name, dtype=self._dtype)


class HorizontalExecutionBuilder:
    def __init__(self) -> None:
        self._body: List[Stmt] = []
        self._mask = None
        self._declarations: List[ScalarDecl] = []

    def add_stmt(self, stmt: Stmt) -> "HorizontalExecutionBuilder":
        self._body.append(stmt)
        return self

    def build(self) -> HorizontalExecution:
        return HorizontalExecution(
            body=self._body, mask=self._mask, declarations=self._declarations
        )


class VerticalLoopBuilder:
    def __init__(self) -> None:
        self._interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        self._horizontal_executions: List[HorizontalExecution] = []
        self._loop_order = LoopOrder.PARALLEL
        self._declarations: List[Temporary] = []
        self._caches: List[CacheDecl] = []

    def add_horizontal_execution(
        self, horizontal_execution: HorizontalExecution
    ) -> "VerticalLoopBuilder":
        self._horizontal_executions.append(horizontal_execution)
        return self

    def add_declaration(self, declaration: Temporary) -> "VerticalLoopBuilder":
        self._declarations.append(declaration)
        return self

    def build(self) -> VerticalLoop:
        return VerticalLoop(
            interval=self._interval,
            horizontal_executions=self._horizontal_executions,
            loop_order=self._loop_order,
            declarations=self._declarations,
            caches=self._caches,
        )


class StencilBuilder:
    def __init__(self, name="foo") -> None:
        self._name: str = name
        self._params: List[Decl] = []
        self._vertical_loops: List[VerticalLoop] = []

    def add_param(self, param: Decl) -> "StencilBuilder":
        self._params.append(param)
        return self

    def add_vertical_loop(self, vertical_loop: VerticalLoop) -> "StencilBuilder":
        self._vertical_loops.append(vertical_loop)
        return self

    def build(self) -> Stencil:
        return Stencil(
            name=self._name,
            params=self._params,
            vertical_loops=self._vertical_loops,
        )
