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
    CacheDesc,
    Decl,
    Expr,
    FieldAccess,
    FieldDecl,
    HorizontalExecution,
    IJCache,
    Interval,
    KCache,
    LocalScalar,
    ScalarAccess,
    ScalarDecl,
    Stencil,
    Stmt,
    SymbolName,
    Temporary,
    VerticalLoop,
    VerticalLoopSection,
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


class LocalScalarBuilder:
    def __init__(self, name: SymbolName = None, dtype: DataType = None) -> None:
        self._name = name
        self._dtype = DataType.FLOAT32 if dtype is None else dtype

    def build(self) -> LocalScalar:
        return LocalScalar(name=self._name, dtype=self._dtype)


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
        self._mask: Optional[Expr] = None
        self._declarations: List[ScalarDecl] = []

    def add_stmt(self, stmt: Stmt) -> "HorizontalExecutionBuilder":
        self._body.append(stmt)
        return self

    def mask(self, mask: Expr) -> "HorizontalExecutionBuilder":
        self._mask = mask
        return self

    def add_declaration(self, decl: ScalarDecl) -> "HorizontalExecutionBuilder":
        self._declarations.append(decl)
        return self

    def build(self) -> HorizontalExecution:
        return HorizontalExecution(
            body=self._body, mask=self._mask, declarations=self._declarations
        )


class VerticalLoopSectionBuilder:
    def __init__(self) -> None:
        self._interval = Interval(start=AxisBound.start(), end=AxisBound.end())
        self._horizontal_executions: List[HorizontalExecution] = []

    def add_horizontal_execution(
        self, horizontal_execution: HorizontalExecution
    ) -> "VerticalLoopSectionBuilder":
        self._horizontal_executions.append(horizontal_execution)
        return self

    def build(self) -> VerticalLoopSection:
        return VerticalLoopSection(
            interval=self._interval, horizontal_executions=self._horizontal_executions
        )


class VerticalLoopBuilder:
    def __init__(self) -> None:
        self._loop_order = LoopOrder.PARALLEL
        self._sections: List[VerticalLoopSection] = []
        self._caches: List[CacheDesc] = []

    def loop_order(self, loop_order: LoopOrder) -> "VerticalLoopBuilder":
        self._loop_order = loop_order
        return self

    def add_section(self, section: VerticalLoopSection) -> "VerticalLoopBuilder":
        self._sections.append(section)
        return self

    def add_cache(self, cache: CacheDesc) -> "VerticalLoopBuilder":
        self._caches.append(cache)
        return self

    def build(self) -> VerticalLoop:
        return VerticalLoop(
            sections=self._sections,
            loop_order=self._loop_order,
            caches=self._caches,
        )


class StencilBuilder:
    def __init__(self, name="foo") -> None:
        self._name: str = name
        self._params: List[Decl] = []
        self._vertical_loops: List[VerticalLoop] = []
        self._declarations: List[Temporary] = []

    def add_param(self, param: Decl) -> "StencilBuilder":
        self._params.append(param)
        return self

    def add_vertical_loop(self, vertical_loop: VerticalLoop) -> "StencilBuilder":
        self._vertical_loops.append(vertical_loop)
        return self

    def add_declaration(self, declaration: Temporary) -> "StencilBuilder":
        self._declarations.append(declaration)
        return self

    def build(self) -> Stencil:
        return Stencil(
            name=self._name,
            params=self._params,
            vertical_loops=self._vertical_loops,
            declarations=self._declarations,
        )


class IJCacheBuilder:
    def __init__(self, name):
        self._name = name

    def build(self) -> IJCache:
        return IJCache(name=self._name)


class KCacheBuilder:
    def __init__(self, name, fill=True, flush=True):
        self._name = name
        self._fill = fill
        self._flush = flush

    def build(self) -> KCache:
        return KCache(name=self._name, fill=self._fill, flush=self._flush)
