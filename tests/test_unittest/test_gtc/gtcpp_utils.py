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

from typing import List, Optional

from gtc.common import CartesianOffset, DataType, ExprKind, LoopOrder
from gtc.gtcpp.gtcpp import (
    AccessorRef,
    ApiParamDecl,
    Arg,
    AssignStmt,
    Expr,
    FieldDecl,
    GTAccessor,
    GTApplyMethod,
    GTComputationCall,
    GTExtent,
    GTFunctor,
    GTInterval,
    GTLevel,
    GTMultiStage,
    GTParamList,
    GTStage,
    IfStmt,
    Intent,
    Literal,
    Program,
    Stmt,
    Temporary,
)


class AccessorRefBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._offset = CartesianOffset.zero()
        self._kind = ExprKind.FIELD
        self._dtype = DataType.FLOAT32

    def offset(self, offset: CartesianOffset) -> "AccessorRefBuilder":
        self._offset = offset
        return self

    def dtype(self, dtype: DataType) -> "AccessorRefBuilder":
        self._dtype = dtype
        return self

    def build(self) -> AccessorRef:
        return AccessorRef(name=self._name, offset=self._offset, dtype=self._dtype, kind=self._kind)


class AssignStmtBuilder:
    def __init__(self, left: str = "left", right: str = "right") -> None:
        self._left = AccessorRefBuilder(left).build()
        self._right = AccessorRefBuilder(right).build()

    def build(self) -> AssignStmt:
        return AssignStmt(left=self._left, right=self._right)


class GTIntervalBuilder:
    def __init__(self) -> None:
        self._from_level = GTLevel(splitter=0, offset=1)
        self._to_level = GTLevel(splitter=1, offset=-1)

    def build(self) -> GTInterval:
        return GTInterval(from_level=self._from_level, to_level=self._to_level)


class GTApplyMethodBuilder:
    def __init__(self) -> None:
        self._interval = GTIntervalBuilder().build()
        self._body: List[Stmt] = []

    def add_stmt(self, stmt: Stmt) -> "GTApplyMethodBuilder":
        self._body.append(stmt)
        return self

    def build(self) -> GTApplyMethod:
        return GTApplyMethod(interval=self._interval, body=self._body)


class IfStmtBuilder:
    def __init__(self) -> None:
        self._cond = Literal(value="true", dtype=DataType.BOOL)
        self._true_branch: Optional[Expr] = None
        self._false_branch: Optional[Expr] = None

    def true_branch(self, stmt: Stmt) -> "IfStmtBuilder":
        self._true_branch = stmt
        return self

    def build(self) -> IfStmt:
        return IfStmt(
            cond=self._cond, true_branch=self._true_branch, false_branch=self._false_branch
        )


class GTAccessorBuilder:
    def __init__(self, name, id) -> None:  # noqa: A002  # shadowing python builtin
        self._name = name
        self._id = id
        self._intent = Intent.INOUT
        self._extent = GTExtent.zero()

    def build(self) -> GTAccessor:
        return GTAccessor(name=self._name, id=self._id, intent=self._intent, extent=self._extent)


class GTFunctorBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._applies: List[GTApplyMethod] = []
        self._param_list_accessors: List[GTAccessor] = []

    def add_accessors(self, accessors: List[GTAccessor]) -> "GTFunctorBuilder":
        self._param_list_accessors.extend(accessors)
        return self

    def add_accessor(self, accessor: GTAccessor) -> "GTFunctorBuilder":
        self._param_list_accessors.append(accessor)
        return self

    def add_apply_method(
        self, apply_method: GTApplyMethod = GTApplyMethodBuilder().build()
    ) -> "GTFunctorBuilder":
        self._applies.append(apply_method)
        return self

    def build(self) -> GTFunctor:
        return GTFunctor(
            name=self._name,
            applies=self._applies,
            param_list=GTParamList(accessors=self._param_list_accessors),
        )


class GTComputationCallBuilder:
    def __init__(self) -> None:
        self._arguments: List[Arg] = []
        self._temporaries: List[Temporary] = []
        self._multi_stages: List[GTMultiStage] = []

    def add_stage(self, stage: GTStage) -> "GTComputationCallBuilder":
        if len(self._multi_stages) == 0:
            self._multi_stages.append(
                GTMultiStage(loop_order=LoopOrder.PARALLEL, stages=[], caches=[])
            )
        mss = self._multi_stages[-1]
        stages = mss.stages
        stages.append(stage)
        self._multi_stages[-1] = GTMultiStage(
            loop_order=mss.loop_order, stages=stages, caches=mss.caches
        )
        return self

    def add_argument(self, name: str) -> "GTComputationCallBuilder":
        self._arguments.append(Arg(name=name))
        return self

    def build(self) -> GTComputationCall:
        return GTComputationCall(
            arguments=self._arguments,
            temporaries=self._temporaries,
            multi_stages=self._multi_stages,
        )


class ProgramBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._parameters: List[ApiParamDecl] = []
        self._functors: List[GTFunctor] = []
        self._gt_computation = GTComputationCallBuilder().build()

    def add_functor(self, functor: GTFunctor) -> "ProgramBuilder":
        self._functors.append(functor)
        return self

    def add_parameter(self, name: str, dtype: DataType) -> "ProgramBuilder":
        self._parameters.append(FieldDecl(name=name, dtype=dtype))
        return self

    def gt_computation(self, gt_computation: GTComputationCall) -> "ProgramBuilder":
        self._gt_computation = gt_computation
        return self

    def build(self) -> Program:
        return Program(
            name=self._name,
            parameters=self._parameters,
            functors=self._functors,
            gt_computation=self._gt_computation,
        )
