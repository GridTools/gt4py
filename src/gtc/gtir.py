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

"""
GridTools Intermediate Representation.

GTIR represents a computation with the semantics of the
`GTScript parallel model <https://github.com/GridTools/concepts/wiki/GTScript-Parallel-model>`.

Type constraints and validators narrow the IR as much as reasonable to valid (executable) IR.

Analysis is required to generate valid code (complying with the parallel model)
- extent analysis to define the extended compute domain
- `FieldIfStmt` expansion to comply with the parallel model
"""

from typing import Any, Dict, List

from pydantic import validator

from eve import Node, Str, SymbolName, SymbolTableTrait
from gtc import common
from gtc.common import AxisBound, LocNode


class Expr(common.Expr):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class Literal(common.Literal, Expr):  # type: ignore
    pass


class CartesianOffset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls) -> "CartesianOffset":
        return cls(i=0, j=0, k=0)

    def to_dict(self) -> Dict[str, int]:
        return {"i": self.i, "j": self.j, "k": self.k}


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class FieldAccess(common.FieldAccess, Expr):  # type: ignore
    pass


class ParAssignStmt(common.AssignStmt[FieldAccess, Expr], Stmt):
    """Parallel assignment.

    R.h.s. is evaluated for all points and the resulting field is assigned
    (GTScript parallel model).
    Scalar variables on the l.h.s. are not allowed,
    as the only scalar variables are read-only stencil parameters.
    """

    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v: Expr) -> Expr:
        if v.offset.i != 0 or v.offset.j != 0:
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v

    _dtype_validation = common.assign_stmt_dtype_validation(strict=False)


class FieldIfStmt(common.IfStmt[BlockStmt, Expr], Stmt):
    """
    If statement with a field expression as condition.

    - The condition is evaluated for all gridpoints and stored in a mask.
    - Each statement inside the if and else branches is executed according
      to the same rules as statements outside of branches.

    The following restriction applies:

    - Inside the if and else blocks the same field cannot be written to
      and read with an offset in the parallel axes (order does not matter).

    See `parallel model
    <https://github.com/GridTools/concepts/wiki/GTScript-Parallel-model#conditionals-on-field-expressions>`
    """

    @validator("cond")
    def verify_scalar_condition(cls, cond: Expr) -> Expr:
        if cond.kind != common.ExprKind.FIELD:
            raise ValueError("Condition is not a field expression")
        return cond

    # TODO(havogt) add validator for the restriction (it's a pass over the subtrees...)


class ScalarIfStmt(common.IfStmt[BlockStmt, Expr], Stmt):
    """
    If statement with a scalar expression as condition.

    No special rules apply.
    """

    @validator("cond")
    def verify_scalar_condition(cls, cond: Expr) -> Expr:
        if cond.kind != common.ExprKind.SCALAR:
            raise ValueError("Condition is not scalar")
        return cond


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_validator = common.binary_op_dtype_propagation(strict=False)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=False)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=False)


class Decl(LocNode):  # TODO probably Stmt
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self) is Decl:
            raise TypeError("Trying to instantiate `Decl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(Decl):
    # TODO dimensions
    pass


class ScalarDecl(Decl):
    pass


class Interval(LocNode):
    start: AxisBound
    end: AxisBound


# TODO(havogt) should vertical loop open a scope?
class VerticalLoop(LocNode):
    interval: Interval
    loop_order: common.LoopOrder
    temporaries: List[FieldDecl]
    body: List[Stmt]


class Stencil(LocNode, SymbolTableTrait):
    name: Str
    # TODO(havogt) deal with gtscript externals
    params: List[Decl]
    vertical_loops: List[VerticalLoop]

    @property
    def param_names(self) -> List:
        return [p.name for p in self.params]

    _validate_symbol_refs = common.validate_symbol_refs()
