from eve import Node, Str
from . import common

from typing import Union, Optional


class Expr(Node):
    pass


class Stmt(Node):
    pass


class Offset(Node):
    i: int
    j: int
    k: int

    @classmethod
    def zero(cls):
        return cls(i=0, j=0, k=0)


# class FieldAccess(Expr):
#     name: Str  # symbol ref to SidCompositeEntry
#     offset: Offset


class VarDecl(Stmt):
    name: Str
    init: Expr
    vtype: common.DataType


class Literal(Expr):
    value: Union[common.BuiltInLiteral, Str]
    vtype: common.DataType


class VarAccess(Expr):
    name: Str  # via symbol table
    dummy: Optional[
        Str
    ]  # to distinguish from FieldAccess, see https://github.com/eth-cscs/eve_toolchain/issues/34


class BinaryOp(Expr):
    op: common.BinaryOperator
    left: Expr
    right: Expr
