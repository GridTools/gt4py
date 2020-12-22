import enum
from typing import List, Optional, Tuple, Union

from eve import Str, StrEnum, SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from pydantic.class_validators import validator

from gt4py.gtc import common
from gt4py.gtc.common import LocNode


class Expr(common.Expr):
    dtype: Optional[common.DataType]

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Offset(common.CartesianOffset):
    pass


class VarDecl(Stmt):
    name: SymbolName
    init: Expr
    dtype: common.DataType


class Literal(common.Literal, Expr):
    pass


class ScalarAccess(common.ScalarAccess, Expr):
    pass


class AccessorRef(common.FieldAccess, Expr):
    pass


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class AssignStmt(common.AssignStmt[Union[ScalarAccess, AccessorRef], Expr], Stmt):
    # TODO remove duplication of this check
    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if isinstance(v, AccessorRef) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


class IfStmt(common.IfStmt[Stmt, Expr], Stmt):
    pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):
    pass


class VerticalDimension(LocNode):
    pass


class Temporary(LocNode):
    name: SymbolName
    dtype: common.DataType


class GTGrid(LocNode):
    pass


class GTLevel(LocNode):
    splitter: int
    offset: int
    # TODO validator offset != 0


class GTInterval(LocNode):
    from_level: GTLevel
    to_level: GTLevel


class GTApplyMethod(LocNode):
    interval: GTInterval
    body: List[Stmt]


@enum.unique
class Intent(StrEnum):
    IN = "in"
    INOUT = "inout"


class GTExtent(LocNode):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    @classmethod
    def zero(cls):
        return cls(i=(0, 0), j=(0, 0), k=(0, 0))

    def __add__(self, other):
        if isinstance(other, common.CartesianOffset):
            return GTExtent(
                i=(min(self.i[0], other.i), max(self.i[1], other.i)),
                j=(min(self.j[0], other.j), max(self.j[1], other.j)),
                k=(min(self.k[0], other.k), max(self.k[1], other.k)),
            )
        else:
            assert "Can only add CartesianOffsets"


class GTAccessor(LocNode):
    name: SymbolName
    id: int
    intent: Intent
    extent: GTExtent


class GTParamList(LocNode):
    accessors: List[GTAccessor]


class GTFunctor(LocNode, SymbolTableTrait):
    name: SymbolName
    applies: List[GTApplyMethod]
    param_list: GTParamList


# A ParamArg is an argument that maps to a parameter of something with the same name.
# Because all things are called exactly once there is a one-to-one mapping.
# TODO with symbol table the concept probably doesn't make sense anymore
class ParamArg(LocNode):
    name: Str


class ApiParamDecl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args, **kwargs):
        if type(self) is ApiParamDecl:
            raise TypeError("Trying to instantiate `ApiParamDecl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(ApiParamDecl):
    # TODO dimensions (or mask?)
    pass


# class ScalarDecl(Decl):
#     pass


class GlobalParamDecl(ApiParamDecl):
    pass


class GTStage(LocNode):
    functor: SymbolRef  # symbol ref
    args: List[ParamArg]  # symbol ref to GTComputation params


class IJCache(LocNode):
    name: Str  # symbol ref to GTComputation params or temporaries


class GTMultiStage(LocNode):
    loop_order: common.LoopOrder
    stages: List[GTStage]  # TODO at least one
    caches: List[Union[IJCache]]


class GTComputation(LocNode):
    name: SymbolName
    parameters: List[ParamArg]  # ?
    temporaries: List[Temporary]
    multi_stages: List[GTMultiStage]  # TODO at least one


class Program(LocNode, SymbolTableTrait):
    name: Str
    # The ParamArg here, doesn't fully work as we need the type for template instantiation.
    # But maybe the module instantiation code is actually generated from a different IR?
    parameters: List[ApiParamDecl]
    functors: List[GTFunctor]
    gt_computation: GTComputation
    # control_flow_ast: List[GTComputation]

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()
