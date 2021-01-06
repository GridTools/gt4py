from typing import List

from gt4py.gtc.common import CartesianOffset, DataType, ExprKind, LoopOrder
from gt4py.gtc.gtcpp.gtcpp import (
    AccessorRef,
    AssignStmt,
    FieldDecl,
    GTAccessor,
    GTApplyMethod,
    GTComputation,
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
    ParamArg,
    Program,
    Stmt,
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
        return AccessorRef(
            name=self._name, offset=self._offset, dtype=self._dtype, kind=self._kind
        )


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
        self._body = []

    def add_stmt(self, stmt: Stmt) -> "GTApplyMethodBuilder":
        self._body.append(stmt)
        return self

    def build(self) -> GTApplyMethod:
        return GTApplyMethod(interval=self._interval, body=self._body)


class IfStmtBuilder:
    def __init__(self) -> None:
        self._cond = Literal(value="true", dtype=DataType.BOOL)
        self._true_branch = None
        self._false_branch = None

    def true_branch(self, stmt: Stmt) -> "IfStmtBuilder":
        self._true_branch = stmt
        return self

    def build(self) -> IfStmt:
        return IfStmt(
            cond=self._cond, true_branch=self._true_branch, false_branch=self._false_branch
        )


class GTAccessorBuilder:
    def __init__(self, name, id) -> None:
        self._name = name
        self._id = id
        self._intent = Intent.INOUT
        self._extent = GTExtent.zero()

    def build(self) -> GTAccessor:
        return GTAccessor(name=self._name, id=self._id, intent=self._intent, extent=self._extent)


class GTFunctorBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._applies = []
        self._param_list_accessors = []

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


class GTComputationBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._parameters = []
        self._temporaries = []
        self._multi_stages = []

    def add_stage(self, stage: GTStage) -> "GTComputationBuilder":
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

    def add_parameter(self, name: str) -> "GTComputationBuilder":
        self._parameters.append(ParamArg(name=name))
        return self

    def build(self) -> GTComputation:
        return GTComputation(
            name=self._name,
            parameters=self._parameters,
            temporaries=self._temporaries,
            multi_stages=self._multi_stages,
        )


class ProgramBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._parameters = []
        self._functors = []
        self._gt_computation = GTComputationBuilder(name).build()

    def add_functor(self, functor: GTFunctor) -> "ProgramBuilder":
        self._functors.append(functor)
        return self

    def add_parameter(self, name: str, dtype: DataType) -> "ProgramBuilder":
        self._parameters.append(FieldDecl(name=name, dtype=dtype))
        return self

    def gt_computation(self, gt_computation: GTComputation) -> "ProgramBuilder":
        self._gt_computation = gt_computation
        return self

    def build(self) -> Program:
        return Program(
            name=self._name,
            parameters=self._parameters,
            functors=self._functors,
            gt_computation=self._gt_computation,
        )
