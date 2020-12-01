from typing import List

from gt4py.gtc.common import CartesianOffset, DataType, ExprKind, LoopOrder
from gt4py.gtc.gtcpp.gtcpp import (
    AccessorRef,
    GTAccessor,
    GTApplyMethod,
    GTComputation,
    GTFunctor,
    GTInterval,
    GTLevel,
    GTMultiStage,
    GTParamList,
    GTStage,
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

    def add_stmt(self, stmt: Stmt):
        self._body.append(stmt)
        return self

    def build(self) -> GTApplyMethod:
        return GTApplyMethod(interval=self._interval, body=self._body)


class GTFunctorBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._applies = []
        self._param_list_accessors = []

    def add_accessors(self, accessors: List[GTAccessor]):
        self._param_list_accessors.extend(accessors)
        return self

    def add_accessor(self, accessor: GTAccessor):
        self._param_list_accessors.append(accessor)
        return self

    def add_apply_method(self, apply_method: GTApplyMethod = GTApplyMethodBuilder().build()):
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

    def add_parameter(self, name: str) -> "ProgramBuilder":
        self._parameters.append(ParamArg(name=name))
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
