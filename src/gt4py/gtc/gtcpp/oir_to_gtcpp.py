from typing import List, Tuple, Sequence
from devtools import debug  # noqa: F401

from eve import NodeTranslator, FindNodes
from gt4py.gtc.common import CartesianOffset, DataType

from gt4py.gtc.gtcpp import gtcpp
from gt4py.gtc import oir, utils, common
from gt4py.gtc.gtcpp.gtcpp import GTFunctor, GTMultiStage, GTParamList

# TODO between oir and gtcpp we need to group oir.VerticalLoops

# - Each vertical loop is a functor (and a stage)
# - All vertical loops build a multistage


class OIRToGTCpp(NodeTranslator):
    def visit_Literal(self, node: oir.Literal, **kwargs):
        return gtcpp.Literal(value=node.value, dtype=node.dtype)

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs):
        return gtcpp.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_Temporary(self, node: oir.Temporary, **kwargs):
        return gtcpp.Temporary(name=node.name, dtype=node.dtype)

    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs):
        return node

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs):
        return gtcpp.AccessorRef(name=node.name, offset=self.visit(node.offset), dtype=node.dtype)

    def visit_AxisBound(self, node: oir.AxisBound, *, is_start: bool, **kwargs):
        if node.level == common.LevelMarker.START:
            splitter = 0
            offset = node.offset + 1 if (node.offset >= 0 and is_start) else node.offset
        elif node.level == common.LevelMarker.END:
            splitter = 1
            offset = node.offset - 1 if (node.offset <= 0 and not is_start) else node.offset
        else:
            raise ValueError("Cannot handle dynamic levels")
        return gtcpp.GTLevel(splitter=splitter, offset=offset)

    def visit_Interval(self, node: oir.Interval, **kwargs):
        return gtcpp.GTInterval(
            from_level=self.visit(node.start, is_start=True),
            to_level=self.visit(node.end, is_start=False),
        )

    def tuple_visit(self, node, **kwargs):
        """Visits a list node and transforms a list of tuples to a ListTuple."""
        assert isinstance(node, Sequence)
        return utils.ListTuple(*map(utils.flatten_list, zip(*self.visit(node, **kwargs))))

    def visit_AssignStmt(self, node: oir.AssignStmt, **kwargs):
        return gtcpp.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, interval, **kwargs):
        body = self.visit(node.body)
        fields = set([f.name for f in FindNodes.by_type(oir.FieldAccess, node)])
        param_list = GTParamList(
            # TODO this is a hack:
            accessors=[
                gtcpp.GTAccessor(
                    name=f, id=i, intent=gtcpp.Intent.INOUT, extent=gtcpp.GTExtent.zero()  # TODO
                )
                for i, f in enumerate(fields)
            ]
        )
        stage_args = [gtcpp.ParamArg(name=f) for f in fields]
        return (
            gtcpp.GTFunctor(
                name=node.id_,
                applies=[gtcpp.GTApplyMethod(interval=self.visit(interval), body=body)],
                param_list=param_list,
            ),
            gtcpp.GTStage(functor=node.id_, args=stage_args),
        )

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs):
        functors, stages = self.tuple_visit(node.horizontal_executions, interval=node.interval)
        assert all([isinstance(decl, oir.Temporary) for decl in node.declarations])
        temporaries = self.visit(node.declarations)
        caches = []  # TODO
        return (
            functors,
            temporaries,
            GTMultiStage(loop_order=node.loop_order, stages=stages, caches=caches),
        )

    def visit_Decl(self, node: oir.Decl, **kwargs):
        return gtcpp.ParamArg(name=node.name)

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        functors, temporaries, multi_stages = self.tuple_visit(node.vertical_loops)
        fields = set(
            [arg.name for mss in multi_stages for stage in mss.stages for arg in stage.args]
        )
        gt_comp_parameters = [gtcpp.ParamArg(name=f) for f in fields]  # TODO
        gt_computation = gtcpp.GTComputation(
            name=node.name,
            parameters=gt_comp_parameters,
            temporaries=temporaries,
            multi_stages=multi_stages,
        )
        parameters = self.visit(node.params)
        return gtcpp.Program(
            name=node.name, parameters=parameters, functors=functors, gt_computation=gt_computation
        )
