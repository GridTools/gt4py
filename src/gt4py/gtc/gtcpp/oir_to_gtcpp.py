from typing import Dict, List, Sequence

import eve
from devtools import debug  # noqa: F401

from gt4py.gtc import common, oir, utils
from gt4py.gtc.gtcpp import gtcpp
from gt4py.gtc.gtcpp.gtcpp import GTParamList


# TODO between oir and gtcpp we need to group oir.VerticalLoops

# - Each vertical loop is a functor (and a stage)
# - All vertical loops build a multistage


def _extract_accessors(node: eve.Node) -> GTParamList:
    extents: Dict[str, gtcpp.GTExtent] = (
        node.iter_tree()
        .filter_by_type(gtcpp.AccessorRef)
        .reduceby(
            (lambda extent, accessor_ref: extent + accessor_ref.offset),
            "name",
            init=gtcpp.GTExtent.zero(),
            as_dict=True,
        )
    )

    inout_fields: List[str] = (
        node.iter_tree()
        .filter_by_type(gtcpp.AssignStmt)
        .getattr("left")
        .filter_by_type(gtcpp.AccessorRef)
        .getattr("name")
        .unique()
    )

    return [
        gtcpp.GTAccessor(
            name=name,
            id=i,
            intent=gtcpp.Intent.INOUT if name in inout_fields else gtcpp.Intent.IN,
            extent=extent,
        )
        for i, (name, extent) in enumerate(extents.items())
    ]


class OIRToGTCpp(eve.NodeTranslator):
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
        apply_method = gtcpp.GTApplyMethod(
            interval=self.visit(interval), body=self.visit(node.body)
        )
        accessors = _extract_accessors(apply_method)
        stage_args = [gtcpp.ParamArg(name=acc.name) for acc in accessors]
        return (
            gtcpp.GTFunctor(
                name=node.id_,
                applies=[apply_method],
                param_list=GTParamList(accessors=accessors),
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
            gtcpp.GTMultiStage(loop_order=node.loop_order, stages=stages, caches=caches),
        )

    def visit_Decl(self, node: oir.Decl, **kwargs):
        return gtcpp.ParamArg(name=node.name)

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        functors, temporaries, multi_stages = self.tuple_visit(node.vertical_loops)

        # TODO think about this pattern, just scanning of used parameters is probably wrong...
        api_fields = set(
            [arg.name for mss in multi_stages for stage in mss.stages for arg in stage.args]
        ) - set(t.name for t in temporaries)
        gt_comp_parameters = [gtcpp.ParamArg(name=f) for f in api_fields]  # TODO
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
