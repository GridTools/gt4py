from typing import Dict, List, Sequence

import eve
from devtools import debug  # noqa: F401

from gt4py.gtc import common, oir, utils
from gt4py.gtc.common import CartesianOffset
from gt4py.gtc.gtcpp import gtcpp
from gt4py.gtc.gtcpp.gtcpp import GTParamList


# TODO between oir and gtcpp we need to group oir.VerticalLoops

# - Each vertical loop is a functor (and a stage)
# - All vertical loops build a multistage


def _extract_accessors(node: eve.Node) -> GTParamList:
    extents: Dict[str, gtcpp.GTExtent] = (
        node.iter_tree()
        .if_isinstance(gtcpp.AccessorRef)
        .reduceby(
            (lambda extent, accessor_ref: extent + accessor_ref.offset),
            "name",
            init=gtcpp.GTExtent.zero(),
            as_dict=True,
        )
    )

    inout_fields: List[str] = (
        node.iter_tree()
        .if_isinstance(gtcpp.AssignStmt)
        .getattr("left")
        .if_isinstance(gtcpp.AccessorRef)
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

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs):
        return gtcpp.UnaryOp(op=node.op, expr=self.visit(node.expr, **kwargs))

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs):
        return gtcpp.BinaryOp(
            op=node.op,
            left=self.visit(node.left, **kwargs),
            right=self.visit(node.right, **kwargs),
        )

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs):
        return gtcpp.TernaryOp(
            cond=self.visit(node.cond, **kwargs),
            true_expr=self.visit(node.true_expr, **kwargs),
            false_expr=self.visit(node.false_expr, **kwargs),
        )

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs):
        return gtcpp.NativeFuncCall(func=node.func, args=self.visit(node.args))

    def visit_Cast(self, node: oir.Cast, **kwargs):
        return gtcpp.Cast(dtype=node.dtype, expr=self.visit(node.expr, **kwargs))

    def visit_Temporary(self, node: oir.Temporary, **kwargs):
        return gtcpp.Temporary(name=node.name, dtype=node.dtype)

    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs):
        return node

    def visit_FieldAccess(self, node: oir.FieldAccess, **kwargs):
        return gtcpp.AccessorRef(name=node.name, offset=self.visit(node.offset), dtype=node.dtype)

    def visit_ScalarAccess(self, node: oir.ScalarAccess, **kwargs):
        assert "stencil_symtable" in kwargs
        if node.name in kwargs["stencil_symtable"]:
            symbol = kwargs["stencil_symtable"][node.name]
            assert isinstance(symbol, oir.ScalarDecl)
            return gtcpp.AccessorRef(
                name=symbol.name, offset=CartesianOffset.zero(), dtype=symbol.dtype
            )
        else:
            return gtcpp.ScalarAccess(name=node.name, dtype=node.dtype)

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
        assert "stencil_symtable" in kwargs
        return gtcpp.AssignStmt(
            left=self.visit(node.left, **kwargs), right=self.visit(node.right, **kwargs)
        )

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, *, interval, **kwargs):
        assert "stencil_symtable" in kwargs
        apply_method = gtcpp.GTApplyMethod(
            interval=self.visit(interval), body=self.visit(node.body, **kwargs)
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
        functors, stages = self.tuple_visit(
            node.horizontal_executions, interval=node.interval, **kwargs
        )
        assert all([isinstance(decl, oir.Temporary) for decl in node.declarations])
        temporaries = self.visit(node.declarations)
        caches = []  # TODO
        return (
            functors,
            temporaries,
            gtcpp.GTMultiStage(loop_order=node.loop_order, stages=stages, caches=caches),
        )

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs):
        return gtcpp.FieldDecl(name=node.name, dtype=node.dtype)

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs):
        return gtcpp.GlobalParamDecl(name=node.name, dtype=node.dtype)

    def visit_Stencil(self, node: oir.Stencil, **kwargs):
        functors, temporaries, multi_stages = self.tuple_visit(
            node.vertical_loops, stencil_symtable=node.symtable_, **kwargs
        )

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
