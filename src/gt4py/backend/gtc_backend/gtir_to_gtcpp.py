from typing import List, Tuple
from devtools import debug  # noqa: F401

import eve  # noqa: F401
from gtc import common
from gt4py.backend.gtc_backend import gtir, gtcppir
from gt4py.backend.gtc_backend import stencil_ast_nodes
from gt4py.backend.gtc_backend import gtcpp_codegen
from gt4py.backend.gtc_backend.gtcppir import (
    GTComputation,
    GTFunctor,
    GTInterval,
    GTMultiStage,
    GTParamList,
    GTStage,
)

# - Each vertical loop is a functor (and a stage)
# - All vertical loops build a multistage


class GTIRToGTCpp(eve.NodeTranslator):
    def visit_FieldDecl(self, node: gtir.FieldDecl, **kwargs):
        return gtcppir.ParamArg(name=node.name)

    def visit_Literal(self, node: gtir.Literal, **kwargs):
        return stencil_ast_nodes.Literal(
            value=node.value, vtype=node.dtype if node.dtype != None else common.DataType.FLOAT64
        )  # TODO this if must not be here (we need a pass to deduce datatypes)

    def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
        return stencil_ast_nodes.BinaryOp(
            op=node.op, left=self.visit(node.left), right=self.visit(node.right)
        )

    def visit_CartesianOffset(self, node: gtir.CartesianOffset, **kwargs):
        return stencil_ast_nodes.Offset(i=node.i, j=node.j, k=node.k)

    def visit_FieldAccess(self, node: gtir.FieldAccess, **kwargs):
        return gtcppir.AccessorRef(name=node.name, offset=self.visit(node.offset))

    def visit_AssignStmt(self, node: gtir.AssignStmt, **kwargs):
        return gtcppir.AssignStmt(left=self.visit(node.left), right=self.visit(node.right))

    def visit_VerticalInterval(self, node: gtir.VerticalInterval, **kwargs):
        return gtcppir.GTApplyMethod(
            interval=GTInterval(), body=[self.visit(hloop.stmt) for hloop in node.horizontal_loops]
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        functor_name = "functor_" + node.id_attr_
        return gtcppir.GTMultiStage(
            loop_order=node.loop_order,
            stages=[gtcppir.GTStage(functor=functor_name, args=[])],
            caches=[],
        ), gtcppir.GTFunctor(
            name=functor_name,
            applies=[self.visit(interval) for interval in node.vertical_intervals],
            param_list=GTParamList(accessors=[]),
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs) -> Tuple[GTComputation, List[GTFunctor]]:
        msses_and_functors: List[Tuple[GTStage, GTFunctor]] = [
            self.visit(v) for v in node.vertical_loops
        ]
        msses, functors = tuple(map(list, zip(*msses_and_functors)))
        return (
            gtcppir.GTComputation(
                name=node.id_attr_,
                parameters=[],
                temporaries=[],
                multistages=msses,
            ),
            functors,
        )

    def visit_Computation(self, node: gtir.Computation, **kwargs):
        functors = []  # TODO let this look like python
        gtcomps = []
        for s in node.stencils:
            gtcomp, functors_of_s = self.visit(s)
            functors.extend(functors_of_s)
            gtcomps.append(gtcomp)
        return gtcppir.Computation(
            name=node.name,
            parameters=[self.visit(p) for p in node.params],
            functors=functors,
            ctrl_flow_ast=gtcomps,
        )
