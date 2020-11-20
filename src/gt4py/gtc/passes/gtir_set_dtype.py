from eve import NodeTranslator

from gt4py.gtc import gtir, common
from gt4py.gtc.gtir import AxisBound, FieldAccess, ParAssignStmt, VerticalLoop, Interval

from devtools import debug


class GTIRSetDtype(NodeTranslator):
    def visit_FieldAccess(self, node: gtir.FieldAccess, *, symtable, **kwargs):
        return gtir.FieldAccess(
            name=node.name, offset=node.offset, dtype=symtable[node.name].dtype
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return self.generic_visit(node, symtable=node.symtable_)


# stencil = gtir.Stencil(
#     name="dummy",
#     params=[gtir.FieldDecl(name="tmp", dtype=common.DataType.BOOL)],
#     vertical_loops=[
#         VerticalLoop(
#             interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
#             loop_order=common.LoopOrder.FORWARD,
#             body=[
#                 ParAssignStmt(
#                     left=FieldAccess.centered(name="tmp"), right=FieldAccess.centered(name="tmp")
#                 )
#             ],
#         )
#     ],
# )


# # debug(stencil)
# debug(GTIRSetDtype().visit(stencil))
