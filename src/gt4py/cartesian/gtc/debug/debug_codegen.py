from gt4py import eve
from gt4py.cartesian import utils
from gt4py.cartesian.gtc.common import AxisBound, HorizontalInterval, HorizontalMask, LevelMarker
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.oir import (
    AssignStmt,
    BinaryOp,
    Cast,
    FieldAccess,
    FieldDecl,
    HorizontalExecution,
    HorizontalRestriction,
    Interval,
    Literal,
    Stencil,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import StencilExtentComputer
from gt4py.eve import codegen


class DebugCodeGen(codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):
    def __init__(self) -> None:
        self.body = utils.text.TextBlock()

    def visit_VerticalLoop(self):
        pass

    def generate_field_decls(self, declarations: list[FieldDecl]):
        field_generation = []
        for declaration in declarations:
            field_generation.append(
                f"{declaration.name} = Field({declaration.name}, _origin_['{declaration.name}'], "
                f"({', '.join([str(x) for x in declaration.dimensions])}))"
            )
        return field_generation

    def visit_FieldAccess(self, field_access: FieldAccess, **_):
        full_string = field_access.name + "[" + field_access.offset.to_str() + "]"
        return full_string

    def visit_AssignStmt(self, assignment_statement: AssignStmt, **_):
        self.body.append(
            self.visit(assignment_statement.left) + "=" + self.visit(assignment_statement.right)
        )

    def visit_BinaryOp(self, binary: BinaryOp, **_):
        return self.visit(binary.left) + str(binary.op) + self.visit(binary.right)

    def visit_Literal(self, literal: Literal, **_):
        return str(literal.value)

    def visit_Cast(self, cast: Cast, **_):
        return self.visit(cast.expr)

    def visit_HorizontalExecution(self, horizontal_execution: HorizontalExecution, **_):
        for stmt in horizontal_execution.body:
            self.visit(stmt)

    def visit_HorizontalMask(self, horizontal_mask: HorizontalMask, **_):
        i_min, i_max = self.visit(horizontal_mask.i, var="i")
        j_min, j_max = self.visit(horizontal_mask.j, var="j")
        conditions = []
        if i_min != "None":
            conditions.append(f"({i_min}) <= i")
        if i_max != "None":
            conditions.append(f"i < ({i_max})")
        if j_min != "None":
            conditions.append(f"({j_min}) <= j")
        if j_max != "None":
            conditions.append(f"j < ({j_max})")
        assert len(conditions)
        if_code = f"if( {' and '.join(conditions)} ):"
        self.body.append(if_code)

    def visit_HorizontalInterval(self, horizontal_interval: HorizontalInterval, **kwargs):
        return self.visit(horizontal_interval.start, **kwargs), self.visit(
            horizontal_interval.end, **kwargs
        )

    def visit_HorizontalRestriction(self, horizontal_restriction: HorizontalRestriction, **_):
        self.visit(horizontal_restriction.mask)
        self.body.indent()
        self.visit(horizontal_restriction.body)
        self.body.dedent()

    @staticmethod
    def compute_extents(node: Stencil, **_) -> tuple[dict[str, Extent], dict[int, Extent]]:
        ctx: StencilExtentComputer.Context = StencilExtentComputer().visit(node)
        return ctx.fields, ctx.blocks

    def visit_Stencil(self, stencil: Stencil, **_):
        _, block_extents = self.compute_extents(stencil)
        self.body.append("from gt4py.cartesian.utils import Field")

        function_signature = "def run(*"
        args = []
        for param in stencil.params:
            args.append(self.visit(param))
        function_signature = ",".join([function_signature, *args])
        function_signature += ",_domain_, _origin_):"
        self.body.append(function_signature)
        self.body.indent()
        field_declarations = self.generate_field_decls(stencil.params)
        for declaration in field_declarations:
            self.body.append(declaration)
        # self.body.append("i_0, j_0, k_0 = _origin_['_all_']")
        self.body.append("i_0, j_0, k_0 = 0,0,0")
        self.body.append("i_size, j_size, k_size = _domain_")
        for loop in stencil.vertical_loops:
            for section in loop.sections:
                loop_bounds = self.visit(section.interval, var="k")
                loop_code = "for k in range(" + loop_bounds + "):"
                self.body.append(loop_code)
                self.body.indent()
                for execution in section.horizontal_executions:
                    extents = block_extents[id(execution)]
                    i_loop = f"for i in range(i_0 + {extents[0][0]} , i_size + {extents[0][1]}):"
                    self.body.append(i_loop)
                    self.body.indent()
                    j_loop = f"for j in range(j_0 + {extents[1][0]} , j_size + {extents[1][1]}):"
                    self.body.append(j_loop)
                    self.body.indent()
                    self.visit(execution)
                    self.body.dedent()
                    self.body.dedent()
                self.body.dedent()
        return self.body.text

    def visit_FieldDecl(self, field_decl: FieldDecl, **_):
        return str(field_decl.name)

    def visit_AxisBound(self, axis_bound: AxisBound, **kwargs):
        if axis_bound.level == LevelMarker.START:
            return f"{kwargs['var']}_0 + {axis_bound.offset}"
        if axis_bound.level == LevelMarker.END:
            return f"{kwargs['var']}_size + {axis_bound.offset}"

    def visit_Interval(self, interval: Interval, **kwargs):
        return ",".join([self.visit(interval.start, **kwargs), self.visit(interval.end, **kwargs)])
