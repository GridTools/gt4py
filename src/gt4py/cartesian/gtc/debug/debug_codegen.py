# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Generator
from contextlib import contextmanager

from gt4py import eve
from gt4py.cartesian import utils
from gt4py.cartesian.gtc.common import (
    AxisBound,
    BuiltInLiteral,
    CartesianOffset,
    DataType,
    FieldAccess,
    HorizontalInterval,
    HorizontalMask,
    LevelMarker,
    LoopOrder,
    While,
)
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.oir import (
    AssignStmt,
    BinaryOp,
    CacheDesc,
    Cast,
    Decl,
    FieldDecl,
    HorizontalExecution,
    HorizontalRestriction,
    IJCache,
    Interval,
    KCache,
    Literal,
    LocalScalar,
    MaskStmt,
    NativeFuncCall,
    ScalarAccess,
    ScalarDecl,
    Stencil,
    Temporary,
    TernaryOp,
    UnaryOp,
    UnboundedInterval,
    VariableKOffset,
    VerticalLoop,
    VerticalLoopSection,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import StencilExtentComputer
from gt4py.eve import codegen
from gt4py.eve.concepts import SymbolRef


class DebugCodeGen(codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):
    def __init__(self) -> None:
        self.body = utils.text.TextBlock()
        self.symbol_table: dict[str, FieldDecl] = {}

    def visit_Stencil(self, stencil: Stencil, **_):
        self.generate_imports()

        self.generate_run_function(stencil)

        field_extents, block_extents = self.compute_extents(stencil)
        self.symbol_table = self.initial_declarations(stencil, field_extents)
        self.generate_stencil_code(stencil, block_extents)

        return self.body.text

    def generate_imports(self):
        self.body.append("import numpy as np")
        self.body.append("from gt4py.cartesian.gtc import ufuncs")
        self.body.append("from gt4py.cartesian.utils import Field")

    @staticmethod
    def compute_extents(node: Stencil, **_) -> tuple[dict[str, Extent], dict[int, Extent]]:
        ctx: StencilExtentComputer.Context = StencilExtentComputer().visit(node)
        return ctx.fields, ctx.blocks

    def initial_declarations(
        self, stencil: Stencil, field_extents: dict[str, Extent]
    ) -> dict[str, FieldDecl]:
        self.body.append("# ===== Domain Description ===== #")
        self.body.append("i_0, j_0, k_0 = 0,0,0")
        self.body.append("i_size, j_size, k_size = _domain_")
        self.body.empty_line()
        self.body.append("# ===== Temporary Declaration ===== #")
        symbol_table = self.generate_temp_decls(stencil.declarations, field_extents)
        self.body.empty_line()
        self.body.append("# ===== Field Declaration ===== #")
        symbol_table |= self.generate_field_decls(stencil.params)
        self.body.empty_line()
        return symbol_table

    def generate_temp_decls(
        self, temporary_declarations: list[Temporary], field_extents: dict[str, Extent]
    ) -> dict[str, FieldDecl]:
        symbol_table: dict[str, FieldDecl] = {}
        for declaration in temporary_declarations:
            self.body.append(self.visit(declaration, field_extents=field_extents))
            symbol_table[str(declaration.name)] = declaration
        return symbol_table

    def generate_field_decls(self, declarations: list[Decl]) -> dict[str, FieldDecl]:
        symbol_table = {}
        for declaration in declarations:
            if isinstance(declaration, FieldDecl):
                self.body.append(
                    f"{declaration.name} = Field({declaration.name}, _origin_['{declaration.name}'], "
                    f"({', '.join([str(x) for x in declaration.dimensions])}))"
                )
                symbol_table[str(declaration.name)] = declaration
        return symbol_table

    def generate_run_function(self, stencil: Stencil):
        function_signature = "def run(*"
        args = []
        for param in stencil.params:
            args.append(self.visit(param))
        function_signature = ",".join([function_signature, *args])
        function_signature += ",_domain_, _origin_):"
        self.body.append(function_signature)
        self.body.indent()

    def generate_stencil_code(
        self,
        stencil: Stencil,
        block_extents: dict[int, Extent],
    ):
        for loop in stencil.vertical_loops:
            for section in loop.sections:
                with self.create_k_loop_code(section, loop):
                    for execution in section.horizontal_executions:
                        with self.generate_ij_loop(block_extents, execution):
                            self.visit(execution)

    @contextmanager
    def create_k_loop_code(self, section: VerticalLoopSection, loop: VerticalLoop) -> Generator:
        loop_bounds: str = self.visit(section.interval, var="k", direction=loop.loop_order)
        iterator = "1" if loop.loop_order != LoopOrder.BACKWARD else "-1"
        loop_code = "for k in range(" + loop_bounds + "," + iterator + "):"
        self.body.append(loop_code)
        self.body.indent()
        yield
        self.body.dedent()

    @contextmanager
    def generate_ij_loop(
        self, block_extents: dict[int, Extent], execution: HorizontalExecution
    ) -> Generator:
        extents = block_extents[id(execution)]
        i_loop = f"for i in range(i_0 + {extents[0][0]} , i_size + {extents[0][1]}):"
        self.body.append(i_loop)
        self.body.indent()
        j_loop = f"for j in range(j_0 + {extents[1][0]} , j_size + {extents[1][1]}):"
        self.body.append(j_loop)
        self.body.indent()
        yield
        self.body.dedent()
        self.body.dedent()

    def visit_While(self, while_node: While, **_) -> None:
        while_condition = self.visit(while_node.cond)
        while_code = f"while {while_condition}:"
        self.body.append(while_code)
        self.body.indent()
        for statement in while_node.body:
            self.visit(statement)
        self.body.dedent()

    def visit_FieldDecl(self, field_decl: FieldDecl, **_) -> str:
        return str(field_decl.name)

    def visit_AxisBound(self, axis_bound: AxisBound, **kwargs):
        if axis_bound.level == LevelMarker.START:
            return f"{kwargs['var']}_0 + {axis_bound.offset}"
        if axis_bound.level == LevelMarker.END:
            return f"{kwargs['var']}_size + {axis_bound.offset}"

    def visit_Interval(self, interval: Interval, **kwargs):
        if kwargs["direction"] == LoopOrder.BACKWARD:
            return ",".join(
                [
                    self.visit(interval.end, **kwargs) + "- 1",
                    self.visit(interval.start, **kwargs) + "- 1",
                ]
            )
        else:
            return ",".join(
                [self.visit(interval.start, **kwargs), self.visit(interval.end, **kwargs)]
            )

    def visit_Temporary(self, temporary_declaration: Temporary, **kwargs) -> str:
        field_extents = kwargs["field_extents"]
        local_field_extent = field_extents[temporary_declaration.name]
        i_padding: int = local_field_extent[0][1] - local_field_extent[0][0]
        j_padding: int = local_field_extent[1][1] - local_field_extent[1][0]
        shape: list[str] = [f"i_size + {i_padding}", f"j_size + {j_padding}", "k_size"]
        data_dimensions: list[str] = [str(dim) for dim in temporary_declaration.data_dims]
        shape = shape + data_dimensions
        shape_decl = ", ".join(shape)
        dtype: str = self.visit(temporary_declaration.dtype)
        field_offset = tuple(-ext[0] for ext in local_field_extent)
        offset = [str(off) for off in field_offset] + ["0"] * (
            1 + len(temporary_declaration.data_dims)
        )
        return f"{temporary_declaration.name} = Field.empty(({shape_decl}), {dtype}, ({', '.join(offset)}))"

    def visit_DataType(self, data_type: DataType, **_) -> str:
        if data_type not in {DataType.BOOL}:
            return f"np.{data_type.name.lower()}"
        else:
            return data_type.name.lower()

    def visit_VariableKOffset(self, variable_k_offset: VariableKOffset, **_) -> str:
        return f"i,j,k+int({self.visit(variable_k_offset.k)})"

    def visit_CartesianOffset(self, cartesian_offset: CartesianOffset, **kwargs) -> str:
        dimensions = kwargs["dimensions"]
        return cartesian_offset.to_str(dimensions)

    def visit_SymbolRef(self, symbol_ref: SymbolRef) -> str:
        return symbol_ref

    def visit_FieldAccess(self, field_access: FieldAccess, **_) -> str:
        if str(field_access.name) in self.symbol_table:
            dimensions = self.symbol_table[str(field_access.name)].dimensions
        else:
            dimensions = (True, True, True)

        offset_str = self.visit(field_access.offset, dimensions=dimensions)

        if field_access.data_index:
            data_index_access = ",".join(
                [self.visit(data_index) for data_index in field_access.data_index]
            )
            if offset_str == "":
                full_string = field_access.name + f"[{data_index_access}]"
            else:
                full_string = field_access.name + "[" + offset_str + "," + data_index_access + "]"
        else:
            full_string = field_access.name + "[" + offset_str + "]"
        return full_string

    def visit_AssignStmt(self, assignment_statement: AssignStmt, **_):
        self.body.append(
            self.visit(assignment_statement.left) + "=" + self.visit(assignment_statement.right)
        )

    def visit_BinaryOp(self, binary: BinaryOp, **_) -> str:
        return f"( {self.visit(binary.left)} {binary.op} {self.visit(binary.right)} )"

    def visit_Literal(self, literal: Literal, **_) -> str:
        if literal.dtype == DataType.BOOL:
            if literal.value == BuiltInLiteral.TRUE:
                literal_value = "True"
            else:
                literal_value = "False"
        else:
            literal_value = str(literal.value)

        if literal.dtype.bit_count() != 4:
            literal_code = f"{self.visit(literal.dtype)}({literal_value})"
        else:
            literal_code = literal_value
        return literal_code

    def visit_Cast(self, cast: Cast, **_) -> str:
        return f"{self.visit(cast.dtype)}({self.visit(cast.expr)})"

    def visit_HorizontalExecution(self, horizontal_execution: HorizontalExecution, **_):
        for stmt in horizontal_execution.body:
            self.visit(stmt)

    def visit_HorizontalMask(self, horizontal_mask: HorizontalMask, **_):
        i_min, i_max = self.visit(horizontal_mask.i, var="i")
        j_min, j_max = self.visit(horizontal_mask.j, var="j")
        conditions = []
        if i_min is not None:
            conditions.append(f"({i_min}) <= i")
        if i_max is not None:
            conditions.append(f"i < ({i_max})")
        if j_min is not None:
            conditions.append(f"({j_min}) <= j")
        if j_max is not None:
            conditions.append(f"j < ({j_max})")
        assert len(conditions)
        if_code = f"if( {' and '.join(conditions)} ):"
        self.body.append(if_code)

    def visit_HorizontalInterval(self, horizontal_interval: HorizontalInterval, **kwargs):
        return self.visit(
            horizontal_interval.start, **kwargs
        ) if horizontal_interval.start else None, self.visit(
            horizontal_interval.end, **kwargs
        ) if horizontal_interval.end else None

    def visit_HorizontalRestriction(self, horizontal_restriction: HorizontalRestriction, **_):
        self.visit(horizontal_restriction.mask)
        self.body.indent()
        self.visit(horizontal_restriction.body)
        self.body.dedent()

    def visit_VerticalLoop(self):
        pass

    def visit_ScalarAccess(self, scalar_access: ScalarAccess, **_):
        return scalar_access.name

    def visit_ScalarDecl(self, scalar_declaration: ScalarDecl, **_) -> str:
        return scalar_declaration.name

    def visit_NativeFuncCall(self, native_function_call: NativeFuncCall, **_) -> str:
        arglist = [self.visit(arg) for arg in native_function_call.args]
        arguments = ",".join(arglist)
        return f"ufuncs.{native_function_call.func.value}({arguments})"

    def visit_UnaryOp(self, unary_operator: UnaryOp, **_) -> str:
        return unary_operator.op.value + " " + self.visit(unary_operator.expr)

    def visit_TernaryOp(self, ternary_operator: TernaryOp, **_) -> str:
        return f"{self.visit(ternary_operator.true_expr)} if {self.visit(ternary_operator.cond)} else {self.visit(ternary_operator.false_expr)}"

    def visit_MaskStmt(self, mask_statement: MaskStmt, **_):
        self.body.append(f"if {self.visit(mask_statement.mask)}:")
        with self.body.indented():
            for statement in mask_statement.body:
                self.visit(statement)

    def visit_LocalScalar(self, local_scalar: LocalScalar, **__) -> None:
        raise NotImplementedError(
            "This state should not be reached because LocalTemporariesToScalars should not have been called."
        )

    def visit_CacheDesc(self, cache_descriptor: CacheDesc, **_):
        raise NotImplementedError("Caches should never be visited in the debug backend")

    def visit_IJCache(self, ij_cache: IJCache, **_):
        raise NotImplementedError("Caches should never be visited in the debug backend")

    def visit_KCache(self, k_cache: KCache, **_):
        raise NotImplementedError("Caches should never be visited in the debug backend")

    def visit_VerticalLoopSection(self, vertical_loop_section: VerticalLoopSection, **_):
        raise NotImplementedError("Vertical Loop section is not in the right place.")

    def visit_UnboundedInterval(self, unbounded_interval: UnboundedInterval, **_) -> None:
        raise NotImplementedError("Unbounded Intervals are not supported in the debug backend.")
