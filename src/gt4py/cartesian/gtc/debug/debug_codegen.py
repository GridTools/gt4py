# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py import eve
from gt4py.cartesian import utils
from gt4py.cartesian.gtc.common import (
    AxisBound,
    DataType,
    FieldAccess,
    HorizontalInterval,
    HorizontalMask,
    LevelMarker,
)
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.oir import (
    AssignStmt,
    BinaryOp,
    Cast,
    Decl,
    FieldDecl,
    HorizontalExecution,
    HorizontalRestriction,
    Interval,
    Literal,
    Stencil,
    Temporary,
)
from gt4py.cartesian.gtc.passes.oir_optimizations.utils import StencilExtentComputer
from gt4py.eve import codegen


class DebugCodeGen(codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):
    def __init__(self) -> None:
        self.body = utils.text.TextBlock()

    def visit_VerticalLoop(self):
        pass

    def generate_field_decls(self, declarations: list[Decl]) -> None:
        for declaration in declarations:
            if isinstance(declaration, FieldDecl):
                self.body.append(
                    f"{declaration.name} = Field({declaration.name}, _origin_['{declaration.name}'], "
                    f"({', '.join([str(x) for x in declaration.dimensions])}))"
                )

    def visit_FieldAccess(self, field_access: FieldAccess, **_):
        full_string = field_access.name + "[" + field_access.offset.to_str() + "]"
        return full_string

    def visit_AssignStmt(self, assignment_statement: AssignStmt, **_):
        self.body.append(
            self.visit(assignment_statement.left) + "=" + self.visit(assignment_statement.right)
        )

    def visit_BinaryOp(self, binary: BinaryOp, **_):
        return self.visit(binary.left) + str(binary.op) + self.visit(binary.right)

    def visit_Literal(self, literal: Literal, **_) -> str:
        if literal.dtype.bit_count() != 4:
            literal_code = f"{self.visit(literal.dtype)}({literal.value})"
        else:
            literal_code = str(literal.value)
        return literal_code

    def visit_Cast(self, cast: Cast, **_):
        return self.visit(cast.expr)

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

    @staticmethod
    def compute_extents(node: Stencil, **_) -> tuple[dict[str, Extent], dict[int, Extent]]:
        ctx: StencilExtentComputer.Context = StencilExtentComputer().visit(node)
        return ctx.fields, ctx.blocks

    def generate_temp_decls(
        self, temporary_declarations: list[Temporary], field_extents: dict[str, Extent]
    ):
        for declaration in temporary_declarations:
            self.body.append(self.visit(declaration, field_extents=field_extents))

    def visit_Temporary(self, temporary_declaration: Temporary, **kwargs):
        field_extents = kwargs["field_extents"]
        local_field_extent = field_extents[temporary_declaration.name]
        i_padding: int = local_field_extent[0][1] - local_field_extent[0][0]
        j_padding: int = local_field_extent[1][1] - local_field_extent[1][0]
        shape: list[str] = [f"i_size + {i_padding}", f"j_size + {j_padding}", "k_size"]
        data_dimensions: list[str] = [str(dim) for dim in temporary_declaration.data_dims]
        shape = shape + data_dimensions
        shape_decl = ", ".join(shape)
        dtype = self.visit(temporary_declaration.dtype)
        field_offset = tuple(-ext[0] for ext in local_field_extent)
        offset = [str(off) for off in field_offset] + ["0"] * (
            1 + len(temporary_declaration.data_dims)
        )
        return f"{temporary_declaration.name} = Field.empty(({shape_decl}), {dtype}, ({', '.join(offset)}))"

    def visit_DataType(self, data_type: DataType, **_):
        if data_type not in {DataType.BOOL}:
            return f"np.{data_type.name.lower()}"
        else:
            return data_type.name.lower()

    def visit_Stencil(self, stencil: Stencil, **_):
        field_extents, block_extents = self.compute_extents(stencil)
        self.body.append("from gt4py.cartesian.utils import Field")
        self.body.append("import numpy as np")

        function_signature = "def run(*"
        args = []
        for param in stencil.params:
            args.append(self.visit(param))
        function_signature = ",".join([function_signature, *args])
        function_signature += ",_domain_, _origin_):"
        self.body.append(function_signature)
        self.body.indent()
        self.body.append("# ===== Domain Description ===== #")
        self.body.append("i_0, j_0, k_0 = 0,0,0")
        self.body.append("i_size, j_size, k_size = _domain_")
        self.body.empty_line()
        self.body.append("# ===== Temporary Declaration ===== #")
        self.generate_temp_decls(stencil.declarations, field_extents)
        self.body.empty_line()
        self.body.append("# ===== Field Declaration ===== #")
        self.generate_field_decls(stencil.params)
        self.body.empty_line()

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
