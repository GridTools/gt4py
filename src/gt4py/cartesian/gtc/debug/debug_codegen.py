# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Generator
from contextlib import contextmanager
from typing import Mapping

from gt4py import eve
from gt4py.cartesian.gtc import common as gtc_common, definitions as gtc_definitions, oir
from gt4py.cartesian.gtc.passes.oir_optimizations import utils as optimization_utils
from gt4py.eve import codegen, concepts


def compute_extents(
    node: oir.Stencil,
) -> tuple[dict[str, gtc_definitions.Extent], dict[int, gtc_definitions.Extent]]:
    ctx: optimization_utils.StencilExtentComputer.Context = (
        optimization_utils.StencilExtentComputer().visit(node)
    )
    return ctx.fields, ctx.blocks


class DebugCodeGen(eve.VisitorWithSymbolTableTrait):
    def __init__(self) -> None:
        self.body = codegen.TextBlock()

    def visit_Stencil(
        self, stencil: oir.Stencil, symtable: Mapping[str, oir.FieldDecl], **_
    ) -> str:
        field_extents, block_extents = compute_extents(stencil)

        self.generate_imports()
        self.generate_run_function(stencil)
        self.initial_declarations(stencil, field_extents)

        self.generate_stencil_code(stencil, block_extents, symtable)

        return self.body.text

    def generate_imports(self) -> None:
        self.body.append("import numpy as np")
        self.body.empty_line()
        self.body.append("from gt4py.cartesian.gtc import ufuncs")
        self.body.append("from gt4py.cartesian.utils import Field")

    def generate_run_function(self, stencil: oir.Stencil) -> None:
        function_signature = "def run(*"
        args = []
        for param in stencil.params:
            args.append(self.visit(param))
        function_signature = ",".join([function_signature, *args])
        function_signature += ",_domain_, _origin_):"
        self.body.append(function_signature)
        self.body.indent()

    def initial_declarations(
        self, stencil: oir.Stencil, field_extents: dict[str, gtc_definitions.Extent]
    ) -> None:
        self.body.append("# ===== Domain Description ===== #")
        self.body.append("i_0, j_0, k_0 = 0, 0, 0")
        self.body.append("i_size, j_size, k_size = _domain_")
        self.body.empty_line()
        self.body.append("# ===== Temporary Declaration ===== #")
        self.generate_temp_decls(stencil.declarations, field_extents)
        self.body.empty_line()
        self.body.append("# ===== Field Declaration ===== #")
        self.generate_field_decls(stencil.params)
        self.body.empty_line()

    def generate_temp_decls(
        self,
        temporary_declarations: list[oir.Temporary],
        field_extents: dict[str, gtc_definitions.Extent],
    ) -> None:
        for declaration in temporary_declarations:
            self.body.append(self.visit(declaration, field_extents=field_extents))

    def generate_field_decls(self, declarations: list[oir.Decl]) -> None:
        for declaration in declarations:
            if isinstance(declaration, oir.FieldDecl):
                self.body.append(
                    f"{declaration.name} = Field({declaration.name}, _origin_['{declaration.name}'], "
                    f"({', '.join([str(x) for x in declaration.dimensions])}))"
                )

    # The visitors for all the control flow code that write directly to the `self.body` are in the following section
    def generate_stencil_code(
        self,
        stencil: oir.Stencil,
        block_extents: dict[int, gtc_definitions.Extent],
        symtable: Mapping[str, oir.FieldDecl],
    ) -> None:
        for loop in stencil.vertical_loops:
            for section in loop.sections:
                with self.create_k_loop_code(section, loop):
                    for execution in section.horizontal_executions:
                        with self.generate_ij_loop(block_extents, execution):
                            self.visit(execution, symtable=symtable)

    @contextmanager
    def create_k_loop_code(
        self, section: oir.VerticalLoopSection, loop: oir.VerticalLoop
    ) -> Generator:
        loop_bounds: str = self.visit(section.interval, var="k", direction=loop.loop_order)
        increment = "-1" if loop.loop_order == gtc_common.LoopOrder.BACKWARD else "1"
        loop_code = f"for k in range( {loop_bounds} , {increment}):"
        self.body.append(loop_code)
        self.body.indent()
        yield
        self.body.dedent()

    @contextmanager
    def generate_ij_loop(
        self, block_extents: dict[int, gtc_definitions.Extent], execution: oir.HorizontalExecution
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

    def visit_While(self, while_node: gtc_common.While, **_) -> None:
        while_condition = self.visit(while_node.cond)
        while_code = f"while {while_condition}:"
        self.body.append(while_code)
        with self.body.indented():
            for statement in while_node.body:
                self.visit(statement)

    def visit_HorizontalExecution(
        self, horizontal_execution: oir.HorizontalExecution, **kwargs
    ) -> None:
        for statement in horizontal_execution.body:
            self.visit(statement, **kwargs)

    def visit_HorizontalMask(self, horizontal_mask: gtc_common.HorizontalMask, **kwargs) -> None:
        i_min, i_max = self.visit(horizontal_mask.i, var="i", **kwargs)
        j_min, j_max = self.visit(horizontal_mask.j, var="j", **kwargs)
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

    def visit_HorizontalRestriction(
        self, horizontal_restriction: oir.HorizontalRestriction, **kwargs
    ) -> None:
        self.visit(horizontal_restriction.mask, **kwargs)
        with self.body.indented():
            self.visit(horizontal_restriction.body, **kwargs)

    def visit_MaskStmt(self, mask_statement: oir.MaskStmt, **kwargs) -> None:
        self.body.append(f"if {self.visit(mask_statement.mask, **kwargs)}:")
        with self.body.indented():
            for statement in mask_statement.body:
                self.visit(statement, **kwargs)

    def visit_AssignStmt(self, assignment_statement: oir.AssignStmt, **kwargs) -> None:
        self.body.append(
            f"{self.visit(assignment_statement.left, **kwargs)} = {self.visit(assignment_statement.right, **kwargs)}"
        )

    # The visitors for the rest of the code-generation all return their strings directly and are in the following section
    def visit_VerticalLoop(self) -> None:
        pass

    def visit_FieldDecl(self, field_decl: oir.FieldDecl, **_) -> str:
        return str(field_decl.name)

    def visit_AxisBound(self, axis_bound: gtc_common.AxisBound, **kwargs) -> str:
        if axis_bound.level == gtc_common.LevelMarker.START:
            return f"{kwargs['var']}_0 + {axis_bound.offset}"
        if axis_bound.level == gtc_common.LevelMarker.END:
            return f"{kwargs['var']}_size + {axis_bound.offset}"

    def visit_Interval(self, interval: oir.Interval, **kwargs) -> str:
        if kwargs["direction"] == gtc_common.LoopOrder.BACKWARD:
            return ",".join(
                [
                    f"{self.visit(interval.end, **kwargs)} - 1",
                    f"{self.visit(interval.start, **kwargs)} - 1",
                ]
            )
        return ",".join([self.visit(interval.start, **kwargs), self.visit(interval.end, **kwargs)])

    def visit_Temporary(self, temporary_declaration: oir.Temporary, **kwargs) -> str:
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

    def visit_DataType(self, data_type: gtc_common.DataType, **_) -> str:
        if data_type in {gtc_common.DataType.BOOL}:
            return data_type.name.lower()
        return f"np.{data_type.name.lower()}"

    def visit_VariableKOffset(self, variable_k_offset: oir.VariableKOffset, **kwargs) -> str:
        return f"i,j,k+int({self.visit(variable_k_offset.k, **kwargs)})"

    def visit_CartesianOffset(self, cartesian_offset: gtc_common.CartesianOffset, **kwargs) -> str:
        if "dimensions" in kwargs.keys():
            dimensions = kwargs["dimensions"]
            dimension_strings = []
            if dimensions[0]:
                dimension_strings.append(f"i + {cartesian_offset.i}")
            if dimensions[1]:
                dimension_strings.append(f"j + {cartesian_offset.j}")
            if dimensions[2]:
                dimension_strings.append(f"k + {cartesian_offset.k}")
            return ",".join(dimension_strings)
        return f"i + {cartesian_offset.i}, j + {cartesian_offset.j}, k + {cartesian_offset.k}"

    def visit_SymbolRef(self, symbol_ref: concepts.SymbolRef) -> str:
        return symbol_ref

    def visit_FieldAccess(
        self,
        field_access: gtc_common.FieldAccess,
        *,
        symtable: Mapping[str, oir.FieldDecl],
        **kwargs,
    ) -> str:
        if str(field_access.name) in symtable:
            dimensions = symtable[str(field_access.name)].dimensions
            kwargs.pop("dimensions", None)
            offset_str = self.visit(
                field_access.offset, dimensions=dimensions, symtable=symtable, **kwargs
            )
        else:
            offset_str = self.visit(field_access.offset, symtable=symtable, **kwargs)

        if field_access.data_index:
            data_index_access = ",".join(
                [
                    self.visit(data_index, symtable=symtable, **kwargs)
                    for data_index in field_access.data_index
                ]
            )
            if offset_str == "":
                return f"{field_access.name}[{data_index_access}]"
            return f"{field_access.name}[{offset_str},{data_index_access}]"
        return f"{field_access.name}[{offset_str}]"

    def visit_AbsoluteKIndex(self, absolute_k_index: oir.AbsoluteKIndex, **kwargs) -> str:
        access_pattern = []
        if kwargs["dimensions"][2] is False:
            raise ValueError(
                "Tried accessing a field with no K-dimensions with an absolute K-index."
            )
        if kwargs["dimensions"][0]:
            access_pattern.append("i")
        if kwargs["dimensions"][1]:
            access_pattern.append("j")
        access_pattern.append(f"int({self.visit(absolute_k_index.k, **kwargs)})")
        return ",".join(access_pattern)

    def visit_BinaryOp(self, binary: oir.BinaryOp, **kwargs) -> str:
        return f"( {self.visit(binary.left, **kwargs)} {binary.op} {self.visit(binary.right, **kwargs)} )"

    def visit_Literal(self, literal: oir.Literal, **_) -> str:
        if literal.dtype == gtc_common.DataType.BOOL:
            literal_value = "True" if literal.value == gtc_common.BuiltInLiteral.TRUE else "False"
        else:
            literal_value = str(literal.value)

        if literal.dtype.bit_count() != 4:
            return f"{self.visit(literal.dtype)}({literal_value})"

        return literal_value

    def visit_Cast(self, cast: oir.Cast, **kwargs) -> str:
        return f"{self.visit(cast.dtype, **kwargs)}({self.visit(cast.expr, **kwargs)})"

    def visit_ScalarAccess(self, scalar_access: oir.ScalarAccess, **_) -> str:
        return scalar_access.name

    def visit_ScalarDecl(self, scalar_declaration: oir.ScalarDecl, **_) -> str:
        return scalar_declaration.name

    def visit_HorizontalInterval(
        self, horizontal_interval: gtc_common.HorizontalInterval, **kwargs
    ) -> tuple[str | None, str | None]:
        return self.visit(
            horizontal_interval.start, **kwargs
        ) if horizontal_interval.start else None, self.visit(
            horizontal_interval.end, **kwargs
        ) if horizontal_interval.end else None

    def visit_NativeFuncCall(self, native_function_call: oir.NativeFuncCall, **kwargs) -> str:
        arglist = [self.visit(arg, **kwargs) for arg in native_function_call.args]
        arguments = ",".join(arglist)
        function = gtc_common.OP_TO_UFUNC_NAME[gtc_common.NativeFunction][native_function_call.func]
        return f"ufuncs.{function}({arguments})"

    def visit_UnaryOp(self, unary_operator: oir.UnaryOp, **kwargs) -> str:
        return f"{unary_operator.op.value} {self.visit(unary_operator.expr, **kwargs)}"

    def visit_TernaryOp(self, ternary_operator: oir.TernaryOp, **kwargs) -> str:
        return f"{self.visit(ternary_operator.true_expr, **kwargs)} if {self.visit(ternary_operator.cond, **kwargs)} else {self.visit(ternary_operator.false_expr, **kwargs)}"

    # The visitors that are blacklisted for all the things that should not occur are here
    def visit_LocalScalar(self, local_scalar: oir.LocalScalar, **__) -> None:
        raise NotImplementedError(
            "This state should not be reached because LocalTemporariesToScalars should not have been called."
        )

    def visit_CacheDesc(self, cache_descriptor: oir.CacheDesc, **_) -> None:
        raise NotImplementedError("CacheDescriptors should never be visited in the debug backends")

    def visit_IJCache(self, ij_cache: oir.IJCache, **_) -> None:
        raise NotImplementedError("IJCaches should never be visited in the debug backend.")

    def visit_KCache(self, k_cache: oir.KCache, **_) -> None:
        raise NotImplementedError("KCaches should never be visited in the debug backend.")

    def visit_VerticalLoopSection(
        self, vertical_loop_section: oir.VerticalLoopSection, **_
    ) -> None:
        raise NotImplementedError("Vertical Loop section is not in the right place.")

    def visit_UnboundedInterval(self, unbounded_interval: oir.UnboundedInterval, **_) -> None:
        raise NotImplementedError("Unbounded Intervals are not supported in the debug backend.")
