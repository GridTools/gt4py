from typing import Any, List, Optional, Union

import dace
import dace.data
import dace.library
import dace.subsets

import gtc.common as common
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from gtc import daceir as dcir
from gtc.dace.expansion.utils import add_origin
from gtc.dace.utils import get_axis_bound_str


class TaskletCodegen(codegen.TemplatedGenerator):

    ScalarAccess = as_fmt("{name}")

    def _visit_offset(
        self,
        node: Union[dcir.VariableKOffset, common.CartesianOffset],
        *,
        access_info: dcir.FieldAccessInfo,
        decl: dcir.FieldDecl,
        **kwargs,
    ):
        int_sizes: List[Optional[int]] = []
        for i, axis in enumerate(access_info.axes()):
            memlet_shape = access_info.shape
            if (
                str(memlet_shape[i]).isnumeric()
                and axis not in decl.access_info.variable_offset_axes
            ):
                int_sizes.append(int(memlet_shape[i]))
            else:
                int_sizes.append(None)
        str_offset = [
            self.visit(off, **kwargs) for off in (node.to_dict()["i"], node.to_dict()["j"], node.k)
        ]
        str_offset = [
            f"{axis.iteration_symbol()} + {str_offset[axis.to_idx()]}"
            for i, axis in enumerate(access_info.axes())
        ]

        res: dace.subsets.Range = add_origin(
            decl.access_info, ",".join(str_offset), add_for_variable=True
        )
        return str(dace.subsets.Range([r for i, r in enumerate(res.ranges) if int_sizes[i] != 1]))

    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_VariableKOffset(self, node: common.CartesianOffset, **kwargs):
        return self._visit_offset(node, **kwargs)

    def visit_IndexAccess(self, node: dcir.IndexAccess, *, is_target, sdfg_ctx, **kwargs):

        memlets = kwargs["write_memlets" if is_target else "read_memlets"]
        memlet = next(mem for mem in memlets if mem.connector == node.name)

        index_strs = []
        if node.offset is not None:
            index_strs.append(
                self.visit(
                    node.offset,
                    decl=sdfg_ctx.field_decls[memlet.field],
                    access_info=memlet.access_info,
                    **kwargs,
                )
            )
        index_strs.extend(self.visit(idx, sdfg_ctx=sdfg_ctx, **kwargs) for idx in node.data_index)
        return f"{node.name}[{','.join(index_strs)}]"

    def visit_AssignStmt(self, node: dcir.AssignStmt, **kwargs):
        right = self.visit(node.right, is_target=False, **kwargs)
        left = self.visit(node.left, is_target=True, **kwargs)
        return f"{left} = {right}"

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({true_expr} if {cond} else {false_expr})")

    def visit_BuiltInLiteral(self, builtin: common.BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == common.BuiltInLiteral.TRUE:
            return "True"
        elif builtin == common.BuiltInLiteral.FALSE:
            return "False"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_fmt("{value}")

    Cast = as_fmt("{dtype}({expr})")

    def visit_NativeFunction(self, func: common.NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                common.NativeFunction.ABS: "abs",
                common.NativeFunction.MIN: "min",
                common.NativeFunction.MAX: "max",
                common.NativeFunction.MOD: "fmod",
                common.NativeFunction.SIN: "dace.math.sin",
                common.NativeFunction.COS: "dace.math.cos",
                common.NativeFunction.TAN: "dace.math.tan",
                common.NativeFunction.ARCSIN: "asin",
                common.NativeFunction.ARCCOS: "acos",
                common.NativeFunction.ARCTAN: "atan",
                common.NativeFunction.SINH: "dace.math.sinh",
                common.NativeFunction.COSH: "dace.math.cosh",
                common.NativeFunction.TANH: "dace.math.tanh",
                common.NativeFunction.ARCSINH: "asinh",
                common.NativeFunction.ARCCOSH: "acosh",
                common.NativeFunction.ARCTANH: "atanh",
                common.NativeFunction.SQRT: "dace.math.sqrt",
                common.NativeFunction.POW: "dace.math.pow",
                common.NativeFunction.EXP: "dace.math.exp",
                common.NativeFunction.LOG: "dace.math.log",
                common.NativeFunction.GAMMA: "tgamma",
                common.NativeFunction.CBRT: "cbrt",
                common.NativeFunction.ISFINITE: "isfinite",
                common.NativeFunction.ISINF: "isinf",
                common.NativeFunction.ISNAN: "isnan",
                common.NativeFunction.FLOOR: "dace.math.ifloor",
                common.NativeFunction.CEIL: "ceil",
                common.NativeFunction.TRUNC: "trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError("Not implemented NativeFunction encountered.") from error

    def visit_NativeFuncCall(self, call: common.NativeFuncCall, **kwargs: Any) -> str:
        # TODO: Unroll integer POW
        return f"{self.visit(call.func, **kwargs)}({','.join([self.visit(a, **kwargs) for a in call.args])})"

    def visit_DataType(self, dtype: common.DataType, **kwargs: Any) -> str:
        if dtype == common.DataType.BOOL:
            return "dace.bool_"
        elif dtype == common.DataType.INT8:
            return "dace.int8"
        elif dtype == common.DataType.INT16:
            return "dace.int16"
        elif dtype == common.DataType.INT32:
            return "dace.int32"
        elif dtype == common.DataType.INT64:
            return "dace.int64"
        elif dtype == common.DataType.FLOAT32:
            return "dace.float32"
        elif dtype == common.DataType.FLOAT64:
            return "dace.float64"
        raise NotImplementedError("Not implemented DataType encountered.")

    def visit_UnaryOperator(self, op: common.UnaryOperator, **kwargs: Any) -> str:
        if op == common.UnaryOperator.NOT:
            return " not "
        elif op == common.UnaryOperator.NEG:
            return "-"
        elif op == common.UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    LocalScalar = as_fmt("{name}: {dtype}")

    def visit_Tasklet(self, node: dcir.Tasklet, **kwargs):
        return "\n".join(self.visit(node.stmts, **kwargs))

    def _visit_conditional(
        self,
        cond: Optional[Union[dcir.Expr, common.HorizontalMask]],
        body: List[dcir.Stmt],
        keyword,
        **kwargs,
    ):
        mask_str = ""
        indent = ""
        if cond is not None and (cond_str := self.visit(cond, is_target=False, **kwargs)):
            mask_str = f"{keyword} {cond_str}:"
            indent = " " * 4
        body_code = [line for block in self.visit(body, **kwargs) for line in block.split("\n")]
        body_code = [indent + b for b in body_code]
        return "\n".join([mask_str] + body_code)

    def visit_MaskStmt(self, node: dcir.MaskStmt, **kwargs):
        return self._visit_conditional(cond=node.mask, body=node.body, keyword="if", **kwargs)

    def visit_HorizontalRestriction(self, node: dcir.HorizontalRestriction, **kwargs):
        return self._visit_conditional(cond=node.mask, body=node.body, keyword="if", **kwargs)

    def visit_While(self, node: dcir.While, **kwargs):
        return self._visit_conditional(cond=node.cond, body=node.body, keyword="while", **kwargs)

    def visit_HorizontalMask(self, node: common.HorizontalMask, **kwargs):
        clauses: List[str] = []

        for axis, interval in zip(dcir.Axis.dims_horizontal(), node.intervals):
            it_sym, dom_sym = axis.iteration_symbol(), axis.domain_symbol()

            min_val = get_axis_bound_str(interval.start, dom_sym)
            max_val = get_axis_bound_str(interval.end, dom_sym)

            if min_val:
                clauses.append(f"{it_sym} >= {min_val}")
            if max_val:
                clauses.append(f"{it_sym} < {max_val}")

        return " and ".join(clauses)

    @classmethod
    def apply(cls, node: dcir.Tasklet, **kwargs: Any) -> str:
        if not isinstance(node, dcir.Tasklet):
            raise ValueError("apply() requires dcir.Tasklet node")
        generated_code = super().apply(node, **kwargs)
        formatted_code = codegen.format_source("python", generated_code)
        return formatted_code
