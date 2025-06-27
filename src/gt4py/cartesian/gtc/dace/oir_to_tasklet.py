# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import operator
from dataclasses import dataclass
from functools import reduce
from typing import Any, Final

from dace import Memlet, subsets

from gt4py import eve
from gt4py.cartesian.gtc import common, oir
from gt4py.cartesian.gtc.dace import treeir as tir, utils


# Tasklet in/out connector prefixes
TASKLET_IN: Final[str] = "gtIN_"
TASKLET_OUT: Final[str] = "gtOUT_"


@dataclass
class Context:
    code: list[str]
    """Tasklet code, line by line."""

    targets: set[str]
    """Names of fields / scalars that we've already written to. Used for read-after-write analysis."""

    inputs: dict[str, Memlet]
    """Mapping connector names to memlets flowing into the Tasklet."""
    outputs: dict[str, Memlet]
    """Mapping connector names to memlets flowing out of the Tasklet."""

    tree: tir.TreeRoot
    """Schedule tree in which this Tasklet will be inserted."""


class OIRToTasklet(eve.NodeVisitor):
    """Translate the numerical code from OIR to DaCe Tasklets.

    This visitor should neither attempt transformations nor do any control flow
    work. Control flow is the responsibility of OIRToTreeIR.
    """

    def visit_CodeBlock(
        self, node: oir.CodeBlock, root: tir.TreeRoot
    ) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
        """Entry point to gather all code, inputs and outputs"""
        ctx = Context(code=[], targets=set(), inputs={}, outputs={}, tree=root)

        self.visit(node.body, ctx=ctx)

        return ("\n".join(ctx.code), ctx.inputs, ctx.outputs)

    def visit_ScalarAccess(self, node: oir.ScalarAccess, ctx: Context, is_target: bool) -> str:
        target = is_target or node.name in ctx.targets
        tasklet_name = _tasklet_name(node, target)

        if (
            node.name in ctx.targets  # (read or write) after write
            or tasklet_name in ctx.inputs  # read after read
        ):
            return tasklet_name

        memlet = Memlet(data=node.name, subset=subsets.Range([(0, 0, 1)]))
        if is_target:
            # Nnote: it doesn't matter if we use is_target or target here because if they
            # were different, we had a read-after-write situation, which was already
            # handled above.
            ctx.targets.add(node.name)
            ctx.outputs[tasklet_name] = memlet
        else:
            ctx.inputs[tasklet_name] = memlet

        return tasklet_name

    def visit_FieldAccess(self, node: oir.FieldAccess, ctx: Context, is_target: bool) -> str:
        # Derive tasklet name of this access
        postfix = _field_offset_postfix(node)
        key = f"{node.name}_{postfix}"
        target = is_target or key in ctx.targets
        tasklet_name = _tasklet_name(node, target, postfix)

        # Gather all parts of the variable name in this list
        name_parts = [tasklet_name]

        # Variable K offset subscript
        if isinstance(node.offset, oir.VariableKOffset):
            symbol = tir.Axis.K.iteration_dace_symbol()
            shift = ctx.tree.shift[node.name][tir.Axis.K]
            offset = self.visit(node.offset.k, ctx=ctx, is_target=False)
            name_parts.append(f"[{symbol} + {shift} + {offset}]")

        # Data dimension subscript
        data_indices: list[str] = []
        for index in node.data_index:
            data_indices.append(self.visit(index, ctx=ctx, is_target=False))

        if data_indices:
            name_parts.append(f"[{', '.join(data_indices)}]")

        # In case this is the second access (inside the same tasklet), we can just return the
        # name and don't have to build a Memlet anymore.
        if (
            key in ctx.targets  # (read or write) after write
            or tasklet_name in ctx.inputs  # read after read
        ):
            return "".join(filter(None, name_parts))

        # Build Memlet and add it to inputs/outputs
        data_domains: list[int] = (
            ctx.tree.containers[node.name].shape[-len(node.data_index) :] if node.data_index else []
        )
        memlet = Memlet(
            data=node.name,
            subset=_memlet_subset(node, data_domains, ctx),
            volume=reduce(operator.mul, data_domains, 1),
        )
        if is_target:
            # Note: it doesn't matter if we use is_target or target here because if they
            # were different, we had a read-after-write situation, which was already
            # handled above.
            ctx.targets.add(key)
            ctx.outputs[tasklet_name] = memlet
        else:
            ctx.inputs[tasklet_name] = memlet

        return "".join(filter(None, name_parts))

    def visit_AssignStmt(self, node: oir.AssignStmt, ctx: Context) -> None:
        # Order matters: always evaluate the right side of an assignment first
        right = self.visit(node.right, ctx=ctx, is_target=False)
        left = self.visit(node.left, ctx=ctx, is_target=True)

        ctx.code.append(f"{left} = {right}")

    def visit_TernaryOp(self, node: oir.TernaryOp, **kwargs: Any) -> str:
        condition = self.visit(node.cond, **kwargs)
        if_code = self.visit(node.true_expr, **kwargs)
        else_code = self.visit(node.false_expr, **kwargs)

        return f"({if_code} if {condition} else {else_code})"

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> str:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)

        return f"({left} {node.op.value} {right})"

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> str:
        expr = self.visit(node.expr, **kwargs)

        return f"{node.op.value}({expr})"

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> str:
        dtype = utils.data_type_to_dace_typeclass(node.dtype)
        expression = self.visit(node.expr, **kwargs)

        return f"{dtype}({expression})"

    def visit_Literal(self, node: oir.Literal, **kwargs: Any) -> str:
        if type(node.value) is str:
            # Note: isinstance(node.value, str) also matches the string enum `BuiltInLiteral`
            # which we don't want to match because it returns lower-case `true`, which isn't
            # defined in (python) tasklet code.
            return node.value

        return self.visit(node.value, **kwargs)

    def visit_BuiltInLiteral(self, node: common.BuiltInLiteral, **_kwargs: Any) -> str:
        if node == common.BuiltInLiteral.TRUE:
            return "True"

        if node == common.BuiltInLiteral.FALSE:
            return "False"

        raise NotImplementedError(f"BuiltInLiteral '{node}' not (yet) implemented.")

    def visit_NativeFunction(self, node: common.NativeFunction, **_kwargs: Any) -> str:
        native_functions = {
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
            common.NativeFunction.LOG10: "log10",
            common.NativeFunction.GAMMA: "tgamma",
            common.NativeFunction.CBRT: "cbrt",
            common.NativeFunction.ISFINITE: "isfinite",
            common.NativeFunction.ISINF: "isinf",
            common.NativeFunction.ISNAN: "isnan",
            common.NativeFunction.FLOOR: "dace.math.ifloor",
            common.NativeFunction.CEIL: "ceil",
            common.NativeFunction.TRUNC: "trunc",
        }
        if node not in native_functions:
            raise NotImplementedError(f"NativeFunction '{node}' not (yet) implemented.")

        return native_functions[node]

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> str:
        function_name = self.visit(node.func, **kwargs)
        arguments = ",".join([self.visit(a, **kwargs) for a in node.args])

        return f"{function_name}({arguments})"

    # Not (yet) supported section
    def visit_CacheDesc(self, node: oir.CacheDesc, **kwargs: Any) -> None:
        raise NotImplementedError("To be implemented: Caches")

    def visit_IJCache(self, node: oir.IJCache, **kwargs: Any) -> None:
        raise NotImplementedError("To be implemented: Caches")

    def visit_KCache(self, node: oir.KCache, **kwargs: Any) -> None:
        raise NotImplementedError("To be implemented: Caches")

    # Should _not_ be called
    def visit_CartesianOffset(self, node: common.CartesianOffset, **kwargs: Any) -> None:
        raise RuntimeError("Cartesian Offset should be dealt in Access IRs.")

    def visit_VariableKOffset(self, node: oir.VariableKOffset, **kwargs: Any) -> None:
        raise RuntimeError("Variable K Offset should be dealt in Access IRs.")

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> None:
        raise RuntimeError("visit_MaskStmt should not be called")

    def visit_While(self, node: oir.While, **kwargs: Any) -> None:
        raise RuntimeError("visit_While should not be called")

    def visit_HorizontalRestriction(self, node: oir.HorizontalRestriction, **kwargs: Any) -> None:
        raise RuntimeError("visit_HorizontalRestriction: should be dealt in TreeIR")

    def visit_LocalScalar(self, node: oir.LocalScalar, **kwargs: Any) -> None:
        raise RuntimeError("visit_LocalScalar should not be called")

    def visit_Temporary(self, node: oir.Temporary, **kwargs: Any) -> None:
        raise RuntimeError("visit_LocalScalar should not be called")

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> None:
        raise RuntimeError("visit_Stencil should not be called")

    def visit_Decl(self, node: oir.Decl, **kwargs: Any) -> None:
        raise RuntimeError("visit_Decl should not be called")

    def visit_FieldDecl(self, node: oir.FieldDecl, **kwargs: Any) -> None:
        raise RuntimeError("visit_FieldDecl should not be called")

    def visit_ScalarDecl(self, node: oir.ScalarDecl, **kwargs: Any) -> None:
        raise RuntimeError("visit_ScalarDecl should not be called")

    def visit_Interval(self, node: oir.Interval, **kwargs: Any) -> None:
        raise RuntimeError("visit_Interval should not be called")

    def visit_UnboundedInterval(self, node: oir.UnboundedInterval, **kwargs: Any) -> None:
        raise RuntimeError("visit_UnboundedInterval should not be called")

    def visit_HorizontalExecution(self, node: oir.HorizontalExecution, **kwargs: Any) -> None:
        raise RuntimeError("visit_HorizontalExecution should not be called")

    def visit_VerticalLoop(self, node: oir.VerticalLoop, **kwargs: Any) -> None:
        raise RuntimeError("visit_VerticalLoop should not be called")

    def visit_VerticalLoopSection(self, node: oir.VerticalLoopSection, **kwargs: Any) -> None:
        raise RuntimeError("visit_VerticalLoopSection should not be called")


def generate(
    node: oir.CodeBlock, tree: tir.TreeRoot
) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
    # This function is basically only here to be able to easily type-hint the
    # return value of the CodeBlock visitor.
    # (OIRToTasklet().visit(node) returns an Any type, which looks ugly in the
    #  CodeBlock visitor of OIRToTreeIR)
    return OIRToTasklet().visit(node, root=tree)


def _tasklet_name(
    node: oir.FieldAccess | oir.ScalarAccess, is_target: bool, postfix: str = ""
) -> str:
    name_prefix = TASKLET_OUT if is_target else TASKLET_IN
    return "_".join(filter(None, [name_prefix, node.name, postfix]))


def _field_offset_postfix(node: oir.FieldAccess) -> str:
    if isinstance(node.offset, oir.VariableKOffset):
        return "var_k"

    offset_indicators = [
        f"{k}{'p' if v > 0 else 'm'}{abs(v)}" for k, v in node.offset.to_dict().items() if v != 0
    ]
    return "_".join(offset_indicators)


def _memlet_subset(node: oir.FieldAccess, data_domains: list[int], ctx: Context) -> subsets.Subset:
    if isinstance(node.offset, common.CartesianOffset):
        return _memlet_subset_cartesian(node, data_domains, ctx)

    if isinstance(node.offset, oir.VariableKOffset):
        return _memlet_subset_variable_offset(node, data_domains, ctx)

    raise NotImplementedError(f"_memlet_subset(): unknown offset type {type(node.offset)}")


def _memlet_subset_cartesian(
    node: oir.FieldAccess, data_domains: list[int], ctx: Context
) -> subsets.Subset:
    offset_dict = node.offset.to_dict()
    dimensions = ctx.tree.dimensions[node.name]
    shift = ctx.tree.shift[node.name]

    ranges: list[tuple[str, str, int]] = []
    # Handle cartesian indices
    for index, axis in enumerate(tir.Axis.dims_3d()):
        if dimensions[index]:
            i = f"{axis.iteration_dace_symbol()} + {shift[axis]} + {offset_dict[axis.lower()]}"
            ranges.append((i, i, 1))

    # Append data dimensions
    for domain_size in data_domains:
        ranges.append(("0", f"{domain_size}-1", 1))  # ranges are inclusive

    return subsets.Range(ranges)


def _memlet_subset_variable_offset(
    node: oir.FieldAccess, data_domains: list[int], ctx: Context
) -> subsets.Subset:
    # Handle cartesian indices
    shift = ctx.tree.shift[node.name]
    i = f"{tir.Axis.I.iteration_symbol()} + {shift[tir.Axis.I]}"
    j = f"{tir.Axis.J.iteration_symbol()} + {shift[tir.Axis.J]}"
    K = f"{tir.Axis.K.domain_symbol()} + {shift[tir.Axis.K]} - 1"  # ranges are inclusive
    ranges = [(i, i, 1), (j, j, 1), ("0", K, 1)]

    # Append data dimensions
    for domain_size in data_domains:
        ranges.append(("0", f"{domain_size}-1", 1))  # ranges are inclusive

    return subsets.Range(ranges)
