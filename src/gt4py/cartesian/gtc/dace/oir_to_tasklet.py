# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

from dace import Memlet, subsets

from gt4py import eve
from gt4py.cartesian.gtc import common, oir
from gt4py.cartesian.gtc.dace import daceir as dcir, prefix, treeir as tir
from gt4py.cartesian.gtc.dace.symbol_utils import data_type_to_dace_typeclass


# oir to tasklet notes
#
# - CartesianOffset -> relative index
# - VariableKOffset -> relative index, but we don't know where in the K-axis
# - AbsoluteKIndex (to be ported from "the physics branch") -> i/j: relative, k: absolute
# - DataDimensions (everything after the 3rd dimension) -> absolute indices


@dataclass
class Context:
    code: list[str]
    """The code, line by line."""

    targets: set[str]
    """Tasklet names of fields / scalars that we've already written to. For read-after-write analysis."""

    inputs: dict[str, Memlet]
    """Mapping connector names to memlets flowing into this code block."""
    outputs: dict[str, Memlet]
    """Mapping connector names to memlets flowing out of this code block."""

    tree: tir.TreeRoot
    """Schedule tree in which this tasklet will be inserted."""


class OIRToTasklet(eve.NodeVisitor):
    def _memlet_subset(self, node: oir.FieldAccess, ctx: Context) -> subsets.Subset:
        if isinstance(node.offset, common.CartesianOffset):
            offset_dict = node.offset.to_dict()
            # TODO
            # This has to be reworked with support for data dimensions
            return subsets.Indices(
                [
                    f"{axis.iteration_dace_symbol()} + {offset_dict[axis.lower()]}"
                    for i, axis in enumerate(dcir.Axis.dims_3d())
                    if i < len(ctx.tree.containers[node.name].shape)
                ]
            )

        if isinstance(node.offset, oir.VariableKOffset):
            # TODO
            # This has to be reworked with support for data dimensions
            i = dcir.Axis.I.iteration_symbol()
            j = dcir.Axis.J.iteration_symbol()
            K = dcir.Axis.K.domain_symbol()
            return subsets.Range([(i, i, 1), (j, j, 1), (0, K, 1)])

        raise NotImplementedError(f"_memlet_subset(): unknown offset type {type(node.offset)}")

    def visit_CodeBlock(
        self, node: oir.CodeBlock, root: tir.TreeRoot
    ) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
        """Entry point to gather all code, input and outputs"""
        ctx = Context(code=[], targets=set(), inputs={}, outputs={}, tree=root)

        self.visit(node.body, ctx=ctx)

        return ("\n".join(ctx.code), ctx.inputs, ctx.outputs)

    def visit_CartesianOffset(self, _node: common.CartesianOffset, **_kwargs: Any) -> None:
        raise ValueError("Cartesian Offset should be dealt in Access IRs.")

    def visit_VariableKOffset(self, node: oir.VariableKOffset, **_kwargs: Any) -> None:
        raise ValueError("Variable K Offset should be dealt in Access IRs.")

    def visit_ScalarAccess(self, node: oir.ScalarAccess, ctx: Context, is_target: bool) -> str:
        target = is_target or node.name in ctx.targets
        tasklet_name = _tasklet_name(node, target)

        if (
            node.name in ctx.targets  # (read or write) after write
            or tasklet_name in ctx.inputs  # read after read
        ):
            return tasklet_name

        memlet = Memlet(data=node.name, subset=subsets.Range([(0, 0, 1)]))  # Memlet(node.name)
        if is_target:
            # note: it doesn't matter if we use is_target or target here because if they
            # were different, we had a read-after-write situation, which was already
            # handled above.
            ctx.targets.add(node.name)
            ctx.outputs[tasklet_name] = memlet
        else:
            ctx.inputs[tasklet_name] = memlet

        return tasklet_name

    def visit_FieldAccess(self, node: oir.FieldAccess, ctx: Context, is_target: bool) -> str:
        # 1. recurse down into offset (because we could offset with another field access)
        #    - this can be an arbitrary expression -> we need more visitors
        #    - don't forget data dimensions
        # 2. build tasklet_name
        # 3. make memlet (if needed, check for read-after-write)
        # 4. return f"{tasklet_name}{offset_string}" (where offset_string is optional)
        if not isinstance(node.offset, (common.CartesianOffset, oir.VariableKOffset)):
            raise NotImplementedError(f"Unexpected offsets offset type: {type(node.offset)}.")
        if node.data_index:
            raise NotImplementedError("Data dimensions aren't supported yet.")

        postfix = _field_offset_postfix(node)
        key = f"{node.name}_{postfix}"
        target = is_target or key in ctx.targets
        tasklet_name = _tasklet_name(node, target, postfix)
        # recurse down
        var_k = (
            self.visit(node.offset.k, ctx=ctx, is_target=False)
            if isinstance(node.offset, oir.VariableKOffset)
            else None
        )

        if (
            key in ctx.targets  # (read or write) after write
            or tasklet_name in ctx.inputs  # read after read
        ):
            return tasklet_name if var_k is None else f"{tasklet_name}[{var_k}]"

        memlet = Memlet(
            data=node.name,
            subset=self._memlet_subset(node, ctx=ctx),
        )
        if is_target:
            # note: it doesn't matter if we use is_target or target here because if they
            # were different, we had a read-after-write situation, which was already
            # handled above.
            ctx.targets.add(key)
            ctx.outputs[tasklet_name] = memlet
        else:
            ctx.inputs[tasklet_name] = memlet

        return tasklet_name if var_k is None else f"{tasklet_name}[{var_k}]"

    def visit_AssignStmt(self, node: oir.AssignStmt, ctx: Context) -> None:
        # order matters: evaluate right side of the assignment first
        right = self.visit(node.right, ctx=ctx, is_target=False)
        left = self.visit(node.left, ctx=ctx, is_target=True)

        ctx.code.append(f"{left} = {right}")

    def visit_BinaryOp(self, node: oir.BinaryOp, **kwargs: Any) -> str:
        left = self.visit(node.left, **kwargs)
        right = self.visit(node.right, **kwargs)
        return f"{left} {node.op.value} {right}"

    def visit_UnaryOp(self, node: oir.UnaryOp, **kwargs: Any) -> str:
        expr = self.visit(node.expr, **kwargs)
        return f"{node.op.value} {expr}"

    def visit_Cast(self, node: oir.Cast, **kwargs: Any) -> str:
        dtype = data_type_to_dace_typeclass(node.dtype)
        return f"{dtype}({self.visit(node.expr, **kwargs)})"

    def visit_Literal(self, node: oir.Literal, **_kwargs: Any) -> str:
        return node.value

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
                common.NativeFunction.LOG10: "log10",
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

    def visit_NativeFuncCall(self, node: oir.NativeFuncCall, **kwargs: Any) -> str:
        print(node.func)
        return f"{self.visit(node.func, **kwargs)}({','.join([self.visit(a, **kwargs) for a in node.args])})"

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs):
        # Skip here, OIR to TreeIR will catch
        pass

    # Not implemented blocks - implement or pass to generic visitor
    def visit_AbsoluteKIndex(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Absolute K")

    def visit_CacheDesc(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Caches")

    def visit_IJCache(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Caches")

    def visit_KCache(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Caches")

    def visit_HorizontalExecution(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Regions")

    def visit_HorizontalRestriction(self, node, **kwargs):
        raise NotImplementedError("To be implemented: Regions")

    def visit_While(self, node, **kwargs):
        raise NotImplementedError("To be implemented: while")

    def visit_TernaryOp(self, node, **kwargs):
        raise NotImplementedError("To be implemented: ops")

    # Should _not_ be called
    def visit_LocalScalar(self, node, **kwargs):
        raise NotImplementedError("visit_LocalScalar should not be called")

    def visit_Temporary(self, node, **kwargs):
        raise NotImplementedError("visit_LocalScalar should not be called")

    def visit_Stencil(self, node, **kwargs):
        raise NotImplementedError("visit_Stencil should not be called")

    def visit_Decl(self, node, **kwargs):
        raise NotImplementedError("visit_Decl should not be called")

    def visit_FieldDecl(self, node, **kwargs):
        raise NotImplementedError("visit_FieldDecl should not be called")

    def visit_ScalarDecl(self, node, **kwargs):
        raise NotImplementedError("visit_ScalarDecl should not be called")

    def visit_Interval(self, node, **kwargs):
        raise NotImplementedError("visit_Interval should not be called")

    def visit_UnboundedInterval(self, node, **kwargs):
        raise NotImplementedError("visit_UnboundedInterval should not be called")

    def visit_VerticalLoop(self, node, **kwargs):
        raise NotImplementedError("visit_VerticalLoop should not be called")

    def visit_VerticalLoopSection(self, node, **kwargs):
        raise NotImplementedError("visit_VerticalLoopSection should not be called")


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
    name_prefix = prefix.TASKLET_OUT if is_target else prefix.TASKLET_IN
    return "_".join(filter(None, [name_prefix, node.name, postfix]))


def _field_offset_postfix(node: oir.FieldAccess) -> str:
    if node.data_index:
        raise NotImplementedError("Data dimensions aren't supported yet.")

    if isinstance(node.offset, oir.VariableKOffset):
        return "var_k"

    offset_indicators = [
        f"{k}{'p' if v > 0 else 'm'}{abs(v)}" for k, v in node.offset.to_dict().items() if v != 0
    ]
    return "_".join(offset_indicators)
