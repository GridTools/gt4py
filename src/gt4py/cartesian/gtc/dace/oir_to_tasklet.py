# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from dace import Memlet, subsets

from gt4py import eve
from gt4py.cartesian.gtc import common, oir
from gt4py.cartesian.gtc.dace import daceir as dcir, prefix
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


class OIRToTasklet(eve.NodeVisitor):
    def visit_CartesianOffset(self, node: common.CartesianOffset, ctx: Context) -> None:
        raise NotImplementedError("#todo")

    def visit_VariableKOffset(self, node: oir.VariableKOffset, ctx: Context) -> None:
        raise NotImplementedError("#todo")

    def visit_ScalarAccess(self, node: oir.ScalarAccess, ctx: Context) -> str:
        # 1. build tasklet_name
        # 2. make memlet (if needed)
        # 3. return tasklet_name
        raise NotImplementedError("#todo")

    def visit_FieldAccess(self, node: oir.FieldAccess, ctx: Context, is_target: bool) -> str:
        # 1. recurse down into offset (because we could offset with another field access)
        #    - this can be an arbitrary expression -> we need more visitors
        #    - don't forget data dimensions
        # 2. build tasklet_name
        # 3. make memlet (if needed, check for read-after-write)
        # 4. return f"{tasklet_name}{offset_string}" (where offset_string is optional)
        if not isinstance(node.offset, common.CartesianOffset):
            raise NotImplementedError("Non-cartesian offsets aren't supported yet.")
        if node.data_index:
            raise NotImplementedError("Data dimensions aren't supported yet.")

        target = is_target or node.name in ctx.targets
        tasklet_name = _tasklet_name(node, target)

        if (
            tasklet_name in ctx.targets  # read after write
            or tasklet_name in ctx.inputs  # read after read
            or tasklet_name in ctx.outputs  # write after write
        ):
            return tasklet_name

        offset_dict = node.offset.to_dict()
        memlet = Memlet(
            data=node.name,
            subset=subsets.Indices(
                [
                    f"{axis.iteration_dace_symbol()} + {offset_dict[axis.lower()]}"
                    for axis in dcir.Axis.dims_3d()
                ]
            ),
        )
        if is_target:
            ctx.targets.add(tasklet_name)
            ctx.outputs[tasklet_name] = memlet
        else:
            ctx.inputs[tasklet_name] = memlet

        return tasklet_name

    def visit_AssignStmt(self, node: oir.AssignStmt, ctx: Context) -> None:
        right = self.visit(node.right, ctx=ctx, is_target=False)
        left = self.visit(node.left, ctx=ctx, is_target=True)

        ctx.code.append(f"{left} = {right}")

    def visit_CodeBlock(
        self, node: oir.CodeBlock
    ) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
        ctx = Context(code=[], targets=set(), inputs={}, outputs={})

        self.visit(node.body, ctx=ctx)

        return ("\n".join(ctx.code), ctx.inputs, ctx.outputs)

    def visit_BinaryOp(self, node: oir.BinaryOp, ctx: Context, **kwargs) -> str:
        right = self.visit(node.right, ctx=ctx, **kwargs)
        left = self.visit(node.left, ctx=ctx, **kwargs)
        return f"{left} {node.op.value} {right}"

    def visit_Cast(self, node: oir.Cast, **kwargs) -> str:
        dtype = data_type_to_dace_typeclass(node.dtype)
        return f"{dtype}({self.visit(node.expr)})"

    def visit_Literal(self, node: oir.Literal, **kwargs) -> str:
        return node.value

    # Not implemented blocks - implement or pass ot generic visitor
    def visit_AbsoluteKIndex(self, node, **kwargs):
        raise NotImplementedError("visit_AbsoluteKIndex")

    def visit_CacheDesc(self, node, **kwargs):
        raise NotImplementedError("visit_CacheDesc")

    def visit_Decl(self, node, **kwargs):
        raise NotImplementedError("visit_Decl")

    def visit_FieldDecl(self, node, **kwargs):
        raise NotImplementedError("visit_FieldDecl")

    def visit_HorizontalExecution(self, node, **kwargs):
        raise NotImplementedError("visit_HorizontalExecution")

    def visit_HorizontalRestriction(self, node, **kwargs):
        raise NotImplementedError("visit_HorizontalRestriction")

    def visit_IJCache(self, node, **kwargs):
        raise NotImplementedError("visit_IJCache")

    def visit_Interval(self, node, **kwargs):
        raise NotImplementedError("visit_Interval")

    def visit_IteratorAccess(self, node, **kwargs):
        raise NotImplementedError("visit_IteratorAccess")

    def visit_KCache(self, node, **kwargs):
        raise NotImplementedError("visit_KCache")

    def visit_LocalScalar(self, node, **kwargs):
        raise NotImplementedError("visit_LocalScalar")

    def visit_MaskStmt(self, node, **kwargs):
        raise NotImplementedError("visit_MaskStmt")

    def visit_NativeFuncCall(self, node, **kwargs):
        raise NotImplementedError("visit_NativeFuncCall")

    def visit_ScalarDecl(self, node, **kwargs):
        raise NotImplementedError("visit_ScalarDecl")

    def visit_Stencil(self, node, **kwargs):
        raise NotImplementedError("visit_Stencil")

    def visit_Temporary(self, node, **kwargs):
        raise NotImplementedError("visit_Temporary")

    def visit_TernaryOp(self, node, **kwargs):
        raise NotImplementedError("visit_TernaryOp")

    def visit_UnaryOp(self, node, **kwargs):
        raise NotImplementedError("visit_UnaryOp")

    def visit_UnboundedInterval(self, node, **kwargs):
        raise NotImplementedError("visit_UnboundedInterval")

    # Should _not_ be called
    def visit_VerticalLoop(self, node, **kwargs):
        raise NotImplementedError("[OIR_to_Tasklet] visit_VerticalLoop should not be called")

    def visit_VerticalLoopSection(self, node, **kwargs):
        raise NotImplementedError("[OIR_to_Tasklet] visit_VerticalLoopSection should not be called")


def generate(node: oir.CodeBlock) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
    # This function is basically only here to be able to easily type-hint the
    # return value of the CodeBlock visitor.
    # (OIRToTasklet().visit(node) returns an Any type, which looks ugly in the
    #  CodeBlock visitor of OIRToTreeIR)
    return OIRToTasklet().visit(node)


def _tasklet_name(field: oir.FieldAccess, is_target: bool) -> str:
    base = f"{prefix.TASKLET_OUT if is_target else prefix.TASKLET_IN}{field.name}"

    if not isinstance(field.offset, common.CartesianOffset):
        raise NotImplementedError("Non-cartesian offsets aren't supported yet.")
    if field.data_index:
        raise NotImplementedError("Data dimensions aren't supported yet.")

    offset_indicators = [
        f"{k}{'p' if v > 0 else 'm'}{abs(v)}" for k, v in field.offset.to_dict().items() if v != 0
    ]

    return f"{base}_{'_'.join(offset_indicators)}" if offset_indicators else base
