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

    # TODO
    # add a bunch more visitors below here to make sure that we raise issues
    # when we visit something "above" a CodeBlock.


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
        f"{k}{'p' if v > 0 else 'm'}{v}" for k, v in field.offset.to_dict().items() if v != 0
    ]

    return f"{base}_{'_'.join(offset_indicators)}" if offset_indicators else base
