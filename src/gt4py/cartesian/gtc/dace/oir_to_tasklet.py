# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from dace import Memlet

from gt4py import eve
from gt4py.cartesian.gtc import common, oir


# oir to tasklet notes
#
# - CartesianOffset -> relative index
# - VariableKOffset -> relative index, but we don't know where in the K-axis
# - AbsoluteKIndex (to be ported from "the physics branch") -> i/j: relative, k: absolute
# - DataDimensions (everything after the 3rd dimension) -> absolute indices


@dataclass
class Context:
    code: list[str]

    inputs: dict[str, Memlet]
    outputs: dict[str, Memlet]


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
        raise NotImplementedError("#todo")

    def visit_AssignStmt(self, node: oir.AssignStmt, ctx: Context) -> None:
        right = self.visit(node.right, ctx=ctx, is_target=False)
        left = self.visit(node.left, ctx=ctx, is_target=True)

        ctx.code.append(f"{left} = {right}")

    def visit_CodeBlock(
        self, node: oir.CodeBlock
    ) -> tuple[str, dict[str, Memlet], dict[str, Memlet]]:
        ctx = Context(code=[], inputs={}, outputs={})

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
