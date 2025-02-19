# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.iterator.type_system import inference as itir_inference
from gt4py.next.type_system import type_specifications as ts


class InlineScalar(eve.NodeTranslator):
    @classmethod
    def apply(cls, program: itir.Program, offset_provider_type: common.OffsetProviderType):
        program = itir_inference.infer(program, offset_provider_type=offset_provider_type)
        return cls().visit(program)

    def generic_visit(self, node, **kwargs):
        if cpm.is_call_to(node, "as_fieldop"):
            return node

        return super().generic_visit(node, **kwargs)

    def visit_Expr(self, node: itir.Expr):
        node = self.generic_visit(node)

        if cpm.is_let(node):
            eligible_params = [isinstance(arg.type, ts.ScalarType) for arg in node.args]
            node = inline_lambdas.inline_lambda(node, eligible_params=eligible_params)
            return node
        return node
