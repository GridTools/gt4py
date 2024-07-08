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
import dataclasses
import math

from gt4py.eve import NodeTranslator
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


def _is_power_call(node: ir.FunCall) -> bool:
    """Match expressions of the form `power(base, integral_literal)`."""
    return (
        isinstance(node.fun, ir.SymRef)
        and node.fun.id == "power"
        and isinstance(node.args[1], ir.Literal)
        and float(node.args[1].value) == int(node.args[1].value)
        and node.args[1].value >= im.literal_from_value(0).value
    )


def _compute_integer_power_of_two(exp: int) -> int:
    return math.floor(math.log2(exp))


@dataclasses.dataclass
class PowerUnrolling(NodeTranslator):
    max_unroll: int

    @classmethod
    def apply(cls, node: ir.Node, max_unroll: int = 5) -> ir.Node:
        return cls(max_unroll=max_unroll).visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        new_node = self.generic_visit(node)

        if _is_power_call(new_node):
            assert len(new_node.args) == 2
            # Check if unroll should be performed or if exponent is too large
            base, exponent = new_node.args[0], int(new_node.args[1].value)
            if 1 <= exponent <= self.max_unroll:
                # Calculate and store powers of two of the base as long as they are smaller than the exponent.
                # Do the same (using the stored values) with the remainder and multiply computed values.
                pow_cur = _compute_integer_power_of_two(exponent)
                pow_max = pow_cur
                remainder = exponent

                # Build target expression
                ret = im.ref(f"power_{2 ** pow_max}")
                remainder -= 2**pow_cur
                while remainder > 0:
                    pow_cur = _compute_integer_power_of_two(remainder)
                    remainder -= 2**pow_cur

                    ret = im.multiplies_(ret, f"power_{2 ** pow_cur}")

                # Nest target expression to avoid multiple redundant evaluations
                for i in range(pow_max, 0, -1):
                    ret = im.let(
                        f"power_{2 ** i}", im.multiplies_(f"power_{2**(i-1)}", f"power_{2**(i-1)}")
                    )(ret)
                ret = im.let("power_1", base)(ret)

                # Simplify expression in case of SymRef by resolving let statements
                if isinstance(base, ir.SymRef):
                    return InlineLambdas.apply(ret, opcount_preserving=True)
                else:
                    return ret
        return new_node
