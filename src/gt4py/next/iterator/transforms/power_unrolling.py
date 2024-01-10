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


@dataclasses.dataclass
class PowerUnrolling(NodeTranslator):
    max_unroll: int

    @classmethod
    def apply(cls, node: ir.Node, max_unroll=5) -> ir.Node:
        return cls(max_unroll=max_unroll).visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        def check_node(
            node: ir.FunCall,
        ) -> bool:
            return (
                isinstance(node.fun, ir.SymRef)
                and node.fun.id == "power"
                and isinstance(node.args[1], ir.Literal)
                and float(node.args[1].value) == int(node.args[1].value)
                and node.args[1].value >= im.literal_from_value(0).value
            )

        def check_node_args0_symref_funcall(
            node: ir.FunCall,
        ) -> bool:
            if isinstance(node.args[0], ir.SymRef):
                return True
            elif isinstance(node.args[0], ir.FunCall):
                return False
            else:
                raise TypeError("Power unrolling is only supported for ir.SymRef and ir.FunCall.")

        def compute_integer_power_of_two(exp):
            return math.floor(math.log2(exp))

        new_node = self.generic_visit(node)

        if check_node(new_node):
            assert len(new_node.args) == 2
            # Check if unroll should be performed or if exponent is too large
            base, exponent = new_node.args[0], int(new_node.args[1].value)
            if exponent > self.max_unroll:
                return new_node
            else:
                if exponent == 0:
                    return im.literal_from_value(
                        1
                    )  # TODO: returned type of literal should be the same as the one of base
                else:
                    # Calculate and store powers of two of the base as long as they are smaller than the exponent.
                    # Do the same (using the stored values) with the remainder and multiply computed values.
                    pow_cur = compute_integer_power_of_two(exponent)
                    pow_max = pow_cur
                    remainder = exponent
                    powers = [None] * (pow_max + 1)

                    # Set up powers list
                    for i in range(pow_max + 1):
                        if check_node_args0_symref_funcall(new_node):
                            if i == 0:  # power of one
                                powers[i] = base
                            else:
                                powers[i] = im.multiplies_(powers[i - 1], powers[i - 1])
                        else:
                            powers[i] = f"power_{2 ** i}"

                    # Build target expression
                    ret = powers[pow_max]
                    remainder -= 2**pow_cur
                    while remainder > 0:
                        pow_cur = compute_integer_power_of_two(remainder)
                        remainder -= 2**pow_cur

                        ret = im.multiplies_(ret, powers[pow_cur])

                    # In case of FunCall: build nested expression to avoid multiple evaluations of base
                    if not check_node_args0_symref_funcall(new_node):
                        for i in range(pow_max, 0, -1):
                            ret = im.let(
                                f"power_{2 ** i}",
                                im.multiplies_(f"power_{2**(i-1)}", f"power_{2**(i-1)}"),
                            )(ret)
                        ret = im.let("power_1", base)(ret)

                return ret
        return new_node
