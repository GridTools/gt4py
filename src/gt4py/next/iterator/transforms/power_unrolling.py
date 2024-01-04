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
            return math.floor(math.log2(int(exp)))

        new_node = self.generic_visit(node)

        if check_node(new_node):
            assert len(new_node.args) == 2
            # Check if unroll should be performed or if exponent is too large
            if int(new_node.args[1].value) > self.max_unroll:
                return new_node
            else:
                if new_node.args[1].value == im.literal_from_value(0).value:
                    return im.literal_from_value(
                        1
                    )  # TODO: returned type of literal should be the same as the one of base
                else:
                    # Calculate and store powers of two of the base as long as they are smaller than the exponent.
                    # Do the same (using the stored values) with the remainder and multiply computed values.
                    pow_cur = compute_integer_power_of_two(new_node.args[1].value)
                    pow_max = pow_cur
                    remainder = int(new_node.args[1].value)
                    powers = [None] * (pow_max + 1)

                    # Account for two new_node.args[0] being either an ir.SymRef or an ir.FunCall
                    if check_node_args0_symref_funcall(new_node):
                        powers[0] = new_node.args[0].id
                    else:
                        powers[0] = new_node.args[0]
                    for i in range(1, pow_max + 1):
                        if check_node_args0_symref_funcall(new_node):
                            powers[i] = im.multiplies_(powers[i - 1], powers[i - 1])
                        else:
                            powers[i] = im.let("tmp", powers[i - 1])(im.multiplies_("tmp", "tmp"))
                    ret = powers[pow_max]
                    remainder -= 2**pow_cur
                    while remainder > 0:
                        pow_cur = compute_integer_power_of_two(remainder)
                        remainder -= 2**pow_cur
                        ret = im.multiplies_(ret, powers[pow_cur])

                return ret
        return new_node
