# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)


@dataclasses.dataclass
class RemoveBroadcast(PreserveLocationVisitor, NodeTranslator):
    """
    Transforms calls to 'broadcast' into calls to 'as_fieldop' with 'deref'
    and the respective domain from node.annex.

    Example:
    >>> from gt4py.next import Dimension, common
    >>> from gt4py.next.common import DimensionIndex
    >>> class IDim(DimensionIndex): ...
    >>> # IR passes rebuild a dimension from its tag by importing it (ADR 0028), and a
    >>> # class declared in a doctest is not an attribute of the real module:
    >>> class JDim(DimensionIndex): ...
    >>> import sys
    >>> sys.modules[__name__].IDim, sys.modules[__name__].JDim = IDim, JDim
    >>> domain = im.domain(common.GridType.CARTESIAN, {IDim: (0, 10), JDim: (0, 10)})
    >>> expr = im.call("broadcast")(
    ...     im.ref("inp"),
    ...     im.make_tuple(
    ...         *(itir.AxisLiteral(value=dim.tag, kind=dim.kind) for dim in (IDim, JDim))
    ...     ),
    ... )
    >>> expr.annex.domain = domain_utils.SymbolicDomain.from_expr(domain)
    >>> transformed = RemoveBroadcast.apply(expr)
    >>> print(transformed)
    as_fieldop(
      deref,
      c⟨ gt4py.next.iterator.transforms.remove_broadcast.IDimₕ: [0, 10[, gt4py.next.iterator.transforms.remove_broadcast.JDimₕ: [0, 10[ ⟩
    )(inp)
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls, node: itir.Node):
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "broadcast"):
            expr = node.args[0]
            assert isinstance(node.annex.domain, domain_utils.SymbolicDomain)
            node = im.as_fieldop("deref", domain_utils.SymbolicDomain.as_expr(node.annex.domain))(
                expr
            )
        return node
