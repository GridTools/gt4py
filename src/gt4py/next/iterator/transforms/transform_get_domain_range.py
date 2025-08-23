# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Dict

from gt4py._core import definitions as core_defs
from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)


DomainOrTupleThereof = tuple["DomainOrTupleThereof", ...] | common.Domain


class _DomainDeduction(NodeTranslator):
    def visit_Node(self, node: itir.Node, **kwargs):
        return None  # means we could not deduce the domain

    def visit_SymRef(
        self, node: itir.SymRef, *, sizes: Dict[str, common.Domain], **kwargs
    ) -> DomainOrTupleThereof | None:
        return sizes.get(node.id, None)

    def visit_Literal(self, node: itir.Literal, **kwargs) -> core_defs.Scalar:
        return ir_misc.value_from_literal(node)

    def visit_FunCall(self, node, **kwargs):
        args = self.generic_visit(node.args, **kwargs)

        if cpm.is_call_to(node, "tuple_get"):
            idx, expr = args
            return expr[idx]
        elif cpm.is_call_to(node, "make_tuple"):
            return tuple(args)

        return node


@dataclasses.dataclass(frozen=True)
class TransformGetDomainRange(PreserveLocationVisitor, NodeTranslator):
    """
    Transforms `get_domain` calls into a tuple containing start and stop.

    Example:
        >>> from gt4py import next as gtx
        >>> KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
        >>> Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)

        >>> sizes = {
        ...     "out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)}),
        ... }

        >>> output_domain = im.call("unstructured_domain")(
        ...     im.named_range(
        ...         Vertex,
        ...         im.tuple_get(0, im.call("get_domain_range")("out", Vertex)),
        ...         im.tuple_get(1, im.call("get_domain_range")("out", Vertex)),
        ...     ),
        ...     im.named_range(
        ...         KDim,
        ...         im.tuple_get(0, im.call("get_domain_range")("out", KDim)),
        ...         im.tuple_get(1, im.call("get_domain_range")("out", KDim)),
        ...     ),
        ... )
        >>> ir = itir.Program(
        ...     id="test",
        ...     function_definitions=[],
        ...     params=[im.sym("inp"), im.sym("out")],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop("deref")(im.ref("inp")),
        ...             domain=output_domain,
        ...             target=im.ref("out"),
        ...         ),
        ...     ],
        ... )
        >>> result = TransformGetDomainRange.apply(ir, sizes=sizes)
        >>> print(result)
        test(inp, out) {
          out @ u⟨ Vertexₕ: [{0, 10}[0], {0, 10}[1][, KDimᵥ: [{0, 20}[0], {0, 20}[1][ ⟩ ← (⇑deref)(inp);
        }
    """

    @classmethod
    def apply(cls, program: itir.Program, sizes: Dict[str, common.Domain]):
        return cls().visit(program, sizes=sizes)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.FunCall:
        if not cpm.is_call_to(node, "get_domain_range"):
            return self.generic_visit(node, **kwargs)

        field, dim = node.args
        assert isinstance(dim, itir.AxisLiteral)
        domain = _DomainDeduction().visit(field, sizes=kwargs["sizes"])

        if not isinstance(domain, common.Domain):
            raise ValueError(
                "Could not deduce domain of field expression. Must be a 'SymRef' or "
                "tuple expression thereof, but got:\n"
                f"'{field}'."
            )

        index = next((i for i, d in enumerate(domain.dims) if d.value == dim.value), None)
        assert index is not None, f"Dimension {dim.value} not found in {domain.dims}"

        start = domain.ranges[index].start
        stop = domain.ranges[index].stop
        node = im.make_tuple(start, stop)
        return node
