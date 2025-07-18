# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Dict

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import collapse_tuple


@dataclasses.dataclass(frozen=True)
class TransformGetDomain(PreserveLocationVisitor, NodeTranslator):
    """
    Transforms `get_domain` calls into `named_range` calls with given size.

    Example:
        >>> from gt4py import next as gtx
        >>> KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
        >>> Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)

        >>> sizes = {
        ...     "out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)}),
        ... }

        >>> unstructured_domain_get_out = im.call("unstructured_domain")(
        ...     im.call("get_domain")("out", im.axis_literal(Vertex)),
        ...     im.call("get_domain")("out", im.axis_literal(KDim)),
        ... )
        >>> ir = itir.Program(
        ...     id="test",
        ...     function_definitions=[],
        ...     params=[im.sym("inp"), im.sym("out")],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
        ...             domain=unstructured_domain_get_out,
        ...             target=im.ref("out"),
        ...         ),
        ...     ],
        ... )
        >>> result = TransformGetDomain.apply(ir, sizes=sizes)
        >>> print(result)
        test(inp, out) {
          out @ u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩ ← (⇑deref)(inp);
        }
    """

    @classmethod
    def apply(cls, program: itir.Program, sizes: Dict[str, common.Domain]):
        return cls().visit(program, sizes=sizes)

    def visit_FunCall(self, node: itir.SetAt, **kwargs) -> itir.FunCall:
        sizes = kwargs["sizes"]

        if not cpm.is_call_to(node, "get_domain"):
            return self.generic_visit(node, sizes=sizes)

        field, dim = node.args

        if cpm.is_call_to(field, "tuple_get"):
            ref = field.args[1]
            if isinstance(ref, itir.SymRef):
                assert ref.id in sizes, f"Symbol '{ref.id}' not found in sizes Dict."
                assert isinstance(sizes[ref.id], tuple), "A domain-tuple must be passed."
                domain = sizes[ref.id][int(field.args[0].value)]
            else:
                field = collapse_tuple.CollapseTuple.apply(
                    field, within_stencil=False, allow_undeclared_symbols=True
                )
                return self.visit(im.call("get_domain")(field, dim), sizes=sizes)
        elif isinstance(field, itir.SymRef):
            assert field.id in sizes, f"Symbol '{field.id}' not found in sizes Dict."
            domain = sizes[field.id]
        else:
            raise NotImplementedError(
                "Only calls to tuple_get or SymRefs are supported as first argument of get_domain."
            )

        input_dims = domain.dims
        index = next((i for i, d in enumerate(input_dims) if d.value == dim.value), None)
        assert index is not None, f"Dimension {dim.value} not found in {input_dims}"

        dim = input_dims[index]
        start = domain.ranges[index].start
        stop = domain.ranges[index].stop
        return im.call("named_range")(im.axis_literal(dim), start, stop)
