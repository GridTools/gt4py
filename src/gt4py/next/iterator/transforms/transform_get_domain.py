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
        >>> from gt4py.next.type_system import type_specifications as ts
        >>> from gt4py import next as gtx
        >>> float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
        >>> IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
        >>> KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
        >>> Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
        >>> float_i_field = ts.FieldType(dims=[IDim], dtype=float64_type)


        >>> sizes = {
        ...     "out": gtx.domain({Vertex: (0, 10), KDim: (0, 20)}),
        ...     "a": (gtx.domain({Vertex: (0, 5)}), gtx.domain({Vertex: (0, 7)})),
        ...     "b": gtx.domain({KDim: (0, 3)}),
        ...     "c": gtx.domain({KDim: (0, 4)}),
        ... }

        >>> unstructured_domain_get_out = im.call("unstructured_domain")(
        ...     im.call("get_domain")("out", im.axis_literal(Vertex)),
        ...     im.call("get_domain")("out", im.axis_literal(KDim)),
        ... )
        >>> ir = itir.Program(
        ...     id="test1",
        ...     function_definitions=[],
        ...     params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
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
        test1(inp, out) {
          out @ u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩ ← (⇑deref)(inp);
        }

        >>> unstructured_domain = im.call(
        ...     "unstructured_domain"
        ... )(  # TODO: remove once the AssertionError is fixed
        ...     im.call("named_range")(im.axis_literal(Vertex), 0, 10),
        ...     im.call("named_range")(im.axis_literal(KDim), 0, 20),
        ... )
        >>> ir = itir.Program(
        ...     id="test2",
        ...     function_definitions=[],
        ...     params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get_out)(
        ...                 im.ref("inp")
        ...             ),  # TODO: unstructured_domain_get raises AssertionError in domain_utils.py line 77: assert cpm.is_call_to(named_range, "named_range")
        ...             domain=unstructured_domain_get_out,
        ...             target=im.ref("out"),
        ...         ),
        ...     ],
        ... )
        >>> result = TransformGetDomain.apply(ir, sizes=sizes)
        >>> print(result)  # TODO: this test still fails because of the AssertionError
        test2(inp, out) {
          out @ u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩
               ← as_fieldop(deref, u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩)(inp);
        }

        >>> unstructured_domain_get_a = im.call("unstructured_domain")(
        ...     im.call("get_domain")(im.tuple_get(0, "a"), im.axis_literal(Vertex))
        ... )
        >>> ir = itir.Program(
        ...     id="test3",
        ...     function_definitions=[],
        ...     params=[im.sym("inp", float_i_field), im.sym("a")],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
        ...             domain=unstructured_domain_get_a,
        ...             target=im.tuple_get(0, "a"),
        ...         ),
        ...     ],
        ... )
        >>> result = TransformGetDomain.apply(ir, sizes=sizes)
        >>> print(result)
        test3(inp, a) {
          a[0] @ u⟨ Vertexₕ: [0, 5[ ⟩ ← (⇑deref)(inp);
        }

        >>> t0 = im.make_tuple("b", "c")
        >>> t1 = im.make_tuple("d", "e")
        >>> tup = im.make_tuple(im.tuple_get(0, t0), im.tuple_get(1, t1))
        >>> unstructured_domain_get_make_tuple_b = im.call("unstructured_domain")(
        ...     im.call("get_domain")(im.tuple_get(0, tup), im.axis_literal(KDim))
        ... )
        >>> ir = itir.Program(
        ...     id="test4",
        ...     function_definitions=[],
        ...     params=[
        ...         im.sym("inp", float_i_field),
        ...         im.sym("b"),
        ...         im.sym("c"),
        ...         im.sym("d"),
        ...         im.sym("e"),
        ...     ],
        ...     declarations=[],
        ...     body=[
        ...         itir.SetAt(
        ...             expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
        ...             domain=unstructured_domain_get_make_tuple_b,
        ...             target=im.ref("b"),
        ...         ),
        ...     ],
        ... )
        >>> result = TransformGetDomain.apply(ir, sizes=sizes)
        >>> print(result)
        test4(inp, b, c, d, e) {
          b @ u⟨ KDimᵥ: [0, 3[ ⟩ ← (⇑deref)(inp);
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
