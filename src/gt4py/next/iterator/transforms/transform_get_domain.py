import dataclasses
from typing import Dict

from gt4py.eve import PreserveLocationVisitor, NodeTranslator
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.ir_utils import ir_makers as im


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

           >>> unstructured_domain_get = im.call("unstructured_domain")(
           ...      im.call("get_domain")("out", im.axis_literal(Vertex)),
           ...      im.call("get_domain")("out", im.axis_literal(KDim)),
           ...   )

           >>> unstructured_domain = im.call("unstructured_domain")(
           ...   im.call("named_range")(im.axis_literal(Vertex), 0, 10),
           ...   im.call("named_range")(im.axis_literal(KDim), 0, 20),
           ...   )

           >>> ir = itir.Program(
           ...     id="test",
           ...     function_definitions=[],
           ...     params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
           ...     declarations=[],
           ...     body=[
           ...         itir.SetAt(
           ...             expr=im.as_fieldop(im.ref("deref"))(im.ref("inp")),
           ...             domain=unstructured_domain_get,
           ...             target=im.ref("out"),
           ...         ),
           ...     ],
           ... )

           >>> sizes = {"out": gtx.domain({Vertex: (0,10), KDim: (0,20)})}

           >>> result = TransformGetDomain.apply(ir, sizes=sizes)
           >>> print(result)
           test(inp, out) {
             out @ u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩ ← (⇑deref)(inp);
           }

           >>> ir = itir.Program(
           ...     id="test",
           ...     function_definitions=[],
           ...     params=[im.sym("inp", float_i_field), im.sym("out", float_i_field)],
           ...     declarations=[],
           ...     body=[
           ...         itir.SetAt(
           ...             expr=im.as_fieldop(im.ref("deref"), unstructured_domain_get)(im.ref("inp")), # TODO: unstructured_domain_get raises AssertionError in domain_utils.py line 77: assert cpm.is_call_to(named_range, "named_range")
           ...             domain=unstructured_domain_get,
           ...             target=im.ref("out"),
           ...         ),
           ...     ],
           ... )

           >>> result = TransformGetDomain.apply(ir, sizes=sizes)
           >>> print (result) # TODO: this test still fails because of the AssertionError
           test(inp, out) {
             out @ u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩
                  ← as_fieldop(deref, u⟨ Vertexₕ: [0, 10[, KDimᵥ: [0, 20[ ⟩)(inp);
           }
       """

    @classmethod
    def apply(cls, program: itir.Program, sizes: Dict[str, common.Domain]):
        return cls().visit(program, sizes=sizes)

    def visit_FunCall(self, node: itir.SetAt, **kwargs) -> itir.FunCall:
        sizes = kwargs["sizes"]

        if cpm.is_call_to(node, "get_domain"):
            ref, dim = node.args
            if isinstance(ref, itir.SymRef):
                assert ref.id in sizes, f"Symbol '{ref.id}' not found in sizes Dict."
                input_dims = sizes[ref.id].dims
                index = next((i for i, d in enumerate(input_dims) if d.value == dim.value), None)
                assert index is not None, f"Dimension {dim.value} not found in {input_dims}"
                dim = input_dims[index]
                start = sizes[ref.id].ranges[index].start
                stop = sizes[ref.id].ranges[index].stop
                return im.call("named_range")(im.axis_literal(dim), start, stop)

        # TODO: handle tuples: get_domain(tuple_get(0, "out"))

        return self.generic_visit(node, sizes=sizes)