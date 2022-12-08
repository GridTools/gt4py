from typing import Any

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits
from functional.ffront.foast_passes.utils import compute_assign_indices


class UnpackedAssignPass(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Explicitly unpack assignments.

    Example
    -------
    # before pass
    def foo():
         a0 = 1
         b0 = 2
         a1, b1 = b0, a0
         return a1, b1

    # after pass
    def foo():
        a0, b0 = (1, 2)
        __tuple_tmp_0 = (1, 2)
        a1 = __tuple_tmp_0[0]
        b1 = __tuple_tmp_0[1]
        return (a1, b1)
    """

    unique_tuple_symbol_id: int = 0

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        node = cls().visit(node)
        return node

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_body = self.visit(node.body, **kwargs)
        unrolled_body = self._unroll_tuple_target_assign(new_body)
        assert isinstance(unrolled_body[-1], foast.Return)

        return foast.FunctionDefinition(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            body=unrolled_body,
            closure_vars=self.visit(node.closure_vars, **kwargs),
            type=node.type,
            location=node.location,
        )

    def _unique_tuple_symbol(self, node: foast.TupleTargetAssign) -> foast.Symbol[Any]:
        sym: foast.Symbol = foast.Symbol(
            id=f"__tuple_tmp_{self.unique_tuple_symbol_id}",
            type=node.value.type,
            location=node.location,
        )
        self.unique_tuple_symbol_id += 1
        return sym

    def _unroll_tuple_target_assign(
        self, body: list[foast.LocatedNode]
    ) -> list[foast.Assign | foast.LocatedNode]:
        unrolled: list[foast.Assign | foast.LocatedNode] = []

        for node in body:
            if isinstance(node, foast.TupleTargetAssign):
                num_elts, targets = len(node.value.type.types), node.targets  # type: ignore
                indices = compute_assign_indices(targets, num_elts)
                tuple_symbol = self._unique_tuple_symbol(node)
                unrolled.append(
                    foast.Assign(target=tuple_symbol, value=node.value, location=node.location)
                )

                for (index, subtarget) in zip(indices, targets):
                    el_type = subtarget.type
                    tuple_name = foast.Name(
                        id=tuple_symbol.id, type=el_type, location=tuple_symbol.location
                    )
                    if isinstance(index, tuple):  # handle starred target
                        lower, upper = index
                        slice_indices = list(range(lower, upper))
                        tuple_slice = [
                            foast.Subscript(
                                value=tuple_name, index=i, type=el_type, location=node.location
                            )
                            for i in slice_indices
                        ]

                        new_tuple = foast.TupleExpr(
                            elts=tuple_slice,
                            type=el_type,
                            location=node.location,
                        )
                        new_assign = foast.Assign(
                            target=subtarget.id,
                            value=new_tuple,
                            location=node.location,
                        )
                    else:
                        new_assign = foast.Assign(
                            target=subtarget,
                            value=foast.Subscript(
                                value=tuple_name, index=index, type=el_type, location=node.location
                            ),
                            location=node.location,
                        )
                    unrolled.append(new_assign)
            else:
                unrolled.append(node)

        return unrolled
