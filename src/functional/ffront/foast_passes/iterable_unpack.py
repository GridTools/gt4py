import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, traits
from functional.ffront import common_types as ct


class UnpackedAssignPass(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Explicitly unpack assignments.

    Example
    -------
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
        typed_foast_node = cls().visit(node)
        return typed_foast_node

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_params = self.visit(node.params, **kwargs)
        new_body = self.visit(node.body, **kwargs)
        self._unroll_multi_target_assign(new_body)
        assert isinstance(new_body[-1], foast.Return)
        return_type = new_body[-1].value.type
        new_type = ct.FunctionType(
            args=[new_param.type for new_param in new_params], kwargs={}, returns=return_type
        )

        return foast.FunctionDefinition(
            id=node.id,
            params=new_params,
            body=new_body,
            closure_vars=self.visit(node.closure_vars, **kwargs),
            type=new_type,
            location=node.location,
        )

    def _unique_tuple_symbol(self, node: foast.MultiTargetAssign) -> foast.Name:
        sym = foast.Symbol(id=f"__tuple_tmp_{self.unique_tuple_symbol_id}", type=node.value.type, location=node.location)
        self.unique_tuple_symbol_id += 1
        return sym

    def _unroll_multi_target_assign(self, body: list[foast.LocatedNode]) -> list[foast.LocatedNode]:
        for pos, node in enumerate(body):
            if isinstance(node, foast.MultiTargetAssign):
                del body[pos]
                tuple_symbol = self._unique_tuple_symbol(node)
                tuple_assign = foast.Assign(target=tuple_symbol, value=node.value, location=node.location)
                body.insert(pos, tuple_assign)

                for index, subtarget in enumerate(node.targets):
                    tuple_name = foast.Name(id=tuple_symbol.id, location=tuple_symbol.location)
                    new_assign = foast.Assign(target=subtarget, value=foast.Subscript(value=tuple_name, index=index, location=node.location), location=node.location)
                    body.insert(pos + index + 1, new_assign)

        return body
