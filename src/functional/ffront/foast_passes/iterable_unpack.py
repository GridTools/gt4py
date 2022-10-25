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
    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        typed_foast_node = cls().visit(node)
        return typed_foast_node

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_params = self.visit(node.params, **kwargs)
        new_body = self.visit(node.body, **kwargs)

        for i, elt in enumerate(new_body):
            if any([isinstance(elt, list)]):
                del new_body[i]
                for assign in elt:
                    new_body.insert(i, assign)


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

    def visit_MultiTargetAssign(self, node: foast.MultiTargetAssign, **kwargs) -> list[foast.Assign]:
        assigns = []
        values = node.value.elts
        if isinstance(targets := node.target, (tuple, list)):
            if len(values) != len(targets):
                raise Exception("Left and right side of MultiTargetAssign must contain same number of elements.")

            for t, v in zip(targets, values):
                assigns.append(foast.Assign(target=t, value=v, location=node.location))
        return assigns
