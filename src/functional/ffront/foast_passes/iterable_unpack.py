import ast
import copy
import functional.ffront.field_operator_ast as foast
from collections.abc import Iterator
from functional.ffront import common_types as ct
from eve import NodeTranslator, traits


class UnpackedAssignPass(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Explicitly unpack assignments.

    Requires AST in SSA form and assumes only single target assigns, check the following passes.

     * ``SingleStaticAssignPass``
     * ``SingleAssignTargetPass``

    Example
    -------
    >>> import ast, inspect

    >>> def foo():
    ...     a0 = 1
    ...     b0 = 5
    ...     a1, b1 = b0, a0
    ...     return a1, b1

    >>> print(ast.unparse(
    ...     UnpackedAssignPass.apply(
    ...         ast.parse(inspect.getsource(foo))
    ...     )
    ... ))
    def foo():
        a0 = 1
        b0 = 5
        __tuple_tmp_0 = (b0, a0)
        a1 = __tuple_tmp_0[0]
        b1 = __tuple_tmp_0[1]
        return (a1, b1)

    which would not have been equivalent had the input AST not been in SSA form.
    """

    unique_name_id: int = 0

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

        # self._unpack_assignment(node, targets=target.elts)

    def _unique_tuple_name(self) -> ast.Name:
        name = ast.Name(id=f"__tuple_tmp_{self.unique_name_id}", ctx=ast.Store())
        self.unique_name_id += 1
        return name

    def _unpack_assignment(
            self, node: list[foast.Symbol | foast.Star], *, targets: list[foast.Symbol | foast.Star]
            # targets passed here for typing
    ) -> Iterator[ast.Assign]:
        tuple_name = self._unique_tuple_name()
        tuple_assign = ast.Assign(targets=[tuple_name], value=node.value)
        ast.copy_location(tuple_name, node)
        ast.copy_location(tuple_assign, node)
        yield from self.visit_Assign(tuple_assign)

        for index, subtarget in enumerate(targets):
            new_assign = copy.copy(node)
            new_assign.targets = [subtarget]
            new_assign.value = ast.Subscript(
                ctx=ast.Load(),  # <- ctx is mandatory for ast.Subscript, Load() for rhs.
                value=tuple_name,
                slice=ast.Constant(value=index),
            )
            ast.copy_location(new_assign.value, node.value)
            ast.copy_location(new_assign.value.slice, node.value)
            yield from self.visit_Assign(new_assign)
