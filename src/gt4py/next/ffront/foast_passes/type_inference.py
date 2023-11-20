from gt4py.next.ffront import field_operator_ast as foast
from gt4py import eve
from gt4py.next import errors
from gt4py.next.type_system_2 import types as ts2
from gt4py.next.ffront.type_system_2 import types as ts2_f
from gt4py.next.type_system_2 import traits
from typing import Optional


class TypeInferencePass(eve.NodeTranslator):
    result: Optional[ts2.Type]

    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs):
        ty = node.type_2
        assert isinstance(ty, ts2_f.FieldOperatorType)
        self.result = ty.result
        definition = self.visit(node.definition, **kwargs)
        return foast.FieldOperator(
            id=node.id,
            definition=definition,
            type_2=node.type_2,
            location=node.location
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs):
        raise NotImplementedError()

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        self.visit(node.body)
        return foast.FunctionDefinition(
            id=node.id,
            params=[self.visit(item, **kwargs) for item in node.params],
            body=self.visit(node.body, **kwargs),
            closure_vars=[self.visit(item, **kwargs) for item in node.closure_vars],
            type_2=node.type_2,
            location=node.location,
        )

    def visit_Symbol(self, node: foast.Symbol, **_) -> foast.Symbol:
        return node

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable:
            raise errors.UndefinedSymbolError(node.location, node.id)
        symbol: foast.Symbol = symtable[node.id]
        assert symbol.type_2 is not None
        return foast.Name(id=node.id, type_2=symbol.type_2, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs) -> foast.Assign:
        value = self.visit(node.value, **kwargs)
        target = self.visit(node.target, **kwargs)
        target.type_2 = value.type_2
        return foast.Assign(target=target, value=value, location=node.location)

    def visit_Return(self, node: foast.Return, **kwargs) -> foast.Return:
        value = self.visit(node.value, **kwargs)
        if self.result.implements(traits.From(value.type_2)):
            raise errors.DSLError(
                node.location,
                f"could not convert returned value of type {value.type_2} to {self.result}"
            )
        return foast.Return(value=value, type_2=value.type_2, location=node.location)
