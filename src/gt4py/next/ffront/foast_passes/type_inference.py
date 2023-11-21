from gt4py.next.ffront import field_operator_ast as foast
from gt4py import eve
from gt4py.next import errors
from gt4py.next.type_system_2 import types as ts2
from gt4py.next.ffront.type_system_2 import types as ts2_f, inference as ti2_f
from gt4py.next.type_system_2 import traits
from typing import Optional, Any
import dataclasses


@dataclasses.dataclass(frozen=True)
class ClosureVarInferencePass(eve.NodeTranslator, eve.traits.VisitorWithSymbolTableTrait):
    closure_vars: dict[str, Any]

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs) -> foast.FunctionDefinition:
        new_closure_vars: list[foast.Symbol] = []
        for sym in node.closure_vars:
            if not isinstance(self.closure_vars[sym.id], type):
                new_symbol: foast.Symbol = foast.Symbol(
                    id=sym.id,
                    location=sym.location,
                    type_2=ti2_f.inferrer.from_instance(self.closure_vars[sym.id]),
                )
                new_closure_vars.append(new_symbol)
            else:
                new_closure_vars.append(sym)
        return foast.FunctionDefinition(
            id=node.id,
            params=node.params,
            body=node.body,
            closure_vars=new_closure_vars,
            type_2=node.type_2,
            location=node.location,
        )


class TypeInferencePass(eve.traits.VisitorWithSymbolTableTrait, eve.NodeTranslator):
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
        ty = node.type_2
        assert isinstance(ty, ts2.FunctionType)
        self.result = ty.result
        return foast.FunctionDefinition(
            id=node.id,
            params=[self.visit(item, **kwargs) for item in node.params],
            body=self.visit(node.body, **kwargs),
            closure_vars=[self.visit(item, **kwargs) for item in node.closure_vars],
            type_2=node.type_2,
            location=node.location,
        )

    def visit_Symbol(self, node: foast.Symbol, **kwargs) -> foast.Symbol:
        symtable = kwargs["symtable"]
        symtable[node.id] = node
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
        return foast.Assign(target=target, value=value, type_2=value.type_2, location=node.location)

    def visit_Return(self, node: foast.Return, **kwargs) -> foast.Return:
        value = self.visit(node.value, **kwargs)
        if traits.is_convertible(value.type_2, self.result):
            raise errors.DSLError(
                node.location,
                f"could not convert returned value of type '{value.type_2}' to {self.result}"
            )
        return foast.Return(value=value, type_2=value.type_2, location=node.location)

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        elements = self.visit(node.elts, **kwargs)
        ty = ts2.TupleType([element.type_2 for element in elements])
        return foast.TupleExpr(elts=elements, type_2=ty, location=node.location)

    def visit_Subscript(self, node: foast.Subscript, **kwargs):
        value = self.visit(node.value, **kwargs)
        if isinstance(value.type_2, ts2.TupleType):
            ty = value.type_2.elements[node.index]
            return foast.Subscript(value=value, index=node.index, type_2=ty, location=node.location)
        raise errors.DSLError(node.location, f"expression of type '{value.type_2}' is not subscriptable")
