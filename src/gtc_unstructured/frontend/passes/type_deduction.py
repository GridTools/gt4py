from numbers import Number
from typing import Any, Dict

import eve

from ..built_in_types import BuiltInType
from ..gtscript import Connectivity, Field, LocalField, Location
from ..gtscript_ast import (
    Argument,
    BinaryOp,
    Call,
    Constant,
    External,
    Generator,
    GTScriptASTNode,
    LocationComprehension,
    LocationSpecification,
    Subscript,
    SubscriptCall,
    SymbolRef,
    TemporaryFieldDecl,
    TemporarySparseFieldDecl,
)
from .const_expr_evaluator import evaluate_const_expr


class TypeDeduction(eve.NodeVisitor):
    @classmethod
    def apply(cls, symtable, values):
        instance = cls()
        return instance.visit(values, symtable=symtable)

    def visit_Node(self, node, *, symtable, **kwargs):
        raise ValueError(f"Type of node {node} not defined.")

    def visit_Argument(self, node: Argument, **kwargs):
        if issubclass(node.type_, Field):
            return Number
        elif issubclass(node.type_, Connectivity):
            return node.type_

        raise ValueError()

    def visit_TemporaryFieldDecl(self, node: TemporaryFieldDecl, **kwargs):
        return Number

    def visit_TemporarySparseFieldDecl(self, node: TemporarySparseFieldDecl, **kwargs):
        return Number

    def visit_Call(self, node: Call, **kwargs):
        # todo: enhance
        return Number
        # return built_in_functions[node.func].return_type(*self.visit(node.args)) # noqa E800

    def visit_Constant(self, node: Constant, **kwargs):
        return type(node.value)

    def visit_SymbolRef(self, node: SymbolRef, symtable, **kwargs):
        return self.visit(symtable[node.name], symtable=symtable, **kwargs)

    def visit_External(self, node: External, **kwargs):
        return type(node.value)

    def visit_LocationSpecification(self, node: LocationSpecification, *, symtable, **kwargs):
        return Location[evaluate_const_expr(symtable, node.location_type)]

    def visit_LocationComprehension(self, node: LocationComprehension, *, symtable, **kwargs):
        connectivity = symtable[node.iterable.value.name].type_
        index_type = self.visit(node.iterable.indices[0], symtable=symtable, **kwargs)
        if index_type != Location[connectivity.primary_location()]:
            raise ValueError(
                f"You are trying to access a connectivity posed on {connectivity.primary_location()} using a location of type {index_type}"
            )
        return Location[connectivity.secondary_location()]

    def visit_SubscriptCall(self, node: SubscriptCall, *, symtable, **kwargs):
        func = evaluate_const_expr(symtable, node.func)
        if issubclass(func, BuiltInType):
            return func
        raise ValueError(f"Type of node {node} not defined.")

    def visit_BinaryOp(self, node: BinaryOp, **kwargs):
        # todo: enhance
        return Number

    def visit_Subscript(self, node: Subscript, *, symtable, **kwargs):
        # todo: enhance
        assert all(isinstance(idx, SymbolRef) for idx in node.indices)
        if all(
            isinstance(symtable[idx.name], (LocationSpecification, LocationComprehension))  # type: ignore[union-attr]
            for idx in node.indices
        ):
            return Number
        raise ValueError(f"Type of node {node} not defined.")

    def visit_Generator(self, node: Generator, *, symtable, **kwargs):
        connectivity = symtable[node.generators[0].iterable.value.name].type_
        assert issubclass(connectivity, Connectivity)
        return LocalField[connectivity]


def deduce_type(symtable: Dict[Any, eve.Node], node: GTScriptASTNode):
    return TypeDeduction.apply(symtable, node)
