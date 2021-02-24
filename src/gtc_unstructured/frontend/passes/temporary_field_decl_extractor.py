from numbers import Number
from typing import Any, Dict, List, Optional, Type, Union

import eve

from .. import built_in_types
from ..gtscript_ast import (
    Assign,
    LocationSpecification,
    Stencil,
    Subscript,
    SymbolName,
    SymbolRef,
    TemporaryFieldDecl,
    TemporarySparseFieldDecl,
)
from .const_expr_evaluator import evaluate_const_expr
from .type_deduction import deduce_type


class TemporaryFieldDeclExtractor(eve.NodeVisitor):
    primary_location: Union[None, LocationSpecification]
    temporary_fields: List[TemporaryFieldDecl]

    def __init__(self):
        self.primary_location = None
        self.temporary_fields = {}

    @classmethod
    def apply(cls, symtable, stencils: List[Stencil]):
        instance = cls()
        instance.visit(stencils, symtable=symtable)
        return list(instance.temporary_fields.values())

    def visit_LocationSpecification(
        self, node: LocationSpecification, *, symtable: Dict[str, Any], **kwargs
    ):
        assert self.primary_location is None
        self.primary_location = symtable[node.name]

    def visit_Assign(self, node: Assign, *, symtable: Dict[str, Any], **kwargs):
        # extract target symbol
        if isinstance(node.target, Subscript):
            target = node.target.value
        elif isinstance(node.target, SymbolRef):
            target = node.target
        assert isinstance(target, SymbolRef)

        if target.name not in symtable and target.name not in self.temporary_fields:
            value_type = deduce_type(symtable, node.value)

            assert self.primary_location is not None
            if issubclass(value_type, built_in_types.LocalField):
                args = (value_type.args[0], evaluate_const_expr(symtable, SymbolRef(name="dtype")))
                self.temporary_fields[target.name] = TemporarySparseFieldDecl(
                    name=SymbolName(target.name), type_=built_in_types.TemporarySparseField[args]
                )
            else:  # TODO(tehrengruber): issubclass(value_type, Number)
                args = (self.primary_location.location_type, SymbolRef(name="dtype"))
                args = tuple(evaluate_const_expr(symtable, arg) for arg in args)
                self.temporary_fields[target.name] = TemporaryFieldDecl(
                    name=SymbolName(target.name), type_=built_in_types.TemporaryField[args]
                )
            # else:
            #    raise ValueError()

    def visit_Stencil(self, node: Stencil, **kwargs):
        self.primary_location = None
        self.generic_visit(node, **kwargs)
        self.primary_location = None
