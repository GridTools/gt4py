# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
import hashlib
import types
from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py._core import definitions as core_defs
from gt4py.eve import NodeTranslator, concepts, traits
from gt4py.next import errors
from gt4py.next.ffront import dialect_ast_enums, experimental, fbuiltins, gtcallable
from gt4py.next.ffront.ast_passes import single_static_assign as ssa
from gt4py.next.type_system import type_translation


_BUILTINS: dict[str, Any] = {
    **fbuiltins.BUILTINS,
    **{name: getattr(experimental, name) for name in experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES},
}


def _builtin_name(value: Any) -> str | None:
    return next((name for name, builtin in _BUILTINS.items() if builtin is value), None)


def _operator_name(value: Any) -> str | None:
    """A symbol name for a `GTCallable` reached through a module, derived from its definition."""
    if isinstance(value, gtcallable.GTCallable) and (
        definition := getattr(value, "definition", None)
    ):
        return _operator_name_from_definition(definition)
    return None


def _operator_name_from_definition(definition: types.FunctionType) -> str:
    # derived from the import path, not the file path, so the symbol (and with it the IR and
    # the on-disk cache keys) is the same for every checkout of the same code
    key = f"{definition.__module__}.{definition.__qualname__}".encode()
    return f"{definition.__name__}_{hashlib.sha256(key).hexdigest()[:8]}"


@dataclasses.dataclass
class ClosureVarFolding(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace references to closure variables or their attributes with constants.

    `Name` nodes that refer to closure variables are replaced by `Constant`
    nodes. `Attribute` nodes that refer to attributes of closure variables
    are recursively replaced by `Constant` nodes.

    References that reach a value through a module (`np.pi`, `gtx.where`,
    `helpers.helper`) or refer to a builtin under another name are resolved by
    value: scalars become `Constant`s, builtins and `GTCallable`s become a
    `Name` of a canonical, value-derived symbol that is added to `closure_vars`
    and to the function's closure variable symbols.
    """

    closure_vars: dict[str, Any]
    _added_closure_vars: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def apply(
        cls, node: foast.FunctionDefinition | foast.FieldOperator, closure_vars: dict[str, Any]
    ) -> foast.FunctionDefinition:
        return cls(closure_vars=closure_vars).visit(node)

    def _closure_value(
        self,
        node: foast.Expr,
        current_closure_vars: list[foast.Symbol],
        symtable: dict[str, foast.Symbol],
    ) -> Any:
        """Python value of a closure variable reference, resolved through module attributes."""
        if isinstance(node, foast.Name):
            if node.id in symtable and symtable[node.id] in current_closure_vars:
                return self.closure_vars[node.id]
        elif isinstance(node, foast.Attribute):
            base = self._closure_value(node.value, current_closure_vars, symtable)
            if isinstance(base, types.ModuleType):
                if not hasattr(base, node.attr):
                    raise errors.MissingAttributeError(node.location, node.attr)
                return getattr(base, node.attr)
        return _MISSING

    def visit_Name(
        self,
        node: foast.Name,
        current_closure_vars: list[foast.Symbol],
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Name | foast.Constant:
        value = self._closure_value(node, current_closure_vars, symtable)
        if isinstance(value, type_translation.ConstantPythonNamespaceObject):
            return foast.Constant(value=value, location=node.location)
        if (name := _builtin_name(value)) is not None and name != node.id:
            return self._reference(name, value, node.location, current_closure_vars, symtable)
        return node

    def visit_Attribute(
        self,
        node: foast.Attribute,
        current_closure_vars: list[foast.Symbol],
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Constant | foast.Attribute | foast.Name:
        value = self.visit(
            node.value, current_closure_vars=current_closure_vars, symtable=symtable, **kwargs
        )
        if isinstance(value, foast.Constant):
            if hasattr(value.value, node.attr):
                const_value = getattr(value.value, node.attr)
                if isinstance(const_value, enum.Enum):
                    const_value = const_value.value
                return foast.Constant(value=const_value, location=node.location)
            raise errors.MissingAttributeError(node.location, node.attr)
        if isinstance(node.value, (foast.Name, foast.Attribute)):
            leaf = self._closure_value(node, current_closure_vars, symtable)
            if (name := _builtin_name(leaf) or _operator_name(leaf)) is not None:
                return self._reference(name, leaf, node.location, current_closure_vars, symtable)
            if core_defs.is_scalar_type(leaf):
                return foast.Constant(value=leaf, location=node.location)
        return node

    def _reference(
        self,
        name: str,
        value: Any,
        location: concepts.SourceLocation,
        current_closure_vars: list[foast.Symbol],
        symtable: dict[str, foast.Symbol],
    ) -> foast.Name:
        if any(
            ssa.original_name(symbol) == name and symtable[symbol] not in current_closure_vars
            for symbol in symtable
        ):
            raise errors.DSLError(
                location,
                f"Reference resolves to '{name}', which is shadowed by a local variable or "
                "parameter of the same name.",
            )
        if name not in self.closure_vars:
            self.closure_vars[name] = value
            self._added_closure_vars[name] = value
        elif self.closure_vars[name] is not value:
            raise errors.DSLError(
                location,
                f"Reference resolves to '{name}', but '{name}' already refers to a different "
                "value in this function.",
            )
        return foast.Name(id=name, location=location)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        new_node: foast.FunctionDefinition = self.generic_visit(
            node, current_closure_vars=node.closure_vars, **kwargs
        )
        if not self._added_closure_vars:
            return new_node
        new_symbols: list[foast.Symbol] = [
            foast.Symbol(
                id=name,
                type=type_translation.from_value(value),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=new_node.location,
            )
            for name, value in self._added_closure_vars.items()
        ]
        return foast.FunctionDefinition(
            id=new_node.id,
            params=new_node.params,
            body=new_node.body,
            closure_vars=[*new_node.closure_vars, *new_symbols],
            type=new_node.type,
            location=new_node.location,
        )


_MISSING = object()
