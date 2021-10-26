from typing import Any, Dict, List, Type

import pydantic

from eve import Node
from eve.traits import SymbolTableTrait
from eve.type_definitions import SymbolRef
from eve.typingx import RootValidatorType, RootValidatorValuesType
from eve.visitors import NodeVisitor


def validate_symbol_refs() -> RootValidatorType:
    """Validate that symbol refs are found in a symbol table valid at the current scope."""

    def _impl(
        cls: Type[pydantic.BaseModel], values: RootValidatorValuesType
    ) -> RootValidatorValuesType:
        class SymtableValidator(NodeVisitor):
            def __init__(self) -> None:
                self.missing_symbols: List[str] = []

            def visit_Node(self, node: Node, *, symtable: Dict[str, Any], **kwargs: Any) -> None:
                for name, metadata in node.__node_children__.items():
                    if isinstance(metadata["definition"].type_, type) and issubclass(
                        metadata["definition"].type_, SymbolRef
                    ):
                        if getattr(node, name) and getattr(node, name) not in symtable:
                            self.missing_symbols.append(getattr(node, name))

                if isinstance(node, SymbolTableTrait):
                    symtable = {**symtable, **node.symtable_}
                self.generic_visit(node, symtable=symtable, **kwargs)

            @classmethod
            def apply(cls, node: Node, *, symtable: Dict[str, Any]) -> List[str]:
                instance = cls()
                instance.visit(node, symtable=symtable)
                return instance.missing_symbols

        missing_symbols = []
        for v in values.values():
            missing_symbols.extend(SymtableValidator.apply(v, symtable=values["symtable_"]))

        if len(missing_symbols) > 0:
            raise ValueError("Symbols {} not found.".format(missing_symbols))

        return values

    return pydantic.root_validator(allow_reuse=True, skip_on_failure=True)(_impl)
