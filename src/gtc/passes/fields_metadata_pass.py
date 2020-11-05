from typing import Any, Dict

from eve.visitors import NodeTranslator

from gtc.gtir import Computation, FieldAccess, FieldDecl, FieldsMetadataBuilder


class FieldsMetadataPass(NodeTranslator):
    def __init__(self, *, memo: Dict = None, **kwargs: Any) -> None:
        super().__init__(memo=memo, **kwargs)
        self.metas = FieldsMetadataBuilder()

    def visit_Computation(self, node: Computation) -> Computation:
        new_node = self.generic_visit(node)
        new_node.fields_metadata = self.metas.build()
        return new_node

    def visit_FieldAccess(self, node: FieldAccess) -> FieldAccess:
        metadata = self.metas.get_or_create(node)
        metadata.boundary.update_from_offset(node.offset)
        return self.generic_visit(node)

    def visit_FieldDecl(self, node: FieldDecl) -> FieldDecl:
        self.metas.get_or_create(node).dtype(node.dtype)
        return self.generic_visit(node)
