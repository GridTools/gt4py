from typing import Any, Dict

from eve.visitors import NodeTranslator

from gt4py.gtc.gtir import FieldAccess, FieldDecl, FieldsMetadataBuilder, Stencil


class FieldsMetadataPass(NodeTranslator):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.metas = FieldsMetadataBuilder()

    def visit_Stencil(self, node: Stencil) -> Stencil:
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
