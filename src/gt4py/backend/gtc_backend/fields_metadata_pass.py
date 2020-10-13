from typing import Any, Dict

from eve.visitors import NodeTranslator

from .gtir import (
    AccessKind,
    Computation,
    FieldAccess,
    FieldBoundaryAccumulator,
    FieldDecl,
    FieldMetadata,
    FieldsMetadata,
)


# TODO(Rico Häuselmann) write unit tests


class FieldsMetadataPass(NodeTranslator):
    def __init__(self, *, memo: Dict = None, **kwargs: Any) -> None:
        super().__init__(memo=memo, **kwargs)
        self.metas = {}

    def visit_Computation(self, node: Computation) -> Computation:
        new_node = self.generic_visit(node)
        for meta in self.metas.values():
            meta["boundary"] = meta["boundary"].to_boundary()
        new_node.fields_metadata = FieldsMetadata(
            metas={name: FieldMetadata(**meta) for name, meta in self.metas.items()}
        )
        return new_node

    def visit_FieldAccess(self, node: FieldAccess) -> FieldAccess:
        metadata = self.metas.setdefault(
            node.name,
            {
                "name": node.name,
                "access": AccessKind.READ_WRITE,
                "boundary": FieldBoundaryAccumulator(),
            },
        )
        metadata["boundary"].update_from_offset(node.offset)
        return self.generic_visit(node)

    def visit_FieldDecl(self, node: FieldDecl) -> FieldDecl:
        metadata = self.metas.setdefault(
            node.name,
            {
                "name": node.name,
                "access": AccessKind.READ_WRITE,
                "boundary": FieldBoundaryAccumulator(),
                "dtype": node.dtype,
            },
        )
        metadata["dtype"] = node.dtype
        return self.generic_visit(node)
