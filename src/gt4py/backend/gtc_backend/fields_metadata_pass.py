from eve.visitors import NodeModifier

from .gtir import (
    AccessKind,
    CartesianOffset,
    Computation,
    FieldAccess,
    FieldBoundary,
    FieldDecl,
    FieldMetadata,
    FieldsMetadata,
)


class FieldsMetadataPass(NodeModifier):
    def __init__(self):
        self.fields_metadata: FieldsMetadata = None

    def visit_Computation(self, node: Computation) -> Computation:
        self.fields_metadata = node.fields_metadata
        return self.generic_visit(node)

    def visit_FieldAccess(self, node: FieldAccess) -> FieldAccess:
        metadata = self.fields_metadata.metas.setdefault(
            node.name,
            FieldMetadata(
                name=node.name,
                access=AccessKind.READ_WRITE,
                boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
                dtype=None,
            ),
        )
        metadata.boundary.update_from_offset(node.offset)
        return self.generic_visit(node)

    def visit_FieldDecl(self, node: FieldDecl) -> FieldDecl:
        metadata = self.fields_metadata.metas.setdefault(
            node.name,
            FieldMetadata(
                name=node.name,
                access=AccessKind.READ_WRITE,
                boundary=FieldBoundary(i=(0, 0), j=(0, 0), k=(0, 0)),
                dtype=node.dtype,
            ),
        )
        metadata.dtype = node.dtype
        return self.generic_visit(node)
