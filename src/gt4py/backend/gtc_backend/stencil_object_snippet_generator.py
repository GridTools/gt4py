from eve.codegen import TemplatedGenerator
from mako import template as mako_tpl


class StencilObjectSnippetGenerator(TemplatedGenerator):

    FieldsMetadata_template = mako_tpl.Template("{${', '.join(metas.values())}}")

    FieldMetadata_template = mako_tpl.Template(
        "'${name}': FieldInfo(access=AccessKind.${_this_node.access.name}, "
        "boundary=${boundary}, "
        "dtype=dtype('${_this_node.dtype.name.lower()}'))"
    )

    FieldBoundary_template = mako_tpl.Template(
        "Boundary((${', '.join(i)}), (${', '.join(j)}), (${', '.join(k)}))"
    )
