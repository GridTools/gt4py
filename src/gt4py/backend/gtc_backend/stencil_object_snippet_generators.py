import jinja2
from eve.codegen import TemplatedGenerator
from mako import template as mako_tpl


ACCESSOR_CLASS_SRC = """
class _Accessor:
    def __init__(self, array, origin):
        self.array = array
        self.origin = origin

    def _shift(self, index):
        return tuple(i + offset for i, offset in zip(index, self.origin))

    def __getitem__(self, index):
        return self.array[self._shift(index)]

    def __setitem__(self, index, value):
        self.array[self._shift(index)] = value
"""


# TODO(Rico Häuselmann): Write unit tests for these


class FieldInfoGenerator(TemplatedGenerator):

    Computation_template = "{fields_metadata}"

    FieldsMetadata_template = mako_tpl.Template("{${', '.join(metas.values())}}")

    FieldMetadata_template = mako_tpl.Template(
        "'${name}': FieldInfo(access=AccessKind.${_this_node.access.name}, "
        "boundary=${boundary}, "
        "dtype=dtype('${_this_node.dtype.name.lower()}'))"
    )

    FieldBoundary_template = mako_tpl.Template(
        "Boundary((${', '.join(i)}), (${', '.join(j)}), (${', '.join(k)}))"
    )


class ParameterInfoGenerator(TemplatedGenerator):

    Computation_template = "{{}}"


class ComputationCallGenerator(TemplatedGenerator):

    Computation_template = mako_tpl.Template("computation.run(${', '.join(params)}, _domain_)")

    FieldDecl_template = "{name}"


class RunBodyGenerator(TemplatedGenerator):

    Computation_template = jinja2.Template("{{ '\\n'.join(params) }}")

    FieldDecl_template = "{name}_at = _Accessor({name}, _origin_['{name}'])"


class DomainInfoGenerator(TemplatedGenerator):

    Computation_template = "DomainInfo(parallel_axes=('I', 'J'), sequential_axis='K', ndims=3)"
