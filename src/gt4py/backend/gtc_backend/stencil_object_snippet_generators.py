from eve.codegen import FormatTemplate, JinjaTemplate, MakoTemplate, TemplatedGenerator


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

    Computation = FormatTemplate("{fields_metadata}")

    FieldsMetadata = MakoTemplate("{${', '.join(metas.values())}}")

    FieldMetadata = MakoTemplate(
        "'${name}': FieldInfo(access=AccessKind.${_this_node.access.name}, "
        "boundary=${boundary}, "
        "dtype=dtype('${_this_node.dtype.name.lower()}'))"
    )

    FieldBoundary = MakoTemplate(
        "Boundary((${', '.join(i)}), (${', '.join(j)}), (${', '.join(k)}))"
    )


class ParameterInfoGenerator(TemplatedGenerator):

    Computation = FormatTemplate("{{}}")


class ComputationCallGenerator(TemplatedGenerator):

    Computation = MakoTemplate("computation.run(${', '.join(params)}, _domain_)")

    FieldDecl = FormatTemplate("{name}")


class RunBodyGenerator(TemplatedGenerator):

    Computation = JinjaTemplate("{{ '\\n'.join(params) }}")

    FieldDecl = FormatTemplate("{name}_at = _Accessor({name}, _origin_['{name}'])")


class DomainInfoGenerator(TemplatedGenerator):

    Computation = FormatTemplate(
        "DomainInfo(parallel_axes=('I', 'J'), sequential_axis='K', ndims=3)"
    )
