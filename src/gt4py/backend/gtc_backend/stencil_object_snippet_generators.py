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


class FieldInfoGenerator(TemplatedGenerator):

    Stencil = FormatTemplate("{fields_metadata}")

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

    Stencil = FormatTemplate("{{}}")


class ComputationCallGenerator(TemplatedGenerator):

    Stencil = MakoTemplate(
        "computation.run("
        "${', '.join(f'{p}={p}' for p in params)}, _domain_=_domain_, _origin_=_origin_)"
    )

    FieldDecl = FormatTemplate("{name}")


class RunBodyGenerator(TemplatedGenerator):

    Stencil = JinjaTemplate("{{ '\\n'.join(params) }}")

    FieldDecl = FormatTemplate("{name}_at = _Accessor({name}, _origin_['{name}'])")


class DomainInfoGenerator(TemplatedGenerator):

    Stencil = FormatTemplate("DomainInfo(parallel_axes=('I', 'J'), sequential_axis='K', ndims=3)")
