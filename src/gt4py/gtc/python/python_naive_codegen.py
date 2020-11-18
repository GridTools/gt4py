from eve.codegen import FormatTemplate, JinjaTemplate, MakoTemplate, TemplatedGenerator


class PythonNaiveCodegen(TemplatedGenerator):

    Stencil = JinjaTemplate(
        """
def default_domain(*args):
    lengths = zip(*(i.shape for i in args))
    return tuple(max(*length) for length in lengths)


def run({{', '.join(_this_node.param_names)}}, _domain_=None):
    if _domain_ is None:
        _domain_ = default_domain({{', '.join(_this_node.param_names)}})\
{{ '\\n'.join(vertical_loop) | indent(4)}}
"""
    )

    VerticalLoop = JinjaTemplate(
        """
for K in range({{interval}}):\
    {{'\\n'.join(body)|indent(4)}}"""
    )

    Interval = FormatTemplate("{start}, {end}")

    ParAssignStmt = JinjaTemplate(
        """
for I in range(_domain_[0]):
    for J in range(_domain_[1]):
        {{left|indent(8)}} = {{right}}"""
    )

    FieldAccess = FormatTemplate("{name}[{offset}]")

    ScalarAccess = FormatTemplate("{name}")

    CartesianOffset = FormatTemplate("I + {i}, J + {j}, K + {k}")

    BinaryOp = FormatTemplate("{left} {op} {right}")

    Literal = FormatTemplate("{value}")

    AxisBound = JinjaTemplate(
        """\
{% if _this_node.level.name == 'END' %}\
_domain_[2]{{' - ' + offset if _this_node.offset > 0 else ''}}\
{% else %}\
{{offset}}\
{% endif %}\
"""
    )
