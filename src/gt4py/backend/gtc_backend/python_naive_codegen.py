import jinja2
from eve import codegen
from mako import template as mako_tpl


class PythonNaiveCodegen(codegen.TemplatedGenerator):

    Computation_template = jinja2.Template(
        """
def default_domain(*args):
    lengths = zip(*(i.shape for i in args))
    return tuple(max(*length) for length in lengths)


def run({{', '.join(_this_node.param_names)}}, _domain_=None):
    if _domain_ is None:
        _domain_ = default_domain({{', '.join(_this_node.param_names)}})\
{{ '\\n'.join(stencils) | indent(4)}}
"""
    )

    Stencil_template = jinja2.Template("""{{'\\n'.join(vertical_loops)}}""")

    VerticalLoop_template = mako_tpl.Template(
        """
${ '\\n'.join(vertical_intervals) }"""
    )

    VerticalInterval_template = jinja2.Template(
        """
for K in range({{start}}, {{end}}):\
    {{'\\n'.join(horizontal_loops)|indent(4)}}"""
    )

    HorizontalLoop_template = jinja2.Template(
        """
for I in range(_domain_[0]):
    for J in range(_domain_[1]):
        {{stmt|indent(8)}}"""
    )

    AssignStmt_template = "{left} = {right}"

    FieldAccess_template = "{name}[{offset}]"

    CartesianOffset_template = "I + {i}, J + {j}, K + {k}"

    BinaryOp_template = "{left} {op} {right}"

    Literal_template = "{value}"

    AxisBound_template = jinja2.Template(
        """\
{% if _this_node.level.name == 'END' %}\
_domain_[2]{{' - ' + offset if _this_node.offset > 0 else ''}}\
{% else %}\
{{offset}}\
{% endif %}\
"""
    )
