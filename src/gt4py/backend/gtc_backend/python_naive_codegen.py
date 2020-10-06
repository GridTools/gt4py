from eve import codegen
from mako import template as mako_tpl
import jinja2


class PythonNaiveCodegen(codegen.TemplatedGenerator):

    Computation_template = mako_tpl.Template(
        """
${ '\\n'.join(params) }
${ '\\n'.join(stencils) }
""")

    # TODO might not be the right solutions as FieldDecl might be used in different context
    FieldDecl_template = "{name}_at = _Accessor({name}, _origin_['{name}'])"

    Stencil_template = jinja2.Template(
        """{{'\\n'.join(vertical_loops)}}"""
    )

    VerticalLoop_template = mako_tpl.Template("""
${ '\\n'.join(vertical_intervals) }""")

    VerticalInterval_template = jinja2.Template("""
for K in range(0, _domain_[2]):
    {{'\\n'.join(horizontal_loops)|indent(4)}}""")

    # AxisBound_template = mako_tpl.Template(
    #     "<%! from gt4py.backend.gtc_backend.common import LevelMarker %>\\\n"
    #     "% if _this_node.level == LevelMarker.START:\n"
    #     "${ offset }\\\n"
    #     "% elif _this_node.level == LevelMarker.END:\n"
    #     "${ _this_node.offset and -_this_node.offset or None }\\\n"
    #     "% endif"
    # )

    HorizontalLoop_template = jinja2.Template(
        """
for I in range(_domain_[0]):
    for J in range(_domain_[1]):
        {{stmt|indent(8)}}""")

    AssignStmt_template = "{left} = {right}"

    FieldAccess_template = "{name}_at[{offset}]"

    CartesianOffset_template = "I + {i}, J + {j}, K + {k}"

    BinaryOp_template = "{left} {op} {right}"

    Literal_template = "{value}"
