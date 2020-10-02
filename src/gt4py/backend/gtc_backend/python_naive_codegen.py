from eve import codegen
from mako import template as mako_tpl


class PythonNaiveCodegen(codegen.TemplatedGenerator):

    Computation_template = mako_tpl.Template(
        "def ${ name }(${ ', '.join(params) }):\n"
        "% for stencil in stencils:\n"
        "    ${ stencil | trim }\n\n"
        "% endfor"
    )

    FieldDecl_template = "{name}"

    Stencil_template = mako_tpl.Template(
        "    % for vloop in vertical_loops:\n" "    ${ vloop | trim }\n" "    % endfor"
    )

    VerticalLoop_template = mako_tpl.Template(
        "    with computation(${ loop_order }):\n"
        "    % for interval in vertical_intervals:\n"
        "        ${ interval | trim }\n"
        "    % endfor"
    )

    VerticalInterval_template = mako_tpl.Template(
        "        with interval(${ start }, ${ end }):\n"
        "        % for hloop in horizontal_loops:\n"
        "            ${ hloop | trim }\n"
        "        % endfor"
    )

    AxisBound_template = mako_tpl.Template(
        "<%! from gt4py.backend.gtc_backend.common import LevelMarker %>\\\n"
        "% if _this_node.level == LevelMarker.START:\n"
        "${ offset }\\\n"
        "% elif _this_node.level == LevelMarker.END:\n"
        "${ _this_node.offset and -_this_node.offset or None }\\\n"
        "% endif"
    )

    HorizontalLoop_template = "{stmt}"

    AssignStmt_template = "{left} = {right}"

    FieldAccess_template = "{name}[{offset}]"

    CartesianOffset_template = "{i}, {j}, {k}"

    BinaryOp_template = "{left} {op} {right}"
