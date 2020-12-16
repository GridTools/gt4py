import textwrap
from typing import Tuple

from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator

from gt4py.gtc import common


__all__ = ["NpirGen"]


def op_delta_from_int(value: int) -> Tuple[str, str]:
    operator = ""
    delta = value
    if value == 0:
        delta = ""
    elif value < 0:
        operator = " - "
        delta = abs(value)
    else:
        operator = " + "
    return operator, delta


class NpirGen(TemplatedGenerator):

    Literal = FormatTemplate("np.{_this_node.dtype.name.lower()}({value})")

    def visit_NumericalOffset(self, node, **kwargs):
        operator, delta = op_delta_from_int(node.value)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    NumericalOffset = FormatTemplate("{op}{delta}")

    def visit_AxisOffset(self, node, **kwargs):
        offset = self.visit(node.offset)
        axis_name = self.visit(node.axis_name)
        if node.parallel:
            parts = [f"{axis_name.lower()}{offset}", f"{axis_name.upper()}{offset}"]
            if node.offset.value != 0:
                parts = [f"({part})" for part in parts]
            return self.generic_visit(node, from_visitor=":".join(parts), **kwargs)
        return self.generic_visit(node, from_visitor=f"{axis_name.lower()}_{offset}", **kwargs)

    AxisOffset = FormatTemplate("{from_visitor}")

    FieldSlice = FormatTemplate("{name}_[{i_offset}, {j_offset}, {k_offset}]")

    VectorAssign = FormatTemplate("{left} = {right}")

    VectorArithmetic = FormatTemplate("({left} {op} {right})")

    def visit_LevelMarker(self, node, **kwargs):
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node, **kwargs):
        delta = ""
        operator = ""
        if node.offset > 0:
            operator = " - " if node.level == common.LevelMarker.END else " + "
            delta = str(node.offset)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    AxisBound = FormatTemplate("DOMAIN_{level}{op}{delta}")

    def visit_VerticalPass(self, node, **kwargs):
        for_loop_line = ""
        body_indent = 0
        if node.direction == common.LoopOrder.FORWARD:
            for_loop_line = "\nfor k_ in range(k, K):"
            body_indent = 4
        elif node.direction == common.LoopOrder.BACKWARD:
            for_loop_line = "\nfor k_ in range(K-1, k-1, -1):"
            body_indent = 4
        return self.generic_visit(
            node, for_loop_line=for_loop_line, body_indent=body_indent, **kwargs
        )

    ## TODO: change to use AxisBounds, k can also be derived from DOMAIN_K and so on...
    VerticalPass = JinjaTemplate(
        textwrap.dedent(
            """\
            ## -- begin vertical region --
            k, K = {{ lower }}, {{ upper }}{{ for_loop_line }}
            {% for assign in body %}{{ assign | indent(body_indent, first=True) }}
            {% endfor %}## -- end vertical region --
            """
        )
    )

    DomainPadding = FormatTemplate(
        textwrap.dedent(
            """\
            ## -- begin domain padding --
            i, j, k = {lower[0]}, {lower[1]}, {lower[2]}
            _ui, _uj, _uk = {upper[0]}, {upper[1]}, {upper[2]}
            _di, _dj, _dk = _domain_
            I, J, K = _di + i, _dj + j, _dk + k
            DOMAIN_k = k
            DOMAIN_K = K
            ## -- end domain padding --
            """
        )
    )

    def visit_Computation(self, node, **kwargs):
        signature = ["*", *node.params, "_domain_", "_origin_"]
        data_views = JinjaTemplate(
            textwrap.dedent(
                """\
                # -- begin data views --
                {% for name in field_params %}{{ name }}_ = {{ name }}[
                    (_origin_["{{ name }}"][0] - i):(_origin_["{{ name }}"][0] + _di + _ui),
                    (_origin_["{{ name }}"][1] - j):(_origin_["{{ name }}"][1] + _dj + _uj),
                    (_origin_["{{ name }}"][2] - k):(_origin_["{{ name }}"][2] + _dk + _uk)
                ]
                {% endfor %}# -- end data views --
                """
            )
        ).render(field_params=node.field_params)
        return self.generic_visit(
            node, signature=", ".join(signature), data_views=data_views, **kwargs
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            def run({{ signature }}):
                {{ domain_padding | indent(4) }}
                {{ data_views | indent(4) }}
                {% for pass in vertical_passes %}
                {{ pass | indent(4) }}
                {% endfor %}
            """
        )
    )
