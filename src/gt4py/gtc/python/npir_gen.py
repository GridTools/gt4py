from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator

from gt4py.gtc import common


__all__ = ["NpirGen"]


VERTICAL_PASS_JTPL = """
k, K = DOMAIN_k{{ lower }}, DOMAIN_K{{ upper }}
{{ for_loop_line }}
{% for assign in body %}\
{{ ' ' * body_indent }}{{ assign | indent(body_indent) }}
{% endfor %}
"""


class NpirGen(TemplatedGenerator):

    Literal = FormatTemplate("np.{_this_node.dtype.name.lower()}({value})")

    NumericalOffset = FormatTemplate("{op}{delta}")

    ParallelOffset = FormatTemplate("{axis_name.lower()}{offset}:{axis_name.upper()}{offset}")

    SequentialOffset = FormatTemplate("{axis_name.lower()}_{offset}")

    FieldSlice = FormatTemplate("{name}[{i_offset}, {j_offset}, {k_offset}]")

    VectorAssign = FormatTemplate("{left} = {right}")

    VectorBinOp = FormatTemplate("({left} {op} {right})")

    VerticalPass = JinjaTemplate(VERTICAL_PASS_JTPL)

    def visit_NumericalOffset(self, node, **kwargs):
        op = ""
        delta = node.value
        if node.value == 0:
            delta = ""
        elif node.value < 0:
            op = " - "
            delta = abs(node.value)
        else:
            op = " + "
        return self.generic_visit(node, op=op, delta=delta, **kwargs)

    def visit_ParallelOffset(self, node, **kwargs):
        if node.offset.value != 0:
            parts = self.generic_visit(node, **kwargs).split(":")
            return ":".join(f"({part})" for part in parts)
        return self.generic_visit(node, **kwargs)

    def visit_VerticalPass(self, node, **kwargs):
        for_loop_line = ""
        body_indent = 0
        if node.direction == common.LoopOrder.FORWARD:
            for_loop_line = "for k_ in range(k, K):"
            body_indent = 4
        elif node.direction == common.LoopOrder.BACKWARD:
            for_loop_line = "for k_ in range(K, k, -1):"
            body_indent = 4
        return self.generic_visit(
            node, for_loop_line=for_loop_line, body_indent=body_indent, **kwargs
        )
