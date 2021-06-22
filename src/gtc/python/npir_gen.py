# -*- coding: utf-8 -*-
import textwrap
from typing import Any, Collection, Tuple, Union

from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator
from gtc import common
from gtc.passes.gtir_legacy_extents import FIELD_EXT_T
from gtc.python import npir


__all__ = ["NpirGen"]


def op_delta_from_int(value: int) -> Tuple[str, str]:
    operator = ""
    delta = str(value)
    if value == 0:
        delta = ""
    elif value < 0:
        operator = " - "
        delta = str(abs(value))
    else:
        operator = " + "
    return operator, delta


class NpirGen(TemplatedGenerator):
    def visit_DataType(self, node: common.DataType, **kwargs: Any) -> Union[str, Collection[str]]:
        return f"np.{node.name.lower()}"

    def visit_BuiltInLiteral(self, node: common.BuiltInLiteral, **kwargs) -> str:
        if node is common.BuiltInLiteral.TRUE:
            return "True"
        elif node is common.BuiltInLiteral.FALSE:
            return "False"
        else:
            return self.generic_visit(node, **kwargs)

    Literal = FormatTemplate("{dtype}({value})")

    BroadCast = FormatTemplate("{expr}")

    Cast = FormatTemplate("np.array({expr}, dtype={dtype})")

    def visit_NumericalOffset(
        self, node: npir.NumericalOffset, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        operator, delta = op_delta_from_int(node.value)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    NumericalOffset = FormatTemplate("{op}{delta}")

    def visit_AxisOffset(self, node: npir.AxisOffset, **kwargs: Any) -> Union[str, Collection[str]]:
        offset = self.visit(node.offset)
        axis_name = self.visit(node.axis_name)
        if node.parallel:
            parts = [f"{axis_name.lower()}{offset}", f"{axis_name.upper()}{offset}"]
            if node.offset.value != 0:
                parts = [f"({part})" for part in parts]
            return self.generic_visit(node, from_visitor=":".join(parts), **kwargs)
        return self.generic_visit(node, from_visitor=f"{axis_name.lower()}_{offset}", **kwargs)

    AxisOffset = FormatTemplate("{from_visitor}")

    def visit_FieldSlice(self, node: npir.FieldSlice, **kwargs: Any) -> Union[str, Collection[str]]:
        kwargs.setdefault("mask_acc", "")
        return self.generic_visit(node, **kwargs)

    FieldSlice = FormatTemplate("{name}_[{i_offset}, {j_offset}, {k_offset}]{mask_acc}")

    def visit_EmptyTemp(
        self, node: npir.EmptyTemp, *, temp_name: str, **kwargs
    ) -> Union[str, Collection[str]]:
        shape = "_domain_"
        origin = [0, 0, 0]
        if extents := kwargs.get("field_extents", {}).get(temp_name):
            boundary = extents.to_boundary()
            i_total = sum(boundary[0])
            j_total = sum(boundary[1])
            shape = f"(_dI_ + {i_total}, _dJ_ + {j_total}, _dK_)"
            origin[:2] = boundary[0][0], boundary[1][0]
        return self.generic_visit(node, shape=shape, origin=origin, **kwargs)

    EmptyTemp = FormatTemplate("ShimmedView(np.zeros({shape}, dtype={dtype}), {origin})")

    NamedScalar = FormatTemplate("{name}")

    VectorTemp = FormatTemplate("{name}_")

    def visit_MaskBlock(self, node: npir.MaskBlock, **kwargs) -> Union[str, Collection[str]]:
        if isinstance(node.mask, npir.FieldSlice):
            mask_def = ""
        elif isinstance(node.mask, npir.BroadCast):
            mask_name = node.mask_name
            mask = self.generic_visit(node.mask)
            mask_def = f"{mask_name}_ = np.full((I - i, J - j, K - k), {mask})\n"
        else:
            mask_name = node.mask_name
            mask = self.generic_visit(node.mask)
            mask_def = f"{mask_name}_ = {mask}\n"
        return self.generic_visit(node, mask_def=mask_def, **kwargs)

    MaskBlock = JinjaTemplate(
        textwrap.dedent(
            """\
                {{ mask_def }}{% for stmt in body %}{{ stmt }}
                {% endfor %}
            """
        )
    )

    def visit_VectorAssign(
        self, node: npir.VectorAssign, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        mask_acc = ""
        if node.mask:
            mask_acc = f"[{self.visit(node.mask)}]"
        if isinstance(node.right, npir.EmptyTemp):
            kwargs["temp_name"] = node.left.name
        return self.generic_visit(node, mask_acc=mask_acc, **kwargs)

    VectorAssign = FormatTemplate("{left} = {right}")

    VectorArithmetic = FormatTemplate("({left} {op} {right})")

    VectorLogic = FormatTemplate("np.bitwise_{op}({left}, {right})")

    def visit_UnaryOperator(
        self, node: common.UnaryOperator, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node is common.UnaryOperator.NOT:
            return "np.bitwise_not"
        return self.generic_visit(node, **kwargs)

    def visit_VectorUnaryOp(
        self, node: npir.VectorUnaryOp, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return self.generic_visit(node, **kwargs)

    VectorUnaryOp = FormatTemplate("({op}({expr}))")

    VectorTernaryOp = FormatTemplate("np.where({cond}, {true_expr}, {false_expr})")

    def visit_LevelMarker(
        self, node: common.LevelMarker, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node: common.AxisBound, **kwargs: Any) -> Union[str, Collection[str]]:
        if node.offset == 0:
            return self.generic_visit(node, op="", delta="", **kwargs)
        operator = " - " if node.offset < 0 else " + "
        delta = str(abs(node.offset))
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    AxisBound = FormatTemplate("_d{level}_{op}{delta}")

    def visit_VerticalPass(
        self, node: npir.VerticalPass, **kwargs: Any
    ) -> Union[str, Collection[str]]:
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

    VerticalPass = JinjaTemplate(
        textwrap.dedent(
            """\
            ## -- begin vertical region --
            {% for assign in temp_defs %}{{ assign }}
            {% endfor %}k, K = {{ lower }}, {{ upper }}{{ for_loop_line }}
            {% for hregion in body %}{{ hregion | indent(body_indent, first=True) }}
            {% endfor %}## -- end vertical region --
            """
        )
    )

    def visit_HorizontalRegion(
        self, node: npir.HorizontalRegion, **kwargs
    ) -> Union[str, Collection[str]]:
        domain_padding = kwargs.get(
            "c_domain_padding", npir.DomainPadding(lower=(0, 0, 0), upper=(0, 0, 0))
        )
        lower = [
            domain_padding.lower[0] - node.padding.lower[0],
            domain_padding.lower[1] - node.padding.lower[1],
        ]
        upper = [
            domain_padding.upper[0] - node.padding.upper[0],
            domain_padding.upper[1] - node.padding.upper[1],
        ]

        if extents := kwargs.get("field_extents"):
            fields = set(node.iter_tree().if_isinstance(npir.FieldSlice).getattr("name")) & set(
                extents
            )
            lower[0] = min(extents[field].to_boundary()[0][0] for field in fields)
            lower[1] = min(extents[field].to_boundary()[1][0] for field in fields)
            upper[0] = min(extents[field].to_boundary()[0][1] for field in fields)
            upper[1] = min(extents[field].to_boundary()[1][1] for field in fields)
        return self.generic_visit(node, h_lower=lower, h_upper=upper, **kwargs)

    HorizontalRegion = JinjaTemplate(
        textwrap.dedent(
            """
            ## --- begin horizontal region --
            i, I = _di_ - {{ h_lower[0] }}, _dI_ + {{ h_upper[0] }}
            j, J = _dj_ - {{ h_lower[1] }}, _dJ_ + {{ h_upper[1] }}
            {% for assign in body %}{{ assign }}
            {% endfor %}## --- end horizontal region --

            """
        )
    )

    DomainPadding = FormatTemplate(
        textwrap.dedent(
            """\
            ## -- begin domain padding --
            _di_, _dj_, _dk_ = 0, 0, 0
            _dI_, _dJ_, _dK_ = _domain_
            ## -- end domain padding --
            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, *, field_extents: FIELD_EXT_T, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.params, "_domain_", "_origin_"]
        kwargs["field_extents"] = field_extents
        data_views = JinjaTemplate(
            textwrap.dedent(
                """\
                # -- begin data views --
                class ShimmedView:
                    def __init__(self, field, offsets):
                        self.field = field
                        self.offsets = offsets

                    def shim_key(self, key):
                        new_args = []
                        if not isinstance(key, tuple):
                            key = (key, )
                        for dim, idx in enumerate(key):
                            if isinstance(idx, slice):
                                start = 0 if idx.start is None else idx.start
                                stop = 0 if idx.stop is None else idx.stop
                                new_args.append(
                                    slice(start + self.offsets[dim], stop + self.offsets[dim], idx.step)
                                )
                            else:
                                new_args.append(idx + self.offsets[dim])
                        return tuple(new_args)

                    def __getitem__(self, key):
                        return self.field.__getitem__(self.shim_key(key))

                    def __setitem__(self, key, value):
                        return self.field.__setitem__(self.shim_key(key), value)

                {% for name in field_params %}{{ name }}_ = ShimmedView({{ name }}, _origin_["{{ name }}"])
                {% endfor %}# -- end data views --
                """
            )
        ).render(
            field_params=node.field_params,
            lower=node.domain_padding.lower,
        )
        kwargs["c_domain_padding"] = node.domain_padding
        return self.generic_visit(
            node, signature=", ".join(signature), data_views=data_views, **kwargs
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            import numpy as np


            def run({{ signature }}):
                {{ domain_padding | indent(4) }}
                {{ data_views | indent(4) }}
                {% for pass in vertical_passes %}
                {{ pass | indent(4) }}
                {% endfor %}
            """
        )
    )

    def visit_NativeFunction(
        self, node: common.NativeFunction, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node == common.NativeFunction.MIN:
            return "minimum"
        elif node == common.NativeFunction.MAX:
            return "maximum"
        elif node == common.NativeFunction.POW:
            return "power"
        return self.generic_visit(node, **kwargs)

    NativeFuncCall = FormatTemplate("np.{func}({', '.join(arg for arg in args)})")
