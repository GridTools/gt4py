# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import functools
import textwrap
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Tuple, Union

from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator
from eve.visitors import NodeVisitor
from gt4py.definitions import Extent
from gtc import common
from gtc.numpy import npir


__all__ = ["NpirCodegen"]


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


ORIGIN_CORRECTED_VIEW_CLASS = textwrap.dedent(
    """\
    class ShimmedView:
        def __init__(self, field, offsets):
            self.field = field
            self.offsets = offsets

        def shim_key(self, key):
            new_args = []
            if not isinstance(key, tuple):
                key = (key, )
            for dim, idx in enumerate(key):
                offset = self.offsets[dim]
                if isinstance(idx, slice):
                    if idx.start is None or idx.stop is None:
                        assert offset == 0
                    new_args.append(
                        slice(idx.start + offset, idx.stop + offset, idx.step) if offset else idx
                    )
                else:
                    new_args.append(idx + offset)
            if not isinstance(new_args[2], (numbers.Integral, slice)):
                assert isinstance(new_args[0], slice) and isinstance(new_args[1], slice)
                new_args[:2] = np.broadcast_arrays(
                    np.expand_dims(
                        np.arange(new_args[0].start, new_args[0].stop),
                        axis=tuple(i for i in range(self.field.ndim) if i != 0)
                    ),
                    np.expand_dims(
                        np.arange(new_args[1].start, new_args[1].stop),
                        axis=tuple(i for i in range(self.field.ndim) if i != 1)
                    ),
                )
            return tuple(new_args)

        def __getitem__(self, key):
            return self.field.data.__getitem__(self.shim_key(key))

        def __setitem__(self, key, value):
            return self.field.__setitem__(self.shim_key(key), value)
    """
)


VARIABLE_OFFSET_FUNCTION = textwrap.dedent(
    """
    def var_k_expr(expr, k):

        k_indices = np.arange(expr.shape[2]) + k
        all_nonk_axes = tuple(i for i in range(expr.ndim) if i != 2)
        expanded_k_indices = np.expand_dims(k_indices, axis=all_nonk_axes)
        return expanded_k_indices + expr
    """
)


def slice_to_extent(acc: npir.FieldSlice) -> Extent:
    return Extent(
        (
            [acc.i_offset.offset.value] * 2 if acc.i_offset else [0, 0],
            [acc.j_offset.offset.value] * 2 if acc.j_offset else [0, 0],
            [0, 0],
        )
    )


HorizontalExtent = Tuple[Tuple[int, int], Tuple[int, int]]


class ExtentCalculator(NodeVisitor):
    @dataclass
    class Context:
        field_extents: Dict[str, Extent] = field(default_factory=dict)
        block_extents: Dict[int, HorizontalExtent] = field(default_factory=dict)

    def visit_Computation(self, node: npir.Computation):
        ctx = self.Context()
        for vertical_pass in reversed(node.vertical_passes):
            self.visit(vertical_pass, ctx=ctx)
        return ctx.field_extents, ctx.block_extents

    def visit_VerticalPass(self, node: npir.VerticalPass, *, ctx: Context):
        for block in reversed(node.body):
            self.visit(block, ctx=ctx)

    def visit_HorizontalBlock(self, node: npir.HorizontalBlock, *, ctx: Context):
        writes = (
            node.iter_tree()
            .if_isinstance(npir.VectorAssign)
            .getattr("left")
            .if_isinstance(npir.FieldSlice)
            .getattr("name")
            .to_set()
        )
        extent = functools.reduce(
            lambda ext, name: ext | ctx.field_extents.get(name, Extent.zeros()),
            writes,
            Extent.zeros(),
        )
        ctx.block_extents[id(node)] = extent

        for acc in node.iter_tree().if_isinstance(npir.FieldSlice).to_list():
            ctx.field_extents[acc.name] = ctx.field_extents.get(acc.name, Extent.zeros()).union(
                extent + slice_to_extent(acc)
            )


class NpirCodegen(TemplatedGenerator):
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
        lpar, rpar = "()" if offset else ("", "")
        variant = self.AxisOffset_parallel if node.parallel else self.AxisOffset_serial
        rendered = variant.render(lpar=lpar, rpar=rpar, axis_name=axis_name, offset=offset)
        return self.generic_visit(node, parallel_or_serial_variant=rendered, **kwargs)

    AxisOffset_parallel = JinjaTemplate(
        "{{ lpar }}{{ axis_name | lower }}{{ offset }}{{ rpar }}:{{ lpar }}{{ axis_name | upper }}{{ offset }}{{ rpar }}"
    )

    AxisOffset_serial = JinjaTemplate(
        "{{ lpar }}{{ axis_name | lower }}_{{ offset }}{{ rpar }}:({{ axis_name | lower }}_{{ offset}} + 1)"
    )

    AxisOffset = FormatTemplate("{parallel_or_serial_variant}")

    def visit_FieldDecl(self, node: npir.FieldDecl, **kwargs) -> Union[str, Collection[str]]:
        if all(node.dimensions):
            return ""
        shape_idx = iter(range(3))
        origin_idx = iter(range(3))
        shape = ", ".join(
            [f"{node.name}.shape[{next(shape_idx)}]" if dim else "1" for dim in node.dimensions]
        )
        origin = ", ".join(
            [
                f"_origin_['{node.name}'][{next(origin_idx)}]" if dim else "0"
                for dim in node.dimensions
            ]
        )
        return self.generic_visit(node, shape=shape, origin=origin, **kwargs)

    FieldDecl = FormatTemplate(
        "{name} = np.reshape({name}, ({shape}))\n_origin_['{name}'] = [{origin}]"
    )

    def visit_VariableKOffset(
        self, node: npir.VariableKOffset, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return self.generic_visit(
            node,
            counter="k_" if kwargs["is_serial"] else "k",
            var_offset=self.visit(node.k, **kwargs),
        )

    VariableKOffset = FormatTemplate("var_k_expr({var_offset}, {counter})")

    def visit_FieldSlice(
        self, node: npir.FieldSlice, mask_acc="", *, is_serial=False, **kwargs: Any
    ) -> Union[str, Collection[str]]:

        offset = [node.i_offset, node.j_offset, node.k_offset]

        offset_str = ", ".join(
            self.visit(off, is_serial=is_serial, **kwargs) if off else ":" for off in offset
        )

        if node.data_index:
            offset_str += ", " + ", ".join(self.visit(x, **kwargs) for x in node.data_index)

        if mask_acc and any(off is None for off in offset):
            k_size = "1" if is_serial else "K - k"
            arr_expr = f"np.broadcast_to({node.name}_[{offset_str}], (I - i, J - j, {k_size}))"
        else:
            arr_expr = f"{node.name}_[{offset_str}]"

        return f"{arr_expr}{mask_acc}"

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
            assert "is_serial" in kwargs
            mask_name = node.mask_name
            mask = self.visit(node.mask)
            k_size = "1" if kwargs["is_serial"] else "K - k"
            mask_def = f"{mask_name}_ = np.full((I - i, J - j, {k_size}), {mask})\n"
        else:
            mask_name = node.mask_name
            mask = self.visit(node.mask)
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
            mask_acc = f"[{self.visit(node.mask, **kwargs)}]"
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

    VectorUnaryOp = FormatTemplate("({op}({expr}))")

    VectorTernaryOp = FormatTemplate("np.where({cond}, {true_expr}, {false_expr})")

    def visit_LevelMarker(
        self, node: common.LevelMarker, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node: common.AxisBound, **kwargs: Any) -> Union[str, Collection[str]]:
        operator, delta = op_delta_from_int(node.offset)
        return self.generic_visit(node, op=operator, delta=delta, **kwargs)

    AxisBound = FormatTemplate("_d{level}_{op}{delta}")

    def visit_LoopOrder(self, node: common.LoopOrder, **kwargs) -> Union[str, Collection[str]]:
        if node is common.LoopOrder.FORWARD:
            return "for k_ in range(k, K):"
        elif node is common.LoopOrder.BACKWARD:
            return "for k_ in range(K-1, k-1, -1):"
        return ""

    def visit_VerticalPass(self, node: npir.VerticalPass, **kwargs):
        return self.generic_visit(
            node, is_serial=(node.direction != common.LoopOrder.PARALLEL), **kwargs
        )

    VerticalPass = JinjaTemplate(
        textwrap.dedent(
            """\
            # -- begin vertical block --{% set body_indent = 0 %}
            {% for assign in temp_defs %}{{ assign }}
            {% endfor %}k, K = {{ lower }}, {{ upper }}{% if direction %}
            {{ direction }}{% set body_indent = 4 %}{% endif %}
            {% for hblock in body %}{{ hblock | indent(body_indent, first=True) }}
            {% endfor %}# -- end vertical block --
            """
        )
    )

    def visit_HorizontalBlock(
        self,
        node: npir.HorizontalBlock,
        *,
        block_extents: Dict[int, HorizontalExtent] = None,
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:
        ij_extent: Extent = (block_extents or {}).get(id(node), Extent.zeros())
        boundary = ij_extent.to_boundary()
        lower = (boundary[0][0], boundary[1][0])
        upper = (boundary[0][1], boundary[1][1])
        return self.generic_visit(node, lower=lower, upper=upper, **kwargs)

    HorizontalBlock = JinjaTemplate(
        textwrap.dedent(
            """\
            # --- begin horizontal block --
            i, I = _di_ - {{ lower[0] }}, _dI_ + {{ upper[0] }}
            j, J = _dj_ - {{ lower[1] }}, _dJ_ + {{ upper[1] }}
            {% for assign in body %}{{ assign }}
            {% endfor %}# --- end horizontal block --

            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.params, "_domain_", "_origin_"]
        field_extents, block_extents = ExtentCalculator().visit(node)
        return self.generic_visit(
            node,
            signature=", ".join(signature),
            data_view_class=ORIGIN_CORRECTED_VIEW_CLASS,
            var_offset_func=VARIABLE_OFFSET_FUNCTION,
            field_extents=field_extents,
            block_extents=block_extents,
            **kwargs,
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            import numpy as np
            import numbers

            def run({{ signature }}):

                # -- begin domain boundary shortcuts --
                _di_, _dj_, _dk_ = 0, 0, 0
                _dI_, _dJ_, _dK_ = _domain_
                # -- end domain padding --

                {% for decl in field_decls %}{{ decl | indent(4) }}
                {% endfor %}
                # -- begin data views --
                {{ data_view_class | indent(4) }}
                {% for name in field_params %}{{ name }}_ = ShimmedView({{ name }}, _origin_["{{ name }}"])
                {% endfor %}# -- end data views --

                {% for pass in vertical_passes %}
                {{ pass | indent(4) }}
                {% endfor %}

            {{ var_offset_func }}
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
