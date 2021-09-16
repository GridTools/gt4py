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

import textwrap
from typing import Any, Collection, Dict, Optional, Tuple, Union

from eve.codegen import FormatTemplate, JinjaTemplate, TemplatedGenerator
from gt4py.definitions import Extent
from gtc import common
from gtc.passes import utils
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
                if self.offsets[dim] == 0:
                    new_args.append(idx)
                elif isinstance(idx, slice):
                    start = 0 if idx.start is None else idx.start
                    stop = -1 if idx.stop is None else idx.stop
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
    """
)

DomainBounds = Tuple[Tuple[str, int], Tuple[str, int]]
DomainSpec = Tuple[Optional[DomainBounds], Optional[DomainBounds], Optional[DomainBounds]]


def get_horizontal_restriction(
    domain_mask: npir.HorizontalMask,
    *,
    h_lower: Tuple[int, int],
    h_upper: Tuple[int, int],
    is_serial: bool,
    **kwargs: Any,
) -> Tuple[DomainBounds, DomainBounds]:
    def base_and_offset(bound: common.AxisBound, axis: str) -> str:
        return (
            axis.lower() if bound.level == common.LevelMarker.START else axis.upper(),
            bound.offset,
        )

    horizontal_extent = Extent(((-h_lower[0], h_upper[0]), (-h_lower[1], h_upper[1]), (0, 0)))
    rel_mask: Optional[common.HorizontalMask] = utils.compute_relative_mask(
        horizontal_extent, domain_mask
    )
    assert rel_mask is not None
    return (
        (base_and_offset(interval.start, axis), base_and_offset(interval.end, axis))
        for axis, interval in (("I", rel_mask.i), ("J", rel_mask.j))
    )


def compute_axis_bounds(bounds: Optional[DomainBounds], axis_name: str, offset: int) -> str:
    def lpar(offset: int) -> str:
        return "(" if offset != 0 else ""

    def rpar(offset: int) -> str:
        return ")" if offset != 0 else ""

    def offset_str(offset: int) -> str:
        if offset < 0:
            return str(offset)
        elif offset > 0:
            return f"+ {offset}"
        else:
            return ""

    if not bounds:
        bounds = ((axis_name.lower(), 0), (axis_name.upper(), 0))
        # NOTE(jdahm): This no longer uses a visitor for the NumericalOffsets..
    loffset = bounds[0][1] + offset
    lower = lpar(loffset) + f"{bounds[0][0]}{offset_str(loffset)}" + rpar(loffset)
    uoffset = bounds[1][1] + offset
    upper = lpar(uoffset) + f"{bounds[1][0]}{offset_str(uoffset)}" + rpar(uoffset)
    return lower, upper


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

    def visit_AxisOffset(
        self,
        node: npir.AxisOffset,
        *,
        bounds: Optional[DomainBounds] = None,
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:
        offset = self.visit(node.offset)
        axis_name = self.visit(node.axis_name)
        if node.parallel:
            lower, upper = compute_axis_bounds(bounds, axis_name, node.offset.value)
            return f"{lower}:{upper}"
        else:
            return f"{axis_name.lower()}_{offset}:({axis_name.lower()}_{offset} + 1)"

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

    def visit_FieldSlice(
        self,
        node: npir.FieldSlice,
        mask_acc="",
        *,
        is_serial=False,
        domain: Optional[DomainSpec] = None,
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:

        offset = [node.i_offset, node.j_offset, node.k_offset]
        if not domain:
            domain = [None] * 3

        offset_str = ", ".join(
            self.visit(off, bounds=bounds, **kwargs) if off else ":"
            for off, bounds in zip(offset, domain)
        )

        if mask_acc and any(off is None for off in offset):
            axes_bounds = (
                compute_axis_bounds(bounds, axis_name, 0)
                for bounds, axis_name in zip(domain, ("I", "J"))
            )
            k_size = "1" if is_serial else "K - k"
            broadcast_str = (
                ",".join([f"{upper}-{lower}" for lower, upper in axes_bounds]) + f", {k_size}"
            )
            arr_expr = f"np.broadcast_to({node.name}_[{offset_str}], ({broadcast_str}))"
        else:
            arr_expr = f"{node.name}_[{offset_str}]"

        return self.generic_visit(node, arr_expr=arr_expr, mask_acc=mask_acc, **kwargs)

    FieldSlice = FormatTemplate("{arr_expr}{mask_acc}")

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

    def visit_MaskBlock(self, node: npir.MaskBlock, **kwargs: Any) -> Union[str, Collection[str]]:
        if isinstance(node.mask, npir.FieldSlice):
            mask_def = ""
        elif isinstance(node.mask, npir.BroadCast):
            mask_name = node.mask_name
            mask = self.visit(node.mask, **kwargs)
            mask_def = f"{mask_name}_ = np.full((I - i, J - j, K - k), {mask})\n"
        else:
            mask_name = node.mask_name
            mask = self.visit(node.mask, **kwargs)
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
        domain = None
        if node.horiz_mask:
            domain = list(get_horizontal_restriction(node.horiz_mask, **kwargs)) + [None]
        if node.mask:
            mask_acc = f"[{self.visit(node.mask, domain=domain, **kwargs)}]"
        if isinstance(node.right, npir.EmptyTemp):
            kwargs["temp_name"] = node.left.name
        return self.generic_visit(node, mask_acc=mask_acc, domain=domain, **kwargs)

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
        self, node: npir.HorizontalBlock, **kwargs
    ) -> Union[str, Collection[str]]:
        lower, upper = [0, 0], [0, 0]

        if extents := kwargs.get("field_extents"):
            fields = set(node.iter_tree().if_isinstance(npir.FieldSlice).getattr("name"))
            for field in fields:
                # The extent of masks has not yet been collected but is always zero.
                extents.setdefault(field, Extent.zeros())
            lower[0] = min(extents[field].to_boundary()[0][0] for field in fields)
            lower[1] = min(extents[field].to_boundary()[1][0] for field in fields)
            upper[0] = min(extents[field].to_boundary()[0][1] for field in fields)
            upper[1] = min(extents[field].to_boundary()[1][1] for field in fields)
        return self.generic_visit(node, h_lower=lower, h_upper=upper, **kwargs)

    HorizontalBlock = JinjaTemplate(
        textwrap.dedent(
            """\
            # --- begin horizontal block --
            i, I = _di_ - {{ h_lower[0] }}, _dI_ + {{ h_upper[0] }}
            j, J = _dj_ - {{ h_lower[1] }}, _dJ_ + {{ h_upper[1] }}
            {% for assign in body %}{{ assign }}
            {% endfor %}# --- end horizontal block --

            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, *, field_extents: Dict[str, Extent], **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.params, "_domain_", "_origin_"]
        kwargs["field_extents"] = field_extents
        return self.generic_visit(
            node,
            signature=", ".join(signature),
            data_view_class=ORIGIN_CORRECTED_VIEW_CLASS,
            **kwargs,
        )

    Computation = JinjaTemplate(
        textwrap.dedent(
            """\
            import numpy as np


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
