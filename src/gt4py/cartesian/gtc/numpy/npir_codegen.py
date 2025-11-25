# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
import textwrap
from dataclasses import dataclass, field
from typing import Any, Collection, List, Optional, Set, Tuple, Union, cast

from gt4py import eve
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.numpy import npir
from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, JinjaTemplate as as_jinja


__all__ = ["NpirCodegen"]


def _offset_to_str(offset: int) -> str:
    if offset > 0:
        return f" + {offset}"
    elif offset < 0:
        return f" - {-offset}"
    else:
        return ""


def _slice_string(ch: str, offset: int, interval: Tuple[common.AxisBound, common.AxisBound]) -> str:
    start_ch = ch if interval[0].level == common.LevelMarker.START else ch.upper()
    end_ch = ch if interval[1].level == common.LevelMarker.START else ch.upper()

    return (
        f"{start_ch}{_offset_to_str(interval[0].offset + offset)}"
        ":"
        f"{end_ch}{_offset_to_str(interval[1].offset + offset)}"
    )


def _make_slice_access(
    offset: Tuple[Optional[int], Optional[int], Union[str, Optional[int]]],
    is_serial: bool,
    interval: Optional[npir.HorizontalMask] = None,
) -> List[str]:
    axes: List[str] = []

    if interval is None:
        interval = npir.HorizontalMask(
            i=(common.AxisBound.start(0), common.AxisBound.end(0)),
            j=(common.AxisBound.start(0), common.AxisBound.end(0)),
        )

    if offset[0] is not None:
        axes.append(_slice_string("i", offset[0], interval.i))
    if offset[1] is not None:
        axes.append(_slice_string("j", offset[1], interval.j))

    if isinstance(offset[2], numbers.Number):
        bounds = (
            (common.AxisBound.start(), common.AxisBound.start(offset=1))
            if is_serial
            else (common.AxisBound.start(), common.AxisBound.end())
        )
        k_str = "k_" if is_serial else "k"
        axes.append(_slice_string(k_str, offset[2], bounds))
    elif isinstance(offset[2], str):
        axes.append(offset[2])

    return axes


class NpirCodegen(codegen.TemplatedGenerator, eve.VisitorWithSymbolTableTrait):
    @dataclass
    class BlockContext:
        locals_declared: Set[str] = field(default_factory=set)

        def add_declared(self, *args):
            self.locals_declared |= set(args)

    FieldDecl = as_fmt("{name} = Field({name}, _origin_['{name}'], ({', '.join(dimensions)}))")

    def visit_TemporaryDecl(
        self, node: npir.TemporaryDecl, **kwargs
    ) -> Union[str, Collection[str]]:
        # Cartesian IJ
        shape = [f"_dI_ + {node.padding[0]}", f"_dJ_ + {node.padding[1]}"]
        offset = [str(off) for off in node.offset]
        # Vertical dimension K
        if node.dimensions[2]:
            shape += ["_dK_"]
            offset += ["0"]
        # Data dimensions
        shape += [str(dim) for dim in node.data_dims]
        offset += ["0"] * len(node.data_dims)

        dtype = self.visit(node.dtype, **kwargs)
        dims = node.dimensions
        return f"{node.name} = Field.empty(({', '.join(shape)}), {dtype}, ({', '.join(offset)}), {dims})"

    LocalScalarDecl = as_fmt(
        "{name} = Field.empty((_dI_ + {upper[0] + lower[0]}, _dJ_ + {upper[1] + lower[1]}, {ksize}), {dtype}, ({', '.join(str(l) for l in lower)}, 0))"
    )

    VarKOffset = as_fmt("lk + {k}")

    def visit_FieldSlice(self, node: npir.FieldSlice, **kwargs: Any) -> Union[str, Collection[str]]:
        k_offset = (
            self.visit(node.k_offset, **kwargs)
            if isinstance(node.k_offset, npir.VarKOffset)
            else node.k_offset
        )

        offsets: Tuple[Optional[int], Optional[int], Union[str, int, None]] = (
            node.i_offset,
            node.j_offset,
            k_offset,
        )

        # To determine: when is the symbol name not in the symtable?
        if node.name in kwargs.get("symtable", {}):
            decl = kwargs["symtable"][node.name]
            dimensions = (
                decl.dimensions
                if isinstance(decl, npir.FieldDecl | npir.TemporaryDecl)
                else [True] * 3
            )
            offsets = cast(
                Tuple[Optional[int], Optional[int], Union[str, int, None]],
                tuple(off if has_dim else None for has_dim, off in zip(dimensions, offsets)),
            )

        args = _make_slice_access(offsets, kwargs["is_serial"], kwargs.get("horizontal_mask"))
        data_index = self.visit(node.data_index, inside_slice=True, **kwargs)

        access_slice = ", ".join(args + list(data_index))

        return f"{node.name}[{access_slice}]"

    def visit_LocalScalarAccess(
        self,
        node: npir.LocalScalarAccess,
        *,
        is_serial: bool,
        horizontal_mask: Optional[npir.HorizontalMask] = None,
        **kwargs: Any,
    ) -> Union[str, Collection[str]]:
        args = _make_slice_access((0, 0, 0), is_serial, horizontal_mask)
        if is_serial:
            args[2] = ":"
        return f"{node.name}[{', '.join(args)}]"

    ParamAccess = as_fmt("{name}")

    def visit_DataType(self, node: common.DataType, **kwargs: Any) -> Union[str, Collection[str]]:
        # `np.bool` is a deprecated alias for the builtin `bool` or `np.bool_`.
        if node not in {common.DataType.BOOL}:
            return f"np.{node.name.lower()}"
        else:
            return node.name.lower()

    def visit_BuiltInLiteral(
        self, node: common.BuiltInLiteral, **kwargs
    ) -> Union[str, Collection[str]]:
        if node is common.BuiltInLiteral.TRUE:
            return "True"
        elif node is common.BuiltInLiteral.FALSE:
            return "False"
        else:
            return self.generic_visit(node, **kwargs)

    def visit_ScalarLiteral(
        self, node: npir.ScalarLiteral, *, inside_slice: bool = False, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        # This could be trivial, but it's convenient for reading if the dtype is omitted in slices.
        dtype = self.visit(node.dtype, inside_slice=inside_slice, **kwargs)
        value = self.visit(node.value, inside_slice=inside_slice, **kwargs)
        return f"{value}" if inside_slice else f"{dtype}({value})"

    ScalarCast = as_fmt("{dtype}({expr})")

    VectorCast = as_fmt("{expr}.astype({dtype})")

    def visit_NativeFunction(
        self, node: common.NativeFunction, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return f"ufuncs.{common.OP_TO_UFUNC_NAME[common.NativeFunction][node]}"

    def visit_NativeFuncCall(
        self, node: npir.NativeFuncCall, *, mask: Optional[str] = None, **kwargs: Any
    ):
        kwargs["mask_arg"] = f", where={mask}" if mask else ""
        return self.generic_visit(node, mask=mask, **kwargs)

    NativeFuncCall = as_fmt("{func}({', '.join(arg for arg in args)}{mask_arg})")

    def visit_VectorAssign(
        self, node: npir.VectorAssign, *, ctx: BlockContext, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        left = self.visit(node.left, horizontal_mask=node.horizontal_mask, **kwargs)
        right = self.visit(node.right, horizontal_mask=node.horizontal_mask, **kwargs)
        return f"{left} = {right}"

    VectorArithmetic = as_fmt("({left} {op} {right})")

    VectorLogic = as_fmt("np.bitwise_{op}({left}, {right})")

    def visit_UnaryOperator(
        self, node: common.UnaryOperator, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        if node is common.UnaryOperator.NOT:
            return "np.bitwise_not"
        return self.generic_visit(node, **kwargs)

    VectorUnaryOp = as_fmt("({op}({expr}))")

    VectorTernaryOp = as_fmt("np.where({cond}, {true_expr}, {false_expr})")

    def visit_LevelMarker(
        self, node: common.LevelMarker, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        return "K" if node == common.LevelMarker.END else "k"

    def visit_AxisBound(self, node: common.AxisBound, **kwargs: Any) -> Union[str, Collection[str]]:
        if node.offset > 0:
            voffset = f" + {node.offset}"
        elif node.offset == 0:
            voffset = ""
        else:
            voffset = f" - {-node.offset}"
        return self.generic_visit(node, voffset=voffset, **kwargs)

    AxisBound = as_fmt("_d{level}_{voffset}")

    def visit_LoopOrder(self, node: common.LoopOrder, **kwargs) -> Union[str, Collection[str]]:
        if node is common.LoopOrder.FORWARD:
            return "for k_ in range(k, K):"
        elif node is common.LoopOrder.BACKWARD:
            return "for k_ in range(K-1, k-1, -1):"
        return ""

    Broadcast = as_fmt("{expr}")

    While = as_jinja(
        textwrap.dedent(
            """\
            while np.any({{ cond }}):
                {% for stmt in body %}{{ stmt }}
                {% endfor %}
            """
        )
    )

    def visit_While(self, node: npir.While, **kwargs: Any) -> str:
        cond = self.visit(node.cond, **kwargs)
        body = []
        for stmt in self.visit(node.body, **kwargs):
            body.extend(stmt.split("\n"))
        return self.While.render(cond=cond, body=body)

    def visit_VerticalPass(self, node: npir.VerticalPass, **kwargs):
        is_serial = node.direction != common.LoopOrder.PARALLEL
        has_variable_k = bool(node.walk_values().if_isinstance(npir.VarKOffset).to_list())
        return self.generic_visit(
            node,
            is_serial=is_serial,
            has_variable_k=has_variable_k,
            ksize="_dK_" if not is_serial else "1",
            lk_stmt="lk = " + ("k_" if is_serial else "np.arange(k, K)[np.newaxis, np.newaxis, :]"),
            **kwargs,
        )

    VerticalPass = as_jinja(
        textwrap.dedent(
            """\
            # --- begin vertical block ---{% set body_indent = 0 %}
            k, K = {{ lower }}, {{ upper }}
            {%- if direction %}
            {{ direction }}{% set body_indent = 4 %}{% endif %}
            {% if has_variable_k %}
            {{ lk_stmt | indent(body_indent, first=True) }}
            {% endif -%}
            {% for hblock in body %}
            {{ hblock | indent(body_indent, first=True) }}
            {% endfor %}# --- end vertical block ---
            """
        )
    )

    def visit_HorizontalBlock(
        self, node: npir.HorizontalBlock, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        lower = (-node.extent[0][0], -node.extent[1][0])
        upper = (node.extent[0][1], node.extent[1][1])
        return self.generic_visit(node, lower=lower, upper=upper, ctx=self.BlockContext(), **kwargs)

    HorizontalBlock = as_jinja(
        textwrap.dedent(
            """\
            # --- begin horizontal block --
            i, I = _di_ - {{ lower[0] }}, _dI_ + {{ upper[0] }}
            j, J = _dj_ - {{ lower[1] }}, _dJ_ + {{ upper[1] }}

            {% for stmt in body %}{{ stmt }}
            {% endfor -%}
            # --- end horizontal block --

            """
        )
    )

    def visit_Computation(
        self, node: npir.Computation, *, ignore_np_errstate: bool = True, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        signature = ["*", *node.arguments, "_domain_", "_origin_"]
        return self.generic_visit(
            node,
            signature=", ".join(signature),
            ignore_np_errstate=ignore_np_errstate,
            **kwargs,
        )

    Computation = as_jinja(
        textwrap.dedent(
            """\
            import numbers
            from typing import Tuple

            import numpy as np
            from gt4py.cartesian.gtc import ufuncs
            from gt4py.cartesian.utils import Field

            def run({{ signature }}):

                # --- begin domain boundary shortcuts ---
                _di_, _dj_, _dk_ = 0, 0, 0
                _dI_, _dJ_, _dK_ = _domain_
                # --- end domain padding ---

                {% for decl in api_field_decls %}{{ decl | indent(4) }}
                {% endfor %}
                {% for decl in temp_decls %}{{ decl | indent(4) }}
                {% endfor %}

                {% if ignore_np_errstate -%}
                with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
                {%- else -%}
                with np.errstate():
                {%- endif %}

                {% for pass in vertical_passes %}
                {{ pass | indent(8) }}
                {% else %}
                    pass
                {% endfor %}
            """
        )
    )
