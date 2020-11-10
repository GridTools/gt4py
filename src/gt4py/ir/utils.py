# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import ast
import inspect
import numbers
import textwrap
import types
from typing import Any, Dict, List, Optional, Tuple, Union

import gt4py.gtscript as gtscript
from gt4py.utils import NOTHING

from .nodes import *


# --- Definition IR ---
DEFAULT_LAYOUT_ID = "_default_layout_id_"


class AxisIntervalParser(ast.NodeVisitor):
    """Replaces usage of gt4py.ir.utils.make_axis_interval.

    This should be moved to gt4py.frontend.gtscript_frontend when AST2IRVisitor
    is removed.
    """

    @classmethod
    def apply(
        cls,
        node: Union[ast.Slice, ast.Ellipsis, ast.Subscript],
        axis_name: str,
        context: Optional[dict] = None,
        loc: Optional[Location] = None,
    ) -> AxisInterval:
        parser = cls(axis_name, context, loc)

        if isinstance(node, ast.Ellipsis):
            interval = AxisInterval.full_interval()
            interval.loc = loc
            return interval

        if isinstance(node, ast.Subscript):
            slice_node = node.slice
        else:
            if not isinstance(node, ast.Slice):
                raise TypeError("Requires Slice node")
            slice_node = node

        start = parser.visit(slice_node.lower)
        end = parser.visit(slice_node.upper)
        return AxisInterval(start=start, end=end, loc=loc)

    def __init__(
        self,
        axis_name: str,
        context: Optional[dict] = None,
        loc: Optional[Location] = None,
    ):
        self.axis_name = axis_name
        self.context = context or dict()
        self.loc = loc

        # initialize possible exceptions
        self.interval_error = ValueError(
            f"Invalid 'interval' specification at line {loc.line} (column {loc.column})"
        )

    @staticmethod
    def make_axis_bound(offset: int, loc: Location = None) -> AxisBound:
        return AxisBound(
            level=LevelMarker.START if offset >= 0 else LevelMarker.END,
            offset=offset,
            loc=loc,
        )

    def visit_Name(self, node: ast.Name) -> AxisBound:
        symbol = node.id
        if symbol in self.context:
            value = self.context[symbol]
            if isinstance(value, gtscript._AxisSplitter):
                if value.axis != self.axis_name:
                    raise self.interval_error
                offset = value.offset
            elif isinstance(value, int):
                offset = value
            else:
                raise self.interval_error
            return self.make_axis_bound(offset, self.loc)
        else:
            return AxisBound(level=VarRef(name=symbol), loc=self.loc)

    def visit_Constant(self, node: ast.Constant) -> AxisBound:
        if node.value is not None:
            if isinstance(node.value, gtscript._AxisSplitter):
                if node.value.axis != self.axis_name:
                    raise self.interval_error
                offset = node.value.offset
            elif isinstance(node.value, int):
                offset = node.value
            else:
                raise self.interval_error
            return self.make_axis_bound(offset, self.loc)
        else:
            return AxisBound(level=LevelMarker.END, offset=0, loc=self.loc)

    def visit_NameConstant(self, node: ast.NameConstant) -> AxisBound:
        """Python < 3.8 uses ast.NameConstant for 'None'."""
        if node.value is not None:
            raise self.interval_error
        else:
            return AxisBound(level=LevelMarker.END, offset=0, loc=self.loc)

    def visit_Num(self, node: ast.Num) -> AxisBound:
        """Equivalent to visit_Constant. Required for Python < 3.8."""
        return self.make_axis_bound(node.n, self.loc)

    def visit_BinOp(self, node: ast.BinOp) -> AxisBound:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            op = lambda x, y: x + y
        elif isinstance(node.op, ast.Sub):
            op = lambda x, y: x - y
        elif isinstance(node.op, ast.Mult):
            if left.level != right.level or not isinstance(left.level, LevelMarker):
                raise self.interval_error
            op = lambda x, y: x * y
        else:
            raise self.interval_error

        if right.level == LevelMarker.END:
            level = LevelMarker.END
        else:
            level = left.level

        return AxisBound(level=level, offset=op(left.offset, right.offset), loc=self.loc)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> AxisBound:
        axis_bound = self.visit(node.operand)
        level = axis_bound.level
        offset = axis_bound.offset
        if isinstance(node.op, ast.USub):
            new_level = LevelMarker.END if level == LevelMarker.START else LevelMarker.START
            return AxisBound(level=new_level, offset=-offset, loc=self.loc)
        else:
            raise self.interval_error

        return AxisBound(level=LevelMarker.END, offset=0, loc=self.loc)

    def visit_Subscript(self, node: ast.Subscript) -> AxisBound:
        if node.value.id != self.axis_name:
            raise self.interval_error

        if not isinstance(node.slice, ast.Index):
            raise self.interval_error

        return self.visit(node.slice.value)


parse_interval_node = AxisIntervalParser.apply


def make_expr(value):
    if isinstance(value, Expr):
        result = value
    elif isinstance(value, numbers.Number):
        data_type = DataType.from_dtype(np.dtype(type(value)))
        result = ScalarLiteral(value=value, data_type=data_type)
    else:
        raise ValueError("Invalid expression value '{}'".format(value))
    return result


def make_field_decl(
    name: str,
    dtype=np.float_,
    masked_axes=None,
    is_api=True,
    layout_id=DEFAULT_LAYOUT_ID,
    loc=None,
    *,
    axes_dict=None,
):
    axes_dict = axes_dict or {ax.name: ax for ax in Domain.LatLonGrid().axes}
    masked_axes = masked_axes or []
    return FieldDecl(
        name=name,
        data_type=DataType.from_dtype(dtype),
        axes=[name for name, axis in axes_dict.items() if name not in masked_axes],
        is_api=is_api,
        layout_id=layout_id,
        loc=loc,
    )


def make_field_ref(name: str, offset=(0, 0, 0), *, axes_names=None):
    axes_names = axes_names or [ax.name for ax in Domain.LatLonGrid().axes]
    offset = {axes_names[i]: value for i, value in enumerate(offset) if value is not None}
    return FieldRef(name=name, offset=offset)


def make_api_signature(args_list: list):
    api_signature = []
    for item in args_list:
        if isinstance(item, str):
            api_signature.append(ArgumentInfo(name=item, is_keyword=False, default=None))
        elif isinstance(item, tuple):
            api_signature.append(
                ArgumentInfo(
                    name=item[0],
                    is_keyword=item[1] if len(item) > 1 else False,
                    default=item[2] if len(item) > 2 else None,
                )
            )
        else:
            assert isinstance(item, ArgumentInfo), "Invalid api_signature"
    return api_signature


# --- Implementation IR ---
def make_field_accessor(name: str, intent=False, extent=((0, 0), (0, 0), (0, 0))):
    if not isinstance(intent, AccessIntent):
        assert isinstance(intent, bool)
        intent = AccessIntent.READ_WRITE if intent else AccessIntent.READ_ONLY
    return FieldAccessor(symbol=name, intent=intent, extent=Extent(extent))
