# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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
