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

import collections
import inspect
import types

import numpy as np

import gt4py.utils as gt_utils


def copy_func(f, name=None):
    return types.FunctionType(
        f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__
    )


def annotate_function(function, dtypes):
    annotated_func = copy_func(function)
    for k in inspect.getfullargspec(annotated_func).args:
        annotated_func.__annotations__[k] = dtypes[k]
    for k in inspect.getfullargspec(annotated_func).kwonlyargs:
        annotated_func.__annotations__[k] = dtypes[k]
    return annotated_func


def standardize_dtype_dict(dtypes):
    """Standardizes the dtype dict as it can be specified for the stencil test suites.

    In the input dictionary, a selection of possible dtypes or just a single dtype can be specified for a set of fields
    or a single field. This function makes sure that all keys are tuples (by wrapping single field names and single
    dtypes as 1-tuples)
    """
    assert isinstance(dtypes, collections.abc.Mapping)
    assert all(
        (isinstance(k, str) or gt_utils.is_iterable_of(k, str)) for k in dtypes.keys()
    ), "Invalid key in 'dtypes'."
    assert all(
        (
            isinstance(k, type)
            or gt_utils.is_iterable_of(k, type)
            or gt_utils.is_iterable_of(k, np.dtype)
        )
        for k in dtypes.values()
    ), "Invalid dtype in 'dtypes'"

    result = {}
    for key, value in dtypes.items():
        if isinstance(key, str):
            key = (key,)
        else:
            key = (*key,)
        if isinstance(value, type):
            value = (value,)
        else:
            value = (*value,)
        result[key] = value

    for key, value in result.items():
        result[key] = [np.dtype(dt) for dt in value]

    keys = [k for t in result.keys() for k in t]
    if not len(keys) == len(set(keys)):
        raise ValueError("Any field can be in only one group.")
    return result
