# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import inspect
import types

import numpy as np

import gt4py.cartesian.utils as gt_utils


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
    assert all((isinstance(k, str) or gt_utils.is_iterable_of(k, str)) for k in dtypes.keys()), (
        "Invalid key in 'dtypes'."
    )
    assert all(
        (
            isinstance(k, (type, np.dtype))
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
        if isinstance(value, (type, np.dtype)):
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
