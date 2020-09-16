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

import gt4py.utils as gt_utils
import collections
import numpy as np
import types
import inspect


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
    dtypes as 1-tuples)"""
    assert isinstance(dtypes, collections.abc.Mapping)
    assert all(
        (isinstance(k, str) or gt_utils.is_iterable_of(k, str)) for k in dtypes.keys()
    ), "Invalid key in 'dtypes'."
    assert all(
        (isinstance(k, type) or gt_utils.is_iterable_of(k, type)) for k in dtypes.values()
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
        result[key] = [np.dtype(dt) for dt in result[key]]

    keys = [k for t in result.keys() for k in t]
    if not len(keys) == len(set(keys)):
        raise ValueError("Any field can be in only one group.")
    return result


import gt4py
from gt4py.backend.dace.base_backend import DaceOptimizer


class ApplyOTFOptimizer(DaceOptimizer):
    def transform_optimize(self, sdfg):
        from gt4py.backend.dace.sdfg.transforms import OnTheFlyMapFusion

        sdfg.apply_transformations_repeated(OnTheFlyMapFusion, validate=False)
        return sdfg


def build_dace_adhoc(
    definition,
    domain,
    halo,
    specialize_strides,
    dtype,
    passes,
    alignment,
    layout,
    loop_order,
    device,
    **params,
) -> gt4py.stencil_object.StencilObject:
    backend_name = f"dace_adhoc_{device}_{dtype}_{loop_order}_{alignment}_"
    backend_name += "_".join(str(int(h)) for h in halo) + "_"
    backend_name += "_".join(str(int(d)) for d in domain) + "_"
    backend_name += "_".join(str(int(s)) for s in specialize_strides) + "_"
    backend_name += "_".join(type(p).__name__ for p in passes) + "_"
    backend_name += "_".join(f"{k}_{v}" for k, v in params)

    from gt4py.backend.dace.cpu_backend import CPUDaceBackend
    from gt4py.backend.dace.gpu_backend import GPUDaceBackend
    from gt4py.backend.dace.base_backend import CudaDaceOptimizer, DaceOptimizer
    from gt4py.backend.concepts import register as register_backend

    base_backend = CPUDaceBackend if device == "cpu" else GPUDaceBackend
    base_optimizer = DaceOptimizer if device == "cpu" else CudaDaceOptimizer

    class CompositeOptimizer(base_optimizer):
        def __init__(self, passes):
            self._passes = passes

        def transform_library(self, sdfg):
            for xform in self._passes:
                sdfg = xform.transform_library(sdfg)
            return sdfg

        def transform_optimize(self, sdfg):
            for xform in self._passes:
                sdfg = xform.transform_optimize(sdfg)
            return sdfg

    @register_backend
    class AdHocBackend(base_backend):
        name = backend_name
        storage_info = {
            "alignment": alignment,  # will not affect temporaries currently
            "device": device,  # change me
            "layout_map": lambda m: layout,
            "is_compatible_layout": lambda m: True,
            "is_compatible_type": lambda m: True,
        }
        DEFAULT_OPTIMIZER = CompositeOptimizer(passes)

    return gt4py.gtscript.stencil(
        definition=definition, backend=backend_name, dtypes={"dtype": dtype}
    )
