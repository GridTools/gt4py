# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2022, ETH Zurich
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

"""GridTools storages classes."""


from .storage import Storage, empty, from_array, ones, zeros


_numpy_array_patch = None
_numpy_asarray_patch = None


def prepare_numpy():
    """Apply NumPy module patch."""

    import numpy as np

    if "__gt_array_patch__" not in np.__dict__:
        from functools import wraps

        from gt4py import utils as gt_utils

        global _numpy_array_patch
        global _numpy_asarray_patch

        original_array_func = np.array
        original_asarray_func = np.asarray

        @wraps(np.array)
        def __array_with_subok_patch(
            object, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, **kwargs
        ):
            subok = getattr(object, "__array_subok__", subok)
            return original_array_func(
                object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, **kwargs
            )

        @wraps(np.asarray)
        def __asarray_with_subok_patch(a, dtype=None, order=None, **kwargs):
            subok = getattr(a, "__array_subok__", False)
            if subok:
                return np.asanyarray(a, dtype=dtype, order=order, **kwargs)
            else:
                return original_asarray_func(a, dtype=dtype, order=order, **kwargs)

        _numpy_array_patch = gt_utils.patch_module(np, np.array, __array_with_subok_patch)
        _numpy_asarray_patch = gt_utils.patch_module(np, np.asarray, __asarray_with_subok_patch)

        np.__dict__["__gt_array_patch__"] = True


def restore_numpy():
    """Revert NumPy module patch."""

    import numpy as np

    if "__gt_array_patch__" in np.__dict__:
        from gt4py import utils as gt_utils

        gt_utils.restore_module(_numpy_asarray_patch)
        gt_utils.restore_module(_numpy_array_patch)
        del np.__dict__["__gt_array_patch__"]
