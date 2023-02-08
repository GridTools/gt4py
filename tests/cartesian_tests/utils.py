# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

try:
    import dace
except ImportError:
    dace = None


class ArrayWrapper:
    def __init__(self, array, **kwargs):
        self.array = array

    @property
    def __array_interface__(self):
        return self.array.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__

    def __descriptor__(self):
        return dace.data.create_datadescriptor(self.array)


class DimensionsWrapper(ArrayWrapper):
    def __init__(self, dimensions, **kwargs):
        self.__gt_dims__ = dimensions
        super().__init__(**kwargs)


class OriginWrapper(ArrayWrapper):
    def __init__(self, *, origin, **kwargs):
        self.__gt_origin__ = origin
        super().__init__(**kwargs)

    def __descriptor__(self):
        res = super().__descriptor__()
        res.__gt_origin__ = self.__gt_origin__
        return res
