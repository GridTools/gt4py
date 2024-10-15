# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import numbers
from typing import Tuple

import numpy as np


class Field:
    def __init__(
        self, field: np.ndarray, offsets: Tuple[int, ...], dimensions: Tuple[bool, bool, bool]
    ):
        ii = iter(range(3))
        self.idx_to_data = tuple(
            [next(ii) if has_dim else None for has_dim in dimensions]
            + list(range(sum(dimensions), len(field.shape)))
        )

        shape = [field.shape[i] if i is not None else 1 for i in self.idx_to_data]
        self.field_view = np.reshape(field.data, shape).view(np.ndarray)

        self.offsets = offsets

    @classmethod
    def empty(cls, shape, dtype, offset):
        return cls(np.empty(shape, dtype=dtype), offset, (True, True, True))

    def shim_key(self, key):
        new_args = []
        if not isinstance(key, tuple):
            key = (key,)
        for index in self.idx_to_data:
            if index is None:
                new_args.append(slice(None, None))
            else:
                idx = key[index]
                offset = self.offsets[index]
                if isinstance(idx, slice):
                    new_args.append(
                        slice(idx.start + offset, idx.stop + offset, idx.step) if offset else idx
                    )
                else:
                    new_args.append(idx + offset)
        if not isinstance(new_args[2], (numbers.Integral, slice)):
            new_args = self.broadcast_and_clip_variable_k(new_args)
        return tuple(new_args)

    def broadcast_and_clip_variable_k(self, new_args: list):
        assert isinstance(new_args[0], slice) and isinstance(new_args[1], slice)
        if np.max(new_args[2]) >= self.field_view.shape[2] or np.min(new_args[2]) < 0:
            new_args[2] = np.clip(new_args[2].copy(), 0, self.field_view.shape[2] - 1)
        new_args[:2] = np.broadcast_arrays(
            np.expand_dims(
                np.arange(new_args[0].start, new_args[0].stop),
                axis=tuple(i for i in range(self.field_view.ndim) if i != 0),
            ),
            np.expand_dims(
                np.arange(new_args[1].start, new_args[1].stop),
                axis=tuple(i for i in range(self.field_view.ndim) if i != 1),
            ),
        )
        return new_args

    def __getitem__(self, key):
        return self.field_view.__getitem__(self.shim_key(key))

    def __setitem__(self, key, value):
        return self.field_view.__setitem__(self.shim_key(key), value)
