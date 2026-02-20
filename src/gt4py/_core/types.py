# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# Extracted from definitions.py to avoid `bool` to shadow the python built-in `bool` type in that file.

import numpy as np


# TODO(havogt): change to `np.bool` once we drop numpy 1.x support
bool = np.bool_  # noqa: A001

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

float32 = np.float32
float64 = np.float64

# This is used to map between dtypes of different array namespaces, as only the name is defined in the array API.
# We don't use `<type>.__name__` for the mapping to be on the safe side.
type_to_name = {
    globals()[name]: name
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    )
}
