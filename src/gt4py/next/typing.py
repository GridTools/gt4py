# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, TypeAlias

from gt4py._core.definitions import Scalar
from gt4py.next import backend
from gt4py.next.common import OffsetProvider
from gt4py.next.ffront import decorator


# TODO(havogt): alternatively we could introduce Protocols
Program: TypeAlias = Annotated[decorator.Program, ""]
FieldOperator: TypeAlias = Annotated[decorator.FieldOperator, ""]
Backend: TypeAlias = Annotated[backend.Backend, ""]


__all__ = [
    "Backend",
    "FieldOperator",
    "OffsetProvider",
    "Program",
    # from _core.definitions for convenience
    "Scalar",
]
