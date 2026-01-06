# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Annotated, Final, TypeAlias

from gt4py._core.definitions import Scalar
from gt4py.next import allocators, backend
from gt4py.next.common import OffsetProvider
from gt4py.next.ffront import decorator


_ONLY_FOR_TYPING: Final[str] = "only for typing"

# TODO(havogt): alternatively we could introduce Protocols
Program: TypeAlias = Annotated[decorator.Program, _ONLY_FOR_TYPING]
FieldOperator: TypeAlias = Annotated[decorator.FieldOperator, _ONLY_FOR_TYPING]
Backend: TypeAlias = Annotated[backend.Backend, _ONLY_FOR_TYPING]
FieldBufferAllocationUtil: TypeAlias = Annotated[
    allocators.FieldBufferAllocationUtil, _ONLY_FOR_TYPING
]

__all__ = [
    "Backend",
    "FieldBufferAllocationUtil",
    "FieldOperator",
    "OffsetProvider",
    "Program",
    # from _core.definitions for convenience
    "Scalar",
]
