# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from dace import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace.library_nodes.broadcast import Broadcast
from gt4py.next.program_processors.runners.dace.library_nodes.reduce_with_skip_values import (
    ReduceWithSkipValues,
)


GTIR_LIBRARY_NODES: Final[tuple[dace_nodes.LibraryNode, ...]] = (Broadcast, ReduceWithSkipValues)
"""List of available GTIR library nodes."""


__all__ = [
    "Broadcast",
    "ReduceWithSkipValues",
]
