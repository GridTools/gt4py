# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from dace.sdfg.analysis.schedule_tree import treenodes as tn


def list_index(collection: list[tn.ScheduleTreeNode], node: tn.ScheduleTreeNode) -> int:
    """
    Get the index of `node` in `collection` with `is` operator. Raises `StopIteration` if not found.

    Comparing with the `is` operator ensures memory comparison. The function `list.index()`
    uses value comparison and might thus yield different results.
    """
    # compare with "is" to get memory comparison. ".index()" uses value comparison
    return next(index for index, element in enumerate(collection) if element is node)
