# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py.cartesian.gtc.dace.passes import utils

# Because "dace tests" filter by `requires_dace`, we still need to add the marker.
# This global variable adds the marker to all test functions in this module.
pytestmark = pytest.mark.requires_dace


def test_list_index_raises_if_not_found():
    collection = [tn.ScheduleTreeRoot(children=[], name="tester")]
    node = tn.ScheduleTreeRoot(children=[], name="tester2")

    with pytest.raises(StopIteration):
        utils.list_index(collection, node)


def test_list_index_happy_case():
    needle = tn.ScheduleTreeRoot(children=[], name="tester")
    haystack = [needle]

    assert utils.list_index(haystack, needle) == 0


def test_list_index_two_copies():
    root = tn.ScheduleTreeRoot(children=[], name="root node")
    copy = tn.ScheduleTreeRoot(children=[], name="root node")
    collection = [root, copy]

    assert utils.list_index(collection, root) == 0
    assert utils.list_index(collection, copy) == 1
