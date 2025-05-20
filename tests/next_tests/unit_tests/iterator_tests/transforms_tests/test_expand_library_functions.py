# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import textwrap

from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.expand_library_functions import ExpandLibraryFunctions

from next_tests.integration_tests.cases import IDim, JDim, KDim


def test_trivial():
    pos = im.make_tuple(0, 1)
    bounds = {
        IDim: (3, 4),
        JDim: (5, 6),
    }
    testee = im.call("in_")(pos, im.domain(common.GridType.CARTESIAN, bounds))
    expected = im.and_(
        im.and_(
            im.less_equal(bounds[IDim][0], im.tuple_get(0, pos)),
            im.less(im.tuple_get(0, pos), bounds[IDim][1]),
        ),
        im.and_(
            im.less_equal(bounds[JDim][0], im.tuple_get(1, pos)),
            im.less(im.tuple_get(1, pos), bounds[JDim][1]),
        ),
    )
    actual = ExpandLibraryFunctions.apply(testee)
    assert actual == expected
