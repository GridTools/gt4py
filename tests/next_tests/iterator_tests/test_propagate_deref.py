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

from gt4py.next.ffront import itir_makers as im
from gt4py.next.iterator.transforms.propagate_deref import PropagateDeref


def test_deref_propagation():
    testee = im.deref_(
        im.call_(im.lambda__("inner_it")(im.lift_("stencil")("inner_it")))("outer_it")
    )
    expected = im.call_(im.lambda__("inner_it")(im.deref_(im.lift_("stencil")("inner_it"))))(
        "outer_it"
    )

    actual = PropagateDeref.apply(testee)
    assert actual == expected
