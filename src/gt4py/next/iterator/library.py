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

from gt4py.next.iterator.builtins import reduce


def sum_(fun=None):
    if fun is None:
        return reduce(lambda a, b: a + b, 0.0)
    else:
        return reduce(lambda first, a, b: first + fun(a, b), 0.0)  # TODO tracing for *args


def dot(a, b):
    return reduce(lambda acc, a_n, c_n: acc + a_n * c_n, 0.0)(a, b)
