# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.iterator.builtins import reduce


def sum_(fun=None):
    if fun is None:
        return reduce(lambda a, b: a + b, 0.0)
    else:
        return reduce(lambda first, a, b: first + fun(a, b), 0.0)  # TODO tracing for *args


def dot(a, b):
    return reduce(lambda acc, a_n, b_n: acc + a_n * b_n, 0.0)(a, b)
