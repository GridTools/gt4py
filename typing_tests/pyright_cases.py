# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Cases checked by pyright in 'test_pyright.py'.

pyright must accept every construct here with zero diagnostics and with no gt4py plugin (pyright
has no plugin mechanism, which is exactly the point -- see ADR 0028). Lines that are *expected* to
error carry a trailing EXPECT-ERROR marker comment, which the test scans for and asserts on.
"""

from typing import TypeVar, assert_type

from gt4py import next as gtx
from gt4py.next import common


class IDim(gtx.DimensionIndex): ...


class JDim(gtx.DimensionIndex): ...


class KDim(gtx.DimensionIndex, kind=gtx.DimensionKind.VERTICAL): ...


# 1. A dimension class is a valid annotation argument.
a: gtx.Field[gtx.Dims[IDim, KDim], gtx.float64]

# 2. Value-level use of the class object: the dimension's name is `tag`.
assert_type(IDim.tag, common.Tag)

# 3. An index is an ordinary instance of its dimension -- no metaclass `__call__` overloads and
#    no `type: ignore` are needed to say so, and mypy and pyright agree natively.
assert_type(IDim(0), IDim)
assert_type(IDim(0).value, int)
assert_type(IDim(0).dim, gtx.Dimension)


# 4. Indices carry their dimension in the type, which a shared index type could not express.
def takes_i_index(i: IDim) -> None: ...


def pass_j_index(j: JDim) -> None:
    takes_i_index(j)  # EXPECT-ERROR


# 5. TypeVars can range over dimensions -- impossible with the old mypy plugin.
D = TypeVar("D", bound=gtx.DimensionIndex)


def name_of(d: type[D]) -> common.Tag:
    return d.tag


assert_type(name_of(IDim), common.Tag)


# 6. A dimension mismatch between fields is a static error.
def takes_i_field(f: gtx.Field[gtx.Dims[IDim], gtx.float64]) -> None: ...


def pass_j_field(g: gtx.Field[gtx.Dims[JDim], gtx.float64]) -> None:
    takes_i_field(g)  # EXPECT-ERROR


# 7. `gtx.Dimension` annotates the class object, so an *index* is not one of them.
ok: gtx.Dimension = IDim
bad: gtx.Dimension = IDim(0)  # EXPECT-ERROR

# 8. The programmatic constructor returns a dimension, not an index.
assert_type(gtx.dimension("Runtime"), gtx.Dimension)
