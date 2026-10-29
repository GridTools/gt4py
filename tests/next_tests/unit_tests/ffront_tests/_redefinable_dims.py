# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Dimensions that `test_dimension_identity_at_call` redefines with `importlib.reload`."""

import gt4py.next as gtx


class IDim(gtx.CartesianAxisIndex): ...


class KDim(gtx.CartesianAxisIndex, kind=gtx.DimensionKind.VERTICAL): ...


class Vertex(gtx.DimensionIndex): ...


class Edge(gtx.DimensionIndex): ...


class V2EDim(gtx.DimensionIndex, kind=gtx.DimensionKind.LOCAL): ...
