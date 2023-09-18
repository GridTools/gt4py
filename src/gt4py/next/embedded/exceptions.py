# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py.next import common
from gt4py.next.errors import exceptions as gt4py_exceptions


class IndexOutOfBounds(gt4py_exceptions.GT4PyError):
    domain: common.Domain
    indices: common.AnyIndexSpec
    index: common.AnyIndexElement
    dim: common.Dimension

    def __init__(
        self,
        domain: common.Domain,
        indices: common.AnyIndexSpec,
        index: common.AnyIndexElement,
        dim: common.Dimension,
    ):
        super().__init__(
            f"Out of bounds: slicing {domain} with index `{indices}`, `{index}` is out of bounds in dimension `{dim}`."
        )
        self.domain = domain
        self.indices = indices
        self.index = index
        self.dim = dim
