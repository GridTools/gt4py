# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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


class NonContiguousDomain(gt4py_exceptions.GT4PyError):
    """Describes an error where a domain would become non-contiguous after an operation."""

    detail: str

    def __init__(self, detail: str):
        super().__init__(f"Operation would result in a non-contiguous domain: `{detail}`.")
        self.detail = detail
