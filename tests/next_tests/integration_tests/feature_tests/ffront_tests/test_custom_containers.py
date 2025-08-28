# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import NamedTuple
import pytest
from gt4py import next as gtx

from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


class NamedTupleContainer(NamedTuple):
    x: gtx.Field[[IDim, JDim], gtx.float32]
    y: gtx.Field[[IDim, JDim], gtx.float32]


@dataclasses.dataclass(frozen=True)
class FrozenDataclassContainer:
    x: gtx.Field[[IDim, JDim], gtx.float32]
    y: gtx.Field[[IDim, JDim], gtx.float32]


@dataclasses.dataclass
class MutableDataclassContainer:
    # We cannot optimize the container -> tuple operation by pre-computing.
    x: gtx.Field[[IDim, JDim], gtx.float32]
    y: gtx.Field[[IDim, JDim], gtx.float32]


@dataclasses.dataclass(frozen=True, slots=True)
class SlotDataclassContainer:
    # We cannot attach metadata trivially (e.g. pre-computed tuple).
    x: gtx.Field[[IDim, JDim], gtx.float32]
    y: gtx.Field[[IDim, JDim], gtx.float32]


@dataclasses.dataclass(frozen=False)
class NestedDataclassContainer:
    x: FrozenDataclassContainer
    y: FrozenDataclassContainer


@dataclasses.dataclass(frozen=True)
class FrozenNestedMutableDataclassContainer:
    x: MutableDataclassContainer
    y: MutableDataclassContainer


@dataclasses.dataclass(frozen=True)
class ImmutableNestedDataclassNamedTupleContainer:
    x: NamedTupleContainer
    y: NamedTupleContainer


class ImmutableNestedNamedTupleDataclassContainer(NamedTuple):
    x: FrozenDataclassContainer
    y: FrozenDataclassContainer


class MutableNestedNamedTupleDataclassContainer(NamedTuple):
    x: MutableDataclassContainer
    y: MutableDataclassContainer


# TODO lowering test
@pytest.mark.parametrize("container", [FrozenDataclassContainer])
def test_container_via_identity(cartesian_case, container: type):
    """
    Simple test that shows all containers can be passed and returned.
    """

    @gtx.field_operator
    def id_op(foo: container) -> container:
        return foo

    # cases

    print(id_op.__gt_itir__())
