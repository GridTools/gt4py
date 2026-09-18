# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next.type_system import type_specifications as ts


@pytest.mark.parametrize(
    "constraint",
    [
        None,
        ts.ScalarType,
        ts.TypeSpec,
        (ts.ScalarType,),
        (ts.ScalarType, ts.FieldType),
    ],
)
def test_deferred_type_accepts_type_spec_constraints(constraint):
    assert ts.DeferredType(constraint=constraint).constraint is constraint


@pytest.mark.parametrize("constraint", [int, ts.ScalarType(kind=ts.ScalarKind.INT32)])
def test_deferred_type_rejects_non_type_spec_constraint(constraint):
    # 'constraint' is annotated as 'type[TypeSpec] | tuple[type[TypeSpec], ...] | None',
    # which is checked with 'issubclass()' and not just with "is a class at all".
    with pytest.raises(TypeError, match="constraint"):
        ts.DeferredType(constraint=constraint)


def test_deferred_type_rejects_non_type_spec_constraint_in_tuple():
    with pytest.raises(TypeError, match="constraint"):
        ts.DeferredType(constraint=(ts.ScalarType, int))
