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

import numpy as np

import gt4py.next as gtx
from gt4py.eve.datamodels import field
from gt4py.next.iterator import embedded


I = gtx.Dimension("I")
J = gtx.Dimension("J")


def make_located_field(dtype=np.float64):
    return gtx.np_as_located_field(I, J)(np.zeros((1, 1), dtype=dtype))


def test_located_field_1d():
    foo = gtx.np_as_located_field(I)(np.zeros((1,)))

    foo[0] = 42

    assert foo.__gt_dims__[0] == I
    assert foo[0] == 42


def test_located_field_2d():
    foo = gtx.np_as_located_field(I, J)(np.zeros((1, 1), dtype=np.float64))

    foo[0, 0] = 42

    assert foo.__gt_dims__[0] == I
    assert foo[0, 0] == 42
    assert foo.dtype == np.float64


def test_tuple_field_concept():
    tuple_of_fields = (make_located_field(), make_located_field())
    assert embedded.can_be_tuple_field(tuple_of_fields)

    field_of_tuples = make_located_field(dtype="f8,f8")
    assert embedded.can_be_tuple_field(field_of_tuples)


def test_field_of_tuple():
    field_of_tuples = make_located_field(dtype="f8,f8")
    assert isinstance(field_of_tuples, embedded.TupleField)


def test_tuple_of_field():
    tuple_of_fields = embedded.TupleOfFields((make_located_field(), make_located_field()))
    assert isinstance(tuple_of_fields, embedded.TupleField)

    tuple_of_fields.field_setitem({I.value: 0, J.value: 0}, (42, 43))
    assert tuple_of_fields.field_getitem({I.value: 0, J.value: 0}) == (42, 43)


def test_tuple_of_tuple_of_field():
    tup = (
        (make_located_field(), make_located_field()),
        (make_located_field(), make_located_field()),
    )
    testee = embedded.TupleOfFields(tup)
    assert isinstance(testee, embedded.TupleField)

    testee.field_setitem({I.value: 0, J.value: 0}, ((42, 43), (44, 45)))
    assert testee.field_getitem({I.value: 0, J.value: 0}) == ((42, 43), (44, 45))
