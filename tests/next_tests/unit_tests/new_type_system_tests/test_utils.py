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

import pytest

from gt4py.next.new_type_system import types, utils


f16 = types.Float16Type()
f32 = types.Float32Type()


def test_flatten_tuples_not_tuple():
    assert utils.flatten_tuples(f16) == [f16]


def test_flatten_tuples_empty():
    ty = types.TupleType([])
    assert utils.flatten_tuples(ty) == []


def test_flatten_tuples_simple():
    ty = types.TupleType([f16, f32])
    assert utils.flatten_tuples(ty) == [f16, f32]


def test_flatten_tuples_nested():
    ty = types.TupleType([f16, types.TupleType([f32, f16])])
    assert utils.flatten_tuples(ty) == [f16, f32, f16]


def test_unflatten_tuples_not_tuple():
    marker = types.Type()
    tys = [f16]
    assert utils.unflatten_tuples(tys, marker) == f16


def test_unflatten_tuples_not_empty():
    tys = []
    structure = types.TupleType([])
    assert utils.unflatten_tuples(tys, structure) == types.TupleType([])


def test_unflatten_tuples_simple():
    marker = types.Type()
    tys = [f16, f32]
    structure = types.TupleType([marker, marker])
    expected = types.TupleType([f16, f32])
    assert utils.unflatten_tuples(tys, structure) == expected


def test_unflatten_tuples_nested():
    marker = types.Type()
    tys = [f16, f32, f16]
    structure = types.TupleType([marker, types.TupleType([marker, marker])])
    expected = types.TupleType([f16, types.TupleType([f32, f16])])
    assert utils.unflatten_tuples(tys, structure) == expected


def test_unflatten_tuples_too_few():
    marker = types.Type()
    tys = [f16, f32]
    structure = types.TupleType([marker, marker, marker])
    with pytest.raises(ValueError):
        utils.unflatten_tuples(tys, structure)


def test_unflatten_tuples_too_many():
    marker = types.Type()
    tys = [f16, f32, f16]
    structure = types.TupleType([marker, marker])
    with pytest.raises(ValueError):
        utils.unflatten_tuples(tys, structure)
