# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from functional.ffront import field_operator_ast as foast
from functional.ffront.builtins import Field, float64
from functional.ffront.func_to_foast import FieldOperatorParser


def test_unpack_assign():
    def unpack_explicit_tuple(
        a: Field[..., float64], b: Field[..., float64]
    ) -> tuple[Field[..., float64], Field[..., float64]]:
        tmp_a, tmp_b = (a, b)
        return tmp_a, tmp_b

    parsed = FieldOperatorParser.apply_to_func(unpack_explicit_tuple)

    assert parsed.symtable_["tmp_a$0"].type == foast.FieldType(
        dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64, shape=None)
    )
    assert parsed.symtable_["tmp_b$0"].type == foast.FieldType(
        dims=Ellipsis, dtype=foast.ScalarType(kind=foast.ScalarKind.FLOAT64, shape=None)
    )
