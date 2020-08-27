# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.gtscript import FORWARD, PARALLEL, Field, computation, function, interval
from gt4py.stencil_builder import StencilBuilder


@function
def smul_add(in_f, factor=-4.0):
    return factor * (in_f[0, 0, 0] + in_f[1, 0, 0])


def double_smul_parallel(in_f: Field[float], out_f: Field[float]):
    with computation(PARALLEL), interval(...):
        tmp_f = smul_add(in_f)
        out_f = smul_add(tmp_f)


def double_smul_forward(in_f: Field[float], out_f: Field[float]):
    with computation(FORWARD), interval(...):
        tmp_f = smul_add(in_f)
        out_f = smul_add(tmp_f)


def test_demote_temporaries_to_variables_pass_parallel():
    builder = StencilBuilder(double_smul_parallel)
    assert builder.implementation_ir.temporary_fields == ["tmp_f"]


def test_demote_temporaries_to_variables_pass_forward():
    builder = StencilBuilder(double_smul_forward)
    assert builder.implementation_ir.temporary_fields == ["tmp_f"]
