# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2021, ETH Zurich
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
def smul_add(in_f, factor=2.0):
    return factor * (in_f + in_f[1, 0, 0])


def double_smul_parallel(in_f: Field[float], out_f: Field[float]):  # type: ignore
    with computation(PARALLEL), interval(...):  # type: ignore
        tmp_f = smul_add(in_f)
        out_f = smul_add(tmp_f)  # type: ignore  # noqa


def double_smul_forward(in_f: Field[float], out_f: Field[float]):  # type: ignore
    with computation(FORWARD), interval(...):  # type: ignore
        tmp_f = smul_add(in_f)
        out_f = smul_add(tmp_f)  # type: ignore  # noqa


def test_demote_temporaries_to_variables_pass_parallel():
    builder = StencilBuilder(double_smul_parallel)
    iir = builder.implementation_ir
    assert iir.temporary_fields == ["tmp_f"]

    multi_stage = iir.multi_stages[0]
    assert len(multi_stage.groups) == 2

    for i in range(len(multi_stage.groups)):
        stage = multi_stage.groups[i].stages[0]
        assert len(stage.apply_blocks[0].body.stmts) == 2


def test_demote_temporaries_to_variables_pass_forward():
    builder = StencilBuilder(double_smul_forward)
    iir = builder.implementation_ir
    assert iir.temporary_fields == ["tmp_f"]

    multi_stage = iir.multi_stages[0]
    assert len(multi_stage.groups) == 2

    for i in range(len(multi_stage.groups)):
        stage = multi_stage.groups[i].stages[0]
        assert len(stage.apply_blocks[0].body.stmts) == 2
