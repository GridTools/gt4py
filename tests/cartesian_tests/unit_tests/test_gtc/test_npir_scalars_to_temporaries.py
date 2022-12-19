# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from cartesian_tests.unit_tests.test_gtc.npir_utils import (
    ComputationFactory,
    FieldSliceFactory,
    HorizontalBlockFactory,
    LocalScalarAccessFactory,
    VectorAssignFactory,
)

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.numpy.npir import LocalScalarDecl
from gt4py.cartesian.gtc.numpy.scalars_to_temps import ScalarsToTemporaries


def test_local_scalar_to_npir_temp() -> None:
    computation = ComputationFactory(
        vertical_passes__0__body__0=HorizontalBlockFactory(
            body=[
                VectorAssignFactory(
                    left=LocalScalarAccessFactory(name="tmp"), right=FieldSliceFactory(name="a")
                ),
                VectorAssignFactory(
                    left=FieldSliceFactory(name="b"), right=LocalScalarAccessFactory(name="tmp")
                ),
            ],
            declarations=[LocalScalarDecl(name="tmp", dtype=common.DataType.FLOAT32)],
        )
    )
    computation = ScalarsToTemporaries().visit(computation)

    # Check that it lowered the local scalar to a temporary field in npir.
    assert "tmp" in {decl.name for decl in computation.temp_decls}
