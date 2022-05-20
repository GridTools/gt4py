# GTC Toolchain - GT4Py Project - GridTools Framework
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


from gtc import common
from gtc.numpy.npir import LocalScalarDecl
from gtc.numpy.scalars_to_temps import ScalarsToTemporaries
from tests.test_unittest.test_gtc.npir_utils import (
    ComputationFactory,
    FieldSliceFactory,
    HorizontalBlockFactory,
    LocalScalarAccessFactory,
    VectorAssignFactory,
)


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
