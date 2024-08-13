# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.numpy.npir import LocalScalarDecl
from gt4py.cartesian.gtc.numpy.scalars_to_temps import ScalarsToTemporaries

from cartesian_tests.unit_tests.test_gtc.npir_utils import (
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
