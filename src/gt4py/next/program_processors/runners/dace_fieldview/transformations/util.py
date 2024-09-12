# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Common functionality for the transformations/optimization pipeline."""

from typing import Iterable, Union, Any, Optional

import dace
from dace.transformation.passes import simplify as dace_passes_simplify

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace_fieldview import (
    transformations as gtx_transformations,
)


def gt_set_iteration_order(
    sdfg: dace.SDFG,
    leading_dim: gtx_common.Dimension,
    validate: bool = True,
    validate_all: bool = False,
) -> Any:
    """Set the iteration order of the Maps correctly.

    Modifies the order of the Map parameters such that `leading_dim`
    is the fastest varying one, the order of the other dimensions in
    a Map is unspecific. `leading_dim` should be the dimensions were
    the stride is one.

    Args:
        sdfg: The SDFG to process.
        leading_dim: The leading dimensions.
        validate: Perform validation during the steps.
        validate_all: Perform extensive validation.
    """
    return sdfg.apply_transformations_once_everywhere(
        gtx_transformations.MapIterationOrder(
            leading_dim=leading_dim,
        ),
        validate=validate,
        validate_all=validate_all,
    )


def gt_simplify(
    sdfg: dace.SDFG,
    validate: bool = True,
    validate_all: bool = False,
    skip: Optional[set[str]] = None,
) -> Any:
    """Performs simplifications on the SDFG in place.

    Instead of calling `sdfg.simplify()` directly, you should use this function,
    as it is specially tuned for GridTool based SDFGs.

    Args:
        sdfg: The SDFG to optimize.
        validate: Perform validation after the pass has run.
        validate_all: Perform extensive validation.
        skip: List of simplify passes that should not be applied.

    Note:
        The reason for this function is that we can influence how simplify works.
        Since some parts in simplify might break things in the SDFG.
        However, currently nothing is customized yet, and the function just calls
        the simplification pass directly.
    """

    return dace_passes_simplify.SimplifyPass(
        validate=validate,
        validate_all=validate_all,
        verbose=False,
        skip=skip,
    ).apply_pass(sdfg, {})




