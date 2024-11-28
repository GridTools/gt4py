# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.cartesian.gtc import gtir


def prune_unused_parameters(node: gtir.Stencil) -> gtir.Stencil:
    """
    Remove unused parameters from the gtir signature.

    (Maybe this pass should go into a later stage. If you need to touch this pass,
    e.g. when the definition_ir gets removed, consider moving it to a more appropriate
    level. Maybe to the backend IR?)
    """
    assert isinstance(node, gtir.Stencil)
    used_variables = (
        node.walk_values()
        .if_isinstance(gtir.FieldAccess, gtir.ScalarAccess)
        .getattr("name")
        .to_list()
    )
    used_params = list(filter(lambda param: param.name in used_variables, node.params))
    return node.copy(update={"params": used_params})
