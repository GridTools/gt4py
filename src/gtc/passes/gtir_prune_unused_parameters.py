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

from gtc import gtir


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
