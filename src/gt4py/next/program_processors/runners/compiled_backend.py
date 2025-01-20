# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import factory

from gt4py.eve.utils import content_hash
from gt4py.next import backend
from gt4py.next.otf import stages

def _compilation_hash(otf_closure: stages.CompilableProgram) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = otf_closure.args.offset_provider
    return hash(
        (
            otf_closure.data,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(arg for arg in otf_closure.args.args)),
            # Directly using the `id` of the offset provider is not possible as the decorator adds
            # the implicitly defined ones (i.e. to allow the `TDim + 1` syntax) resulting in a
            # different `id` every time. Instead use the `id` of each individual offset provider.
            tuple((k, id(v)) for (k, v) in offset_provider.items()) if offset_provider else None,
            otf_closure.args.column_axis,
        )
    )


class CompiledBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        hash_function = _compilation_hash