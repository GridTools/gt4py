# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import factory

from gt4py.next import backend
from gt4py.next.otf import stages, workflow


class CachedBackendFactory(factory.Factory):
    class Meta:
        model = backend.Backend

    class Params:
        name_cached = ""
        cached = factory.Trait(
            executor=factory.LazyAttribute(
                lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
            ),
            name_cached="_cached",
        )
        hash_function = stages.compilation_hash
