# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Sequence, TypedDict

from gt4py.next import common, config


class CompilationOptionsArgs(TypedDict, total=False):
    enable_jit: bool
    static_params: Sequence[str]
    connectivities: common.OffsetProvider


@dataclasses.dataclass
class CompilationOptions:
    enable_jit: bool = config.ENABLE_JIT_DEFAULT
    #: if the user requests static params, they will be used later to initialize CompiledPrograms
    static_params: Sequence[str] | None = (
        None  # TODO: describe that this value will eventually be a sequence of strings
    )
    # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    #: A dictionary holding static/compile-time information about the offset providers.
    #: For now, it is used for ahead of time compilation in DaCe orchestrated programs,
    #: i.e. DaCe programs that call GT4Py Programs -SDFGConvertible interface-.
    connectivities: common.OffsetProvider | None = None
