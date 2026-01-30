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
    static_domains: bool


@dataclasses.dataclass(frozen=True)
class CompilationOptions:
    #: Enable Just-in-Time compilation, otherwise a program has to be compiled manually by a call
    #: to `compile` before calling.
    # Uses a factory to make changes to the config after module import time take effect. This is
    # mostly important for testing. Users should not rely on it.
    enable_jit: bool = dataclasses.field(default_factory=lambda: config.ENABLE_JIT_DEFAULT)

    #: if the user requests static params, they will be used later to initialize CompiledPrograms
    static_params: Sequence[str] | None = (
        None  # TODO: describe that this value will eventually be a sequence of strings
    )

    # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    #: A dictionary holding static/compile-time information about the offset providers.
    #: For now, it is used for ahead of time compilation in DaCe orchestrated programs,
    #: i.e. DaCe programs that call GT4Py Programs -SDFGConvertible interface-.
    connectivities: common.OffsetProvider | None = None

    static_domains: bool = False


assert CompilationOptionsArgs.__annotations__.keys() == CompilationOptions.__annotations__.keys()
