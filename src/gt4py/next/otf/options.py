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
    connectivities: common.OffsetProviderLike
    static_domains: bool


@dataclasses.dataclass(frozen=True)
class CompilationOptions:
    #: Enable Just-in-Time compilation, otherwise a program has to be compiled manually by a call
    #: to `compile` before calling.
    # Uses a factory to make changes to the config after module import time take effect. This is
    # mostly important for testing. Users should not rely on it.
    enable_jit: bool = dataclasses.field(default_factory=lambda: config.ENABLE_JIT_DEFAULT)

    #: If the user requests static params, they will be used later to initialize CompiledPrograms.
    #: By default the set of static params is set when compiling for the first time, e.g. on call
    #: when jitting is enabled, or on a call to `compile`.
    static_params: Sequence[str] | None = None

    # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    #: A dictionary holding static/compile-time information about the offset providers.
    #: For now, it is used for ahead of time compilation in DaCe orchestrated programs,
    #: i.e. DaCe programs that call GT4Py Programs -SDFGConvertible interface-.
    connectivities: common.OffsetProviderLike | None = None

    static_domains: bool = False

    def __post_init__(self) -> None:
        if self.connectivities is not None:
            object.__setattr__(
                self, "connectivities", common.as_tag_keyed_offset_provider(self.connectivities)
            )
            # the DaCe orchestration reads these directly, without passing an offset provider
            common.check_offset_provider(self.connectivities, deep=True)


assert CompilationOptionsArgs.__annotations__.keys() == CompilationOptions.__annotations__.keys()
