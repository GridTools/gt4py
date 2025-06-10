import dataclasses
from typing import TypedDict, Sequence

from gt4py.next import common, config

class CompilationOptionsArgs(TypedDict, total=False):
    enable_jit: bool
    static_params: Sequence[str]
    connectivities: common.OffsetProvider


@dataclasses.dataclass(frozen=True)
class CompilationOptions:
    enable_jit: bool = config.DEFAULT_ENABLE_JIT
    #: if the user requests static params, they will be used later to initialize CompiledPrograms
    static_params: Sequence[str] = None
    # TODO(ricoh): replace with common.OffsetProviderType once the temporary pass doesn't require the runtime information
    #: A dictionary holding static/compile-time information about the offset providers.
    #: For now, it is used for ahead of time compilation in DaCe orchestrated programs,
    #: i.e. DaCe programs that call GT4Py Programs -SDFGConvertible interface-.
    connectivities: common.OffsetProvider | None = None