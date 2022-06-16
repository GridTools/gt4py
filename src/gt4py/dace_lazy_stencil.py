from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set, Tuple

import dace
from dace.frontend.python.common import SDFGClosure, SDFGConvertible

from gt4py.lazy_stencil import LazyStencil


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class DaCeLazyStencil(LazyStencil, SDFGConvertible):
    def __init__(self, builder: "StencilBuilder"):
        if "dace" not in builder.backend.name:
            raise ValueError()
        self.builder = builder

    def closure_resolver(
        self,
        constant_args: Dict[str, Any],
        given_args: Set[str],
        parent_closure: Optional[SDFGClosure] = None,
    ) -> SDFGClosure:
        return SDFGClosure()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        return self.implementation.__sdfg__(*args, **kwargs)

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        return self.implementation.__sdfg_signature__()
