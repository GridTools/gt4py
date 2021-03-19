from typing import Callable, Sequence

from gtc import gtir
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast


PASS_T = Callable[[gtir.Stencil], gtir.Stencil]


class GtirPipeline:
    def __init__(self, node: gtir.Stencil):
        self.gtir = node

    def apply(self, step: PASS_T) -> "GtirPipeline":
        return self.__class__(step(self.gtir))

    def full(self, skip: Sequence[PASS_T] = None) -> "GtirPipeline":
        order = [prune_unused_parameters, resolve_dtype, upcast]
        for step in skip or []:
            order.remove(step)  # type: ignore
        result = self
        for step in order:
            result = result.apply(step)
        return result
