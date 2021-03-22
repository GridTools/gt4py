from typing import Callable, Sequence

from gtc import gtir
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast


PASS_T = Callable[[gtir.Stencil], gtir.Stencil]


class GtirPipeline:
    def __init__(self, node: gtir.Stencil):
        self.gtir = node

    def steps(self) -> Sequence[PASS_T]:
        return [prune_unused_parameters, resolve_dtype, upcast]

    def apply(self, steps: Sequence[PASS_T]) -> gtir.Stencil:
        result = self.gtir
        for step in steps:
            result = step(result)
        return result

    def full(self, skip: Sequence[PASS_T] = None) -> gtir.Stencil:
        skip = skip or []
        pipeline = [step for step in self.steps() if step not in skip]
        return self.apply(pipeline)
