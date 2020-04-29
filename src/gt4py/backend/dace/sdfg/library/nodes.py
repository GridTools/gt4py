import dace
from dace.properties import SymbolicProperty
import dace.library
from .expansions import ForLoopExpandTransformation


@dace.library.node
class ApplyMethodLibraryNode(dace.library.LibraryNode):
    implementations = {"loop": ForLoopExpandTransformation}
    default_implementation = "loop"
    iteration_order = dace.properties.Property(
        dtype=str, allow_none=False, desc="'parallel', 'forward' or 'backward'"
    )
    range = dace.properties.RangeProperty(allow_none=True, desc="range as subset descriptor")
    code = dace.properties.CodeProperty(default="", desc="apply method body")

    def __init__(
        self,
        name,
        inputs=[],
        outputs=[],
        iteration_order="forward",
        code="",
        *,
        k_range=None,
        implementation=None,
    ):
        super().__init__(
            name,
            inputs=set("IN_" + input for input in inputs),
            outputs=set("OUT_" + output for output in outputs),
        )
        self.iteration_order = iteration_order
        self.code = code
        if k_range is not None:
            self.k_range = k_range
        if implementation is not None:
            self.implementation = implementation
