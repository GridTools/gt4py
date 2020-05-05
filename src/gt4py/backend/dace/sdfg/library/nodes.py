import dace
from dace.properties import SymbolicProperty
import dace.library
from .expansions import ForLoopExpandTransformation

from gt4py import ir as gt_ir
from gt4py.backend.dace.util import axis_interval_to_range
from gt4py.backend.dace.sdfg.builder import MappedMemletInfo


@dace.library.node
class ApplyMethodLibraryNode(dace.library.LibraryNode):
    implementations = {"loop": ForLoopExpandTransformation}
    default_implementation = "loop"

    iteration_order = dace.properties.Property(
        dtype=gt_ir.IterationOrder,
        allow_none=False,
        desc="gt4py.ir.IterationOrder",
        default=gt_ir.IterationOrder.PARALLEL,
    )
    read_accesses = dace.properties.Property(
        dtype=dict, default={}, desc="map local symbol to MappedMemletInfo"
    )
    write_accesses = dace.properties.Property(
        dtype=dict, default={}, desc="map local symbol to MappedMemletInfo"
    )
    input_extents = dace.properties.Property(
        dtype=dict, default={}, desc="maximum offsets per dimension in read accesses"
    )
    output_extents = dace.properties.Property(
        dtype=dict, default={}, desc="maximum offsets per dimension in write accesses"
    )
    k_range = dace.properties.RangeProperty(allow_none=True, desc="range as subset descriptor")
    code = dace.properties.CodeProperty(
        default=dace.properties.CodeProperty.from_string("", language=dace.dtypes.Language.Python),
        desc="apply method body",
    )

    def __init__(
        self,
        name,
        read_accesses={},
        write_accesses={},
        iteration_order=gt_ir.IterationOrder.PARALLEL,
        *,
        code=None,
        k_range=None,
        implementation=None,
    ):
        super().__init__(
            name,
            inputs=set("IN_" + info.outer_name for info in read_accesses.values()),
            outputs=set("OUT_" + info.outer_name for info in write_accesses.values()),
        )
        self.iteration_order = iteration_order
        self.read_accesses = read_accesses
        self.write_accesses = write_accesses
        if code is None:
            code = ""
        if isinstance(code, str):
            code = dace.properties.CodeProperty.from_string(
                code, language=dace.dtypes.Language.Python
            )
        self.code = code
        if isinstance(k_range, gt_ir.AxisInterval):
            k_range = axis_interval_to_range(k_range)

        if k_range is not None:
            self.k_range = k_range
        if implementation is not None:
            self.implementation = implementation

        for acc in read_accesses.values():
            offset = (acc.offset.get("I", 0), acc.offset.get("J", 0), acc.offset.get("K", 0))
            if acc.outer_name not in self.input_extents:
                self.input_extents[acc.outer_name] = gt_ir.Extent.from_offset(offset)
            else:
                self.input_extents[acc.outer_name] = self.input_extents[
                    acc.outer_name
                ] | gt_ir.Extent.from_offset(offset)
        for acc in write_accesses.values():
            offset = (acc.offset.get("i", 0), acc.offset.get("j", 0), acc.offset.get("k", 0))
            if acc.outer_name not in self.output_extents:
                self.output_extents[acc.outer_name] = gt_ir.Extent.from_offset(offset)
            else:
                self.output_extents[acc.outer_name] = self.output_extents[
                    acc.outer_name
                ] | gt_ir.Extent.from_offset(offset)
