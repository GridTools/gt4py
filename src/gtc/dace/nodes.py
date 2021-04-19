# -*- coding: utf-8 -*-
from typing import List, Tuple

import dace.properties
import dace.subsets
from dace import library

from gtc.common import LoopOrder
from gtc.dace.expansion import NaiveHorizontalExecutionExpansion, NaiveVerticalLoopExpansion
from gtc.oir import CacheDesc, HorizontalExecution, Interval, Stencil, VerticalLoop


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"naive": NaiveVerticalLoopExpansion}
    default_implementation = "naive"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(
        element_type=List[CacheDesc], default=[], allow_none=False
    )

    _dace_library_name = "oir.VerticalLoop"

    def __init__(
        self,
        name="unnamed_vloop",
        stencil: Stencil = None,
        oir_node: VerticalLoop = None,
        *args,
        **kwargs,
    ):

        from gtc.dace.oir_to_dace import VerticalLoopSectionOirSDFGBuilder

        if oir_node is not None:
            name = "VerticalLoop_" + str(id(oir_node))

        if stencil is not None:
            self.loop_order = oir_node.loop_order
            self.sections = [
                (
                    section.interval,
                    VerticalLoopSectionOirSDFGBuilder.build(
                        "VerticalLoopSection_" + str(id(section)), stencil, section
                    ),
                )
                for section in oir_node.sections
            ]
            self.caches = oir_node.caches

        super().__init__(name=name, *args, **kwargs)

    def validate(self, *args, **kwargs):
        for _, sdfg in self.sections:
            sdfg.validate()
        super().validate(*args, **kwargs)


@library.node
class HorizontalExecutionLibraryNode(dace.nodes.LibraryNode):
    implementations = {
        "naive": NaiveHorizontalExecutionExpansion,
    }
    default_implementation = "naive"

    oir_node = dace.properties.DataclassProperty(
        dtype=HorizontalExecution, default=None, allow_none=True
    )

    _dace_library_name = "oir.HorizontalExecution"

    def __init__(self, name="unnamed_vloop", oir_node: HorizontalExecution = None, *args, **kwargs):
        if oir_node is not None:
            name = "HorizontalExecution_" + str(id(oir_node))
            self.oir_node = oir_node

        super().__init__(name=name, *args, **kwargs)
