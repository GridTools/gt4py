# -*- coding: utf-8 -*-
from typing import List, Tuple

import dace.properties
from dace import library

from gtc.common import LoopOrder
from gtc.oir import CacheDesc, HorizontalExecution, Interval, Stencil, VerticalLoop


@library.expansion
class NoLibraryNodeImplementation(dace.library.ExpandTransformation):
    environments: List = []


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(
        element_type=List[CacheDesc], default=[], allow_none=False
    )

    def __init__(
        self,
        name="unnamed_vloop",
        stencil: Stencil = None,
        oir_node: VerticalLoop = None,
        *args,
        **kwargs,
    ):

        from gtc.oir_to_dace import VerticalLoopSectionOirSDFGBuilder

        if oir_node is not None:
            name = oir_node.id_

        if stencil is not None:
            self.loop_order = oir_node.loop_order
            self.sections = [
                (
                    section.interval,
                    VerticalLoopSectionOirSDFGBuilder.build(section.id_, stencil, section),
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
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    oir_node = dace.properties.DataclassProperty(
        dtype=HorizontalExecution, default=None, allow_none=True
    )

    def __init__(self, name="unnamed_vloop", oir_node: HorizontalExecution = None, *args, **kwargs):
        if oir_node is not None:
            name = oir_node.id_
            self.oir_node = oir_node

        super().__init__(name=name, *args, **kwargs)
