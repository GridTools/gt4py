# -*- coding: utf-8 -*-
from typing import List, Tuple

import dace.properties
from dace import SDFG, library

from gtc.common import LoopOrder
from gtc.oir import CacheDesc, HorizontalExecution, Interval, VerticalLoop, VerticalLoopSection


def get_vertical_loop_section_sdfg(section: "VerticalLoopSection") -> SDFG:
    sdfg = SDFG(section.id_)
    state = sdfg.add_state("start_state", is_start_state=True)
    he_iter = iter(section.horizontal_executions)
    last_node = HorizontalExecutionLibraryNode(oir_node=next(he_iter))
    state.add_node(last_node)
    for he in he_iter:
        new_node = HorizontalExecutionLibraryNode(oir_node=he)
        state.add_node(new_node)
        state.add_edge(last_node, None, new_node, None, dace.memlet.Memlet())
        last_node = new_node
    return sdfg


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

    def __init__(self, name="unnamed_vloop", oir_node: VerticalLoop = None, *args, **kwargs):
        if oir_node is not None:
            name = oir_node.id_

        self.loop_order = oir_node.loop_order
        self.sections = [
            (section.interval, get_vertical_loop_section_sdfg(section))
            for section in oir_node.sections
        ]
        self.caches = oir_node.caches

        super().__init__(name=name, *args, **kwargs)


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
