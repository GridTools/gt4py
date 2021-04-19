# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple

import dace.properties
import dace.subsets
from dace import library

from gtc.common import LoopOrder
from gtc.oir import CacheDesc, HorizontalExecution, Interval


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
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
        loop_order: LoopOrder = None,
        sections: List[Tuple[Interval, dace.SDFG]] = None,
        caches: List[CacheDesc] = None,
        *args,
        **kwargs,
    ):

        if loop_order is not None:
            self.loop_order = loop_order
            self.sections = sections
            self.caches = caches

        super().__init__(name=name, *args, **kwargs)

    def validate(self, *args, **kwargs):
        for _, sdfg in self.sections:
            sdfg.validate()
        super().validate(*args, **kwargs)


@library.node
class HorizontalExecutionLibraryNode(dace.nodes.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
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
