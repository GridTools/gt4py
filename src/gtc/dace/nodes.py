# -*- coding: utf-8 -*-
from typing import List

import dace
import dace.properties
from dace import library

from gtc.oir import VerticalLoop


@library.expansion
class NoLibraryNodeImplementation(dace.library.ExpandTransformation):
    environments: List = []


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    oir_node = dace.properties.DataclassProperty(dtype=VerticalLoop, default=None, allow_none=True)

    def __init__(self, name="unnamed_vloop", oir_node: VerticalLoop = None, *args, **kwargs):
        if oir_node is not None:
            self.oir_node = oir_node
            name = oir_node.id_
        super().__init__(name=name, *args, **kwargs)
