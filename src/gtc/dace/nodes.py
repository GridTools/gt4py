# -*- coding: utf-8 -*-
from typing import List

import dace
from dace import library

from gtc.oir import VerticalLoop


@library.expansion
class NoLibraryNodeImplementation(dace.library.ExpandTransformation):
    environments: List = []


@library.node
class VerticalLoopLibraryNode(dace.nodes.LibraryNode):
    implementations = {"none": NoLibraryNodeImplementation}
    default_implementation = "none"

    _oir_node: VerticalLoop

    def __init__(self, oir_node: VerticalLoop, *args, **kwargs):
        self._oir_node = oir_node
        super().__init__(name=oir_node.id_, *args, **kwargs)
