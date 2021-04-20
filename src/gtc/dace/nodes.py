# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import dace.data
import dace.properties
import dace.subsets
import networkx as nx
from dace import library

from gtc.common import DataType, LoopOrder, typestr_to_data_type
from gtc.dace.utils import OIRFieldRenamer, dace_dtype_to_typestr, get_node_name_mapping
from gtc.oir import CacheDesc, HorizontalExecution, Interval, VerticalLoop, VerticalLoopSection


class OIRLibraryNode(ABC, dace.nodes.LibraryNode):
    @abstractmethod
    def as_oir(self):
        raise NotImplementedError("Implement in child class.")


@library.node
class VerticalLoopLibraryNode(OIRLibraryNode):
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

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):

        get_node_name_mapping(parent_state, self)

        for _, sdfg in self.sections:
            sdfg.validate()
            is_correct_node_types = all(
                isinstance(
                    n, (dace.SDFGState, dace.nodes.AccessNode, HorizontalExecutionLibraryNode)
                )
                for n, _ in sdfg.all_nodes_recursive()
            )
            is_correct_data_and_dtype = all(
                isinstance(array, dace.data.Array)
                and typestr_to_data_type(dace_dtype_to_typestr(array.dtype)) != DataType.INVALID
                for array in sdfg.arrays.values()
            )
            if not is_correct_node_types or not is_correct_data_and_dtype:
                raise ValueError("Tried to convert incompatible SDFG to OIR.")

        super().validate(parent_sdfg, parent_state, *args, **kwargs)

    def as_oir(self):

        sections = []
        for interval, sdfg in self.sections:
            horizontal_executions = []
            for state in sdfg.topological_sort(sdfg.start_state):

                for node in (
                    n
                    for n in nx.topological_sort(state.nx)
                    if isinstance(n, HorizontalExecutionLibraryNode)
                ):
                    horizontal_executions.append(
                        OIRFieldRenamer(get_node_name_mapping(state, node)).visit(node.as_oir())
                    )
            sections.append(
                VerticalLoopSection(interval=interval, horizontal_executions=horizontal_executions)
            )

        return VerticalLoop(
            sections=sections,
            loop_order=self.loop_order,
            caches=self.caches,
        )


@library.node
class HorizontalExecutionLibraryNode(OIRLibraryNode):
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

    def as_oir(self):
        return self.oir_node

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):
        get_node_name_mapping(parent_state, self)
