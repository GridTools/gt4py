# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import dace.data
import dace.dtypes
import dace.properties
import dace.subsets
import networkx as nx
from dace import library

from gtc.common import DataType, LoopOrder, SourceLocation, typestr_to_data_type
from gtc.dace.utils import (
    CartesianIterationSpace,
    OIRFieldRenamer,
    assert_sdfg_equal,
    dace_dtype_to_typestr,
    get_node_name_mapping,
)
from gtc.oir import CacheDesc, HorizontalExecution, Interval, VerticalLoop, VerticalLoopSection


class OIRLibraryNode(ABC, dace.nodes.LibraryNode):
    @abstractmethod
    def as_oir(self):
        raise NotImplementedError("Implement in child class.")

    @abstractmethod
    def __eq__(self, other):
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
        oir_node: VerticalLoop = None,
        *args,
        **kwargs,
    ):

        if loop_order is not None:
            self.loop_order = loop_order
            self.sections = sections
            self.caches = caches

        super().__init__(name=name, *args, **kwargs)

        if oir_node is not None and oir_node.loc is not None:
            self.debuginfo = dace.dtypes.DebugInfo(
                oir_node.loc.line,
                oir_node.loc.column,
                oir_node.loc.line,
                oir_node.loc.column,
                oir_node.loc.source,
            )

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
        loc = SourceLocation(
            self.debuginfo.start_line or 1,
            self.debuginfo.start_column or 1,
            self.debuginfo.filename or "<unknown>",
        )
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
                VerticalLoopSection(
                    interval=interval, horizontal_executions=horizontal_executions, loc=loc
                )
            )

        return VerticalLoop(
            sections=sections, loop_order=self.loop_order, caches=self.caches, loc=loc
        )

    def __eq__(self, other):
        try:
            assert isinstance(other, VerticalLoopLibraryNode)
            assert self.loop_order == other.loop_order
            assert self.caches == other.caches
            assert len(self.sections) == len(other.sections)
            for (interval1, he_sdfg1), (interval2, he_sdfg2) in zip(self.sections, other.sections):
                assert interval1 == interval2
                assert_sdfg_equal(he_sdfg1, he_sdfg2)
        except AssertionError:
            return False
        return True

    def __hash__(self):
        return super(OIRLibraryNode, self).__hash__()


@library.node
class HorizontalExecutionLibraryNode(OIRLibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "naive"

    oir_node = dace.properties.DataclassProperty(
        dtype=HorizontalExecution, default=None, allow_none=True
    )
    iteration_space = dace.properties.Property(
        dtype=CartesianIterationSpace, default=None, allow_none=True
    )
    _dace_library_name = "oir.HorizontalExecution"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: HorizontalExecution = None,
        iteration_space: CartesianIterationSpace = None,
        debuginfo: dace.dtypes.DebugInfo = None,
        *args,
        **kwargs,
    ):
        if oir_node is not None:
            name = "HorizontalExecution_" + str(id(oir_node))
            self.oir_node = oir_node
            self.iteration_space = iteration_space

        super().__init__(name=name, *args, **kwargs)

        if debuginfo is None and oir_node is not None and oir_node.loc is not None:
            self.debuginfo = dace.dtypes.DebugInfo(
                oir_node.loc.line,
                oir_node.loc.column,
                oir_node.loc.line,
                oir_node.loc.column,
                oir_node.loc.source,
            )
        elif debuginfo is not None:
            self.debuginfo = debuginfo

    def as_oir(self):
        return self.oir_node

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):
        get_node_name_mapping(parent_state, self)

    def __eq__(self, other):
        if not isinstance(other, HorizontalExecutionLibraryNode):
            return False
        return self.as_oir() == other.as_oir()

    def __hash__(self):
        return super(OIRLibraryNode, self).__hash__()
