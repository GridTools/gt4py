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

import base64
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import dace.data
import dace.dtypes
import dace.properties
import dace.subsets
import networkx as nx
from dace import library

from gt4py.definitions import Extent
from gtc import oir
from gtc.common import DataType, LoopOrder, SourceLocation, VariableKOffset, typestr_to_data_type
from gtc.dace.utils import OIRFieldRenamer, dace_dtype_to_typestr, get_node_name_mapping
from gtc.oir import CacheDesc, HorizontalExecution, Interval, VerticalLoop, VerticalLoopSection


class OIRLibraryNode(ABC, dace.nodes.LibraryNode):
    @abstractmethod
    def as_oir(self):
        raise NotImplementedError("Implement in child class.")

    def to_json(self, parent):
        protocol = pickle.DEFAULT_PROTOCOL
        pbytes = pickle.dumps(self, protocol=protocol)

        jsonobj = super().to_json(parent)
        jsonobj["classpath"] = dace.nodes.full_class_path(self)
        jsonobj["attributes"]["protocol"] = protocol
        jsonobj["attributes"]["pickle"] = base64.b64encode(pbytes).decode("utf-8")

        return jsonobj

    @classmethod
    def from_json(cls, json_obj, context=None):
        if "attributes" not in json_obj:
            b64string = json_obj["pickle"]
        else:
            b64string = json_obj["attributes"]["pickle"]
        pbytes = base64.b64decode(b64string)
        return pickle.loads(pbytes)


@library.node
class VerticalLoopLibraryNode(OIRLibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "naive"

    loop_order = dace.properties.Property(dtype=LoopOrder, default=None, allow_none=True)
    sections = dace.properties.ListProperty(
        element_type=Tuple[Interval, dace.SDFG], default=[], allow_none=False
    )
    caches = dace.properties.ListProperty(element_type=CacheDesc, default=[], allow_none=False)
    default_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    ijcache_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    kcache_storage_type = dace.properties.EnumProperty(
        dtype=dace.StorageType, default=dace.StorageType.Default
    )
    tiling_map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    tile_sizes = dace.properties.ListProperty(element_type=int, default=None, allow_none=True)

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


@dataclass
class PreliminaryHorizontalExecution:
    body: List[oir.Stmt]
    declarations: List[oir.LocalScalar]


@library.node
class HorizontalExecutionLibraryNode(OIRLibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {}
    default_implementation = "naive"

    _oir_node: Union[HorizontalExecution, PreliminaryHorizontalExecution] = None
    extent = dace.properties.Property(dtype=Extent, default=None, allow_none=True)

    map_schedule = dace.properties.EnumProperty(
        dtype=dace.ScheduleType, default=dace.ScheduleType.Default
    )
    index_symbols = dace.properties.ListProperty(element_type=str, default=["i", "j", "0"])
    global_domain_symbols = dace.properties.ListProperty(element_type=str, default=["__I", "__J"])

    _dace_library_name = "oir.HorizontalExecution"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: HorizontalExecution = None,
        extent: Extent = None,
        debuginfo: dace.dtypes.DebugInfo = None,
        *args,
        **kwargs,
    ):
        if oir_node is not None:
            name = "HorizontalExecution_" + str(id(oir_node))
            self._oir_node = oir_node
            self.extent = extent

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

    @property
    def oir_node(self):
        return self._oir_node

    def commit_horizontal_execution(self):
        self._oir_node = HorizontalExecution(
            body=self._oir_node.body, declarations=self._oir_node.declarations
        )

    def as_oir(self):
        self.commit_horizontal_execution()
        return self.oir_node

    def validate(self, parent_sdfg: dace.SDFG, parent_state: dace.SDFGState, *args, **kwargs):
        get_node_name_mapping(parent_state, self)

    @property
    def free_symbols(self):
        res = super().free_symbols
        if len(self.oir_node.iter_tree().if_isinstance(VariableKOffset).to_list()) > 0:
            res.add("k")
        return res
