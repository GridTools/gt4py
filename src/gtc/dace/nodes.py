# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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
import typing
from typing import Dict, List, Set, Union

import dace.data
import dace.dtypes
import dace.properties
import dace.subsets
import numpy as np
from dace import library

from gtc import common
from gtc import daceir as dcir
from gtc import oir
from gtc.dace.expansion.expansion import StencilComputationExpansion
from gtc.definitions import Extent
from gtc.oir import Decl, FieldDecl, VerticalLoop, VerticalLoopSection

from .expansion.utils import HorizontalExecutionSplitter, get_dace_debuginfo
from .expansion_specification import ExpansionItem, make_expansion_order


def _set_expansion_order(
    node: "StencilComputation", expansion_order: Union[List[ExpansionItem], List[str]]
):
    res = make_expansion_order(node, expansion_order)
    node._expansion_specification = res


def _set_tile_sizes_interpretation(node: "StencilComputation", tile_sizes_interpretation: str):
    valid_values = {"shape", "strides"}
    if tile_sizes_interpretation not in valid_values:
        raise ValueError(f"tile_sizes_interpretation must be one in {valid_values}.")
    node._tile_sizes_interpretation = tile_sizes_interpretation


class PickledProperty:
    def to_json(self, obj):
        protocol = pickle.DEFAULT_PROTOCOL
        pbytes = pickle.dumps(obj, protocol=protocol)
        jsonobj = dict(pickle=base64.b64encode(pbytes).decode("utf-8"))
        return jsonobj

    @classmethod
    def from_json(cls, d, sdfg=None):
        b64string = d["pickle"]
        byte_repr = base64.b64decode(b64string)
        return pickle.loads(byte_repr)


class PickledDataclassProperty(PickledProperty, dace.properties.DataclassProperty):
    pass


class PickledListProperty(PickledProperty, dace.properties.ListProperty):
    pass


class PickledDictProperty(PickledProperty, dace.properties.DictProperty):
    pass


@library.node
class StencilComputation(library.LibraryNode):
    implementations: Dict[str, dace.library.ExpandTransformation] = {
        "default": StencilComputationExpansion
    }
    default_implementation = "default"

    oir_node = PickledDataclassProperty(dtype=VerticalLoop, allow_none=True)

    declarations = PickledDictProperty(key_type=str, value_type=Decl, allow_none=True)
    extents = PickledDictProperty(key_type=int, value_type=Extent, allow_none=False)
    access_infos = PickledDictProperty(
        key_type=str, value_type=dcir.FieldAccessInfo, allow_none=True
    )

    device = dace.properties.EnumProperty(
        dtype=dace.DeviceType, default=dace.DeviceType.CPU, allow_none=True
    )
    expansion_specification = PickledListProperty(
        element_type=ExpansionItem,
        allow_none=True,
        setter=_set_expansion_order,
    )
    tile_sizes = PickledDictProperty(
        key_type=dcir.Axis,
        value_type=int,
        default={dcir.Axis.I: 8, dcir.Axis.J: 8, dcir.Axis.K: 8},
    )

    tile_sizes_interpretation = dace.properties.Property(
        setter=_set_tile_sizes_interpretation, dtype=str, default="strides"
    )

    symbol_mapping = dace.properties.DictProperty(
        key_type=str, value_type=dace.symbolic.pystr_to_symbolic, default=None, allow_none=True
    )
    _dace_library_name = "StencilComputation"

    def __init__(
        self,
        name="unnamed_vloop",
        oir_node: VerticalLoop = None,
        extents: Dict[int, Extent] = None,
        declarations: Dict[str, Decl] = None,
        expansion_order=None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)

        from gtc.dace.utils import compute_dcir_access_infos

        if oir_node is not None:
            assert extents is not None
            assert declarations is not None
            extents_dict = dict()
            for i, section in enumerate(oir_node.sections):
                for j, he in enumerate(section.horizontal_executions):
                    extents_dict[j * len(oir_node.sections) + i] = extents[id(he)]

            # TODO: Why is this conversion required?
            self.oir_node = typing.cast(PickledDataclassProperty, oir_node)
            self.extents = extents_dict  # type: ignore
            self.declarations = declarations  # type: ignore
            self.symbol_mapping = {
                decl.name: dace.symbol(
                    decl.name,
                    dtype=dace.typeclass(np.dtype(common.data_type_to_typestr(decl.dtype)).type),
                )
                for decl in declarations.values()
                if isinstance(decl, oir.ScalarDecl)
            }
            self.symbol_mapping.update(
                {
                    axis.domain_symbol(): dace.symbol(axis.domain_symbol(), dtype=dace.int32)
                    for axis in dcir.Axis.dims_horizontal()
                }
            )
            self.access_infos = compute_dcir_access_infos(
                oir_node,
                oir_decls=declarations,
                block_extents=self.get_extents,
                collect_read=True,
                collect_write=True,
            )
            if any(
                interval.start.level == common.LevelMarker.END
                or interval.end.level == common.LevelMarker.END
                for interval in oir_node.walk_values()
                .if_isinstance(VerticalLoopSection)
                .getattr("interval")
            ) or any(
                decl.dimensions[dcir.Axis.K.to_idx()]
                for decl in self.declarations.values()
                if isinstance(decl, oir.FieldDecl)
            ):
                self.symbol_mapping[dcir.Axis.K.domain_symbol()] = dace.symbol(
                    dcir.Axis.K.domain_symbol(), dtype=dace.int32
                )

            self.debuginfo = get_dace_debuginfo(oir_node)

        if expansion_order is None:
            expansion_order = [
                "TileI",
                "TileJ",
                "Sections",
                "K",  # Expands to either Loop or Map
                "Stages",
                "I",
                "J",
            ]
        _set_expansion_order(self, expansion_order)

    def get_extents(self, he):
        for i, section in enumerate(self.oir_node.sections):
            for j, cand_he in enumerate(section.horizontal_executions):
                if he is cand_he:
                    return self.extents[j * len(self.oir_node.sections) + i]

    @property
    def field_decls(self) -> Dict[str, FieldDecl]:
        return {
            name: decl for name, decl in self.declarations.items() if isinstance(decl, FieldDecl)
        }

    @property
    def free_symbols(self) -> Set[str]:
        result: Set[str] = set()
        for v in self.symbol_mapping.values():
            result.update(map(str, v.free_symbols))
        return result

    def has_splittable_regions(self):
        for he in self.oir_node.walk_values().if_isinstance(oir.HorizontalExecution):
            if not HorizontalExecutionSplitter.is_horizontal_execution_splittable(he):
                return False
        return True

    @property
    def tile_strides(self):
        if self.tile_sizes_interpretation == "strides":
            return self.tile_sizes
        else:
            overall_extent: Extent = next(iter(self.extents.values()))
            for extent in self.extents.values():
                overall_extent |= extent
            return {
                key: value + overall_extent[key.to_idx()] for key, value in self.tile_sizes.items()
            }
