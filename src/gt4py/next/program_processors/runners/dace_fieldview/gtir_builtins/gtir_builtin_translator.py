# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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


from abc import abstractmethod
from dataclasses import dataclass
from typing import final
from gt4py import eve

import dace
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type, unique_name
from gt4py.next.type_system import type_specifications as ts


@dataclass(frozen=True)
class GtirBuiltinTranslator(eve.NodeVisitor):
    head_state: dace.SDFGState
    sdfg: dace.SDFG

    @final
    def __call__(
        self,
    ) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """The callable interface is used to build the dataflow graph.

        It allows to build the dataflow graph inside a given state starting
        from the innermost nodes, by propagating the intermediate results
        as access nodes to temporary local storage.
        """
        return self.build()

    @final
    def add_local_storage(
        self, data_type: ts.FieldType | ts.ScalarType, shape: list[str]
    ) -> dace.nodes.AccessNode:
        """Allocates temporary storage to be used in the local scope for intermediate results."""
        name = unique_name("var")
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = as_dace_type(data_type.dtype)
            name, _ = self.sdfg.add_array(name, shape, dtype, find_new_name=True, transient=True)
        else:
            assert len(shape) == 0
            dtype = as_dace_type(data_type)
            name, _ = self.sdfg.add_scalar(name, dtype, find_new_name=True, transient=True)
        return self.head_state.add_access(name)

    @abstractmethod
    def build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        """Creates the dataflow subgraph representing a given GTIR builtin.

        This method is used by derived classes of `GtirDataflowBuilder`,
        which build a specialized subgraph for a certain GTIR builtin.

        Returns a list of SDFG nodes and the associated GT4Py data type:
        tuple(node, data_type)

        The GT4Py data type is useful in the case of fields, because it provides
        information on the field domain (e.g. order of dimensions, types of dimensions).
        """
        raise NotImplementedError