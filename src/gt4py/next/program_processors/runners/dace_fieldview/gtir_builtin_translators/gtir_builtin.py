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


import abc
from dataclasses import dataclass
from typing import Callable, TypeAlias, final

import dace

from gt4py.next.program_processors.runners.dace_fieldview import (
    utility as dace_fieldview_util,
)
from gt4py.next.type_system import type_specifications as ts


# Define aliases for return types
SDFGField: TypeAlias = tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]
SDFGFieldBuilder: TypeAlias = Callable[[], list[SDFGField]]


@dataclass(frozen=True)
class GTIRPrimitiveTranslator(abc.ABC):
    sdfg: dace.SDFG
    head_state: dace.SDFGState

    @final
    def __call__(self) -> list[SDFGField]:
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
        if isinstance(data_type, ts.FieldType):
            assert len(data_type.dims) == len(shape)
            dtype = dace_fieldview_util.as_dace_type(data_type.dtype)
            name, _ = self.sdfg.add_array("var", shape, dtype, find_new_name=True, transient=True)
        else:
            assert len(shape) == 0
            dtype = dace_fieldview_util.as_dace_type(data_type)
            name, _ = self.sdfg.add_scalar("var", dtype, find_new_name=True, transient=True)
        return self.head_state.add_access(name)

    @abc.abstractmethod
    def build(self) -> list[SDFGField]:
        """Creates the dataflow subgraph representing a given GTIR builtin.

        This method is used by derived classes of `GTIRBuiltinTranslator`,
        which build a specialized subgraph for a certain GTIR builtin.

        Returns a list of SDFG nodes and the associated GT4Py data type:
        tuple(node, data_type)

        The GT4Py data type is useful in the case of fields, because it provides
        information on the field domain (e.g. order of dimensions, types of dimensions).
        """