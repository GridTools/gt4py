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


from typing import Optional

import dace

from gt4py.next.program_processors.runners.dace_fieldview.gtir_builtins.gtir_builtin_translator import (
    GTIRBuiltinTranslator,
    SDFGField,
)
from gt4py.next.type_system import type_specifications as ts


class GTIRBuiltinSymbolRef(GTIRBuiltinTranslator):
    """Generates the dataflow subgraph for a `itir.SymRef` node."""

    sym_name: str
    sym_type: ts.FieldType | ts.ScalarType

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sym_name: str,
        sym_type: ts.FieldType | ts.ScalarType,
    ):
        super().__init__(sdfg, state)
        self.sym_name = sym_name
        self.sym_type = sym_type

    def _get_access_node(self) -> Optional[dace.nodes.AccessNode]:
        """Returns, if present, the access node in current state for the data symbol."""
        access_nodes = [
            node
            for node in self.head_state.nodes()
            if isinstance(node, dace.nodes.AccessNode) and node.data == self.sym_name
        ]
        if len(access_nodes) == 0:
            return None
        assert len(access_nodes) == 1
        return access_nodes[0]

    def build(self) -> list[SDFGField]:
        # check if access node is already present in current state
        sym_node = self._get_access_node()
        if sym_node is None:
            if isinstance(self.sym_type, ts.FieldType):
                # add access node to current state
                sym_node = self.head_state.add_access(self.sym_name)

            else:
                # scalar symbols are passed to the SDFG as symbols: build tasklet node
                # to write the symbol to a scalar access node
                assert self.sym_name in self.sdfg.symbols
                tasklet_node = self.head_state.add_tasklet(
                    f"get_{self.sym_name}",
                    {},
                    {"__out"},
                    f"__out = {self.sym_name}",
                )
                sym_node = self.add_local_storage(self.sym_type, shape=[])
                self.head_state.add_edge(
                    tasklet_node,
                    "__out",
                    sym_node,
                    None,
                    dace.Memlet(data=sym_node.data, subset="0"),
                )

        return [(sym_node, self.sym_type)]
