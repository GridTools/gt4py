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


import dace

from gt4py.next.program_processors.runners.dace_fieldview.gtir_tasklet_codegen import (
    GtirTaskletCodegen,
)
from gt4py.next.program_processors.runners.dace_fieldview.utility import as_dace_type
from gt4py.next.type_system import type_specifications as ts


class GtirBuiltinSymbolRef(GtirTaskletCodegen):
    _sym_name: str
    _sym_type: ts.FieldType | ts.ScalarType

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        sym_name: str,
        data_type: ts.FieldType | ts.ScalarType,
    ):
        super().__init__(sdfg, state)
        self._sym_name = sym_name
        self._sym_type = data_type

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        if isinstance(self._sym_type, ts.FieldType):
            sym_node = self._state.add_access(self._sym_name)

        else:
            # scalar symbols are passed to the SDFG as symbols
            assert self._sym_name in self._sdfg.symbols
            tasklet_node = self._state.add_tasklet(
                f"get_{self._sym_name}",
                {},
                {"__out"},
                f"__out = {self._sym_name}",
            )
            name = f"{self._state.label}_var"
            dtype = as_dace_type(self._sym_type)
            sym_node = self._state.add_scalar(name, dtype, find_new_name=True, transient=True)
            self._state.add_edge(
                tasklet_node, "__out", sym_node, None, dace.Memlet(data=sym_node.data, subset="0")
            )

        return [(sym_node, self._sym_type)]
