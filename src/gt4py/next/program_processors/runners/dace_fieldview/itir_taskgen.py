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


class ItirTaskgenContext:
    sdfg: dace.SDFG
    state: dace.SDFGState
    node_mapping: dict[str, dace.nodes.AccessNode]
    symrefs: list[str]

    def __init__(
        self,
        current_sdfg: dace.SDFG,
        current_state: dace.SDFGState,
    ):
        self.sdfg = current_sdfg
        self.state = current_state
        self.node_mapping = {}
        self.symrefs = []

    def add_node(self, data: str) -> dace.nodes.AccessNode:
        assert data in self.sdfg.arrays
        self.symrefs.append(data)
        if data in self.node_mapping:
            node = self.node_mapping[data]
        else:
            node = self.state.add_access(data)
            self.node_mapping[data] = node
        return node

    def clone(self) -> "ItirTaskgenContext":
        ctx = ItirTaskgenContext(self.sdfg, self.state)
        ctx.node_mapping = self.node_mapping
        return ctx

    def tasklet_name(self) -> str:
        return f"{self.state.label}_tasklet"

    def var_name(self) -> str:
        return f"{self.state.label}_var"
