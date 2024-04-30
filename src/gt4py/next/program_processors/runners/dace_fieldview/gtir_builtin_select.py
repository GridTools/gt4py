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
from gt4py.next.type_system import type_specifications as ts


class GtirBuiltinSelect(GtirTaskletCodegen):
    _true_br_builder: GtirTaskletCodegen
    _false_br_builder: GtirTaskletCodegen

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        true_br_builder: GtirTaskletCodegen,
        false_br_builder: GtirTaskletCodegen,
    ):
        super().__init__(sdfg, state)
        self._true_br_builder = true_br_builder
        self._false_br_builder = false_br_builder

    def _build(self) -> list[tuple[dace.nodes.Node, ts.FieldType | ts.ScalarType]]:
        true_br_args, _ = self._true_br_builder()
        false_br_args, _ = self._false_br_builder()
        assert len(true_br_args) == len(false_br_args)

        output_nodes = []
        for true_br, false_br in zip(true_br_args, false_br_args):
            true_br_node, true_br_type = true_br
            assert isinstance(true_br_node, dace.nodes.AccessNode)
            false_br_node, false_br_type = false_br
            assert isinstance(false_br_node, dace.nodes.AccessNode)
            assert true_br_type == false_br_type
            array_type = self._sdfg.arrays[true_br_node.data]
            access_node = self._add_local_storage(true_br_type, array_type.shape)
            output_nodes.append((access_node, true_br_type))

            data_name = access_node.data
            true_br_output_node = self._true_br_builder._state.add_access(data_name)
            self._true_br_builder._state.add_nedge(
                true_br_node,
                true_br_output_node,
                dace.Memlet.from_array(
                    true_br_output_node.data, true_br_output_node.desc(self._sdfg)
                ),
            )

            false_br_output_node = self._false_br_builder._state.add_access(data_name)
            self._false_br_builder._state.add_nedge(
                false_br_node,
                false_br_output_node,
                dace.Memlet.from_array(
                    false_br_output_node.data, false_br_output_node.desc(self._sdfg)
                ),
            )
        return output_nodes
