# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main module of the analysis pipeline.
"""

import pprint

from gt4py import ir as gt_ir
from gt4py.analysis import TransformData

from .passes import (
    BuildIIRPass,
    ComputeExtentsPass,
    ComputeUsedSymbolsPass,
    DataTypePass,
    DemoteLocalTemporariesToVariablesPass,
    HousekeepingPass,
    InitInfoPass,
    MergeBlocksPass,
    NormalizeBlocksPass,
)


class IRTransformer:
    """Analysis pipeline implementation.

    A sequence of :class:`gt4py.analysis.common.TransformPass` objects
    is created and applied to the passed :class:`gt4py.ir.StencilDefinition`.
    Those passes share the transformation context by using an instance
    of the :class:`gt4py.analysis.common.TransformData` object.
    """

    @classmethod
    def apply(cls, definition_ir, options):
        """Utility method used to run the pipeline.

        An instance is created behind the scenes and called with the input data.

        Parameters
        ----------
        definition_ir : `gt4py.ir_old.StencilDefinition`
            High-level IR with the definition of the stencil.
        options : `gt4py.definitions.Options`
            Build options provided by the user.

        Returns
        -------
        implementation_ir : `gt4py.ir_old.StencilImplementation`
            Implementation IR with the final implementation of the stencil.
        """
        return cls()(definition_ir, options)

    def __init__(self):
        self.transform_data = None

    def __call__(self, definition_ir, options):
        implementation_ir = gt_ir.StencilImplementation(
            name=definition_ir.name,
            api_signature=[],
            domain=definition_ir.domain,
            fields={},
            parameters={},
            splitters=definition_ir.splitters,
            multi_stages=[],
            fields_extents={},
            unreferenced=[],
            axis_splitters_var=None,
            externals=definition_ir.externals,
            sources=definition_ir.sources,
            docstring=definition_ir.docstring,
        )
        self.transform_data = TransformData(
            definition_ir=definition_ir, implementation_ir=implementation_ir, options=options
        )

        # Initialize auxiliary data
        init_pass = InitInfoPass()
        init_pass.apply(self.transform_data)

        # Turn compute units into atomic execution units
        normalize_blocks_pass = NormalizeBlocksPass()
        normalize_blocks_pass.apply(self.transform_data)

        # Compute stage extents
        compute_extent_pass = ComputeExtentsPass()
        compute_extent_pass.apply(self.transform_data)

        # Merge compatible blocks
        merge_blocks_pass = MergeBlocksPass()
        merge_blocks_pass.apply(self.transform_data)

        # Compute used symbols
        compute_used_symbols_pass = ComputeUsedSymbolsPass()
        compute_used_symbols_pass.apply(self.transform_data)

        # Build IIR
        build_iir_pass = BuildIIRPass()
        build_iir_pass.apply(self.transform_data)

        # Fill in missing dtypes
        data_type_pass = DataTypePass()
        data_type_pass.apply(self.transform_data)

        # turn temporary fields that are only written and read within the same function
        # into local scalars
        demote_local_temporaries_to_variables_pass = DemoteLocalTemporariesToVariablesPass()
        demote_local_temporaries_to_variables_pass.apply(self.transform_data)

        # prune some stages that don't have effect
        housekeeping_pass = HousekeepingPass()
        housekeeping_pass.apply(self.transform_data)

        if options.build_info is not None:
            options.build_info["def_ir"] = self.transform_data.definition_ir
            options.build_info["iir"] = self.transform_data.implementation_ir
            options.build_info["symbol_info"] = self.transform_data.symbols

        return self.transform_data.implementation_ir


transform = IRTransformer.apply
