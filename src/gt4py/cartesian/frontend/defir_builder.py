# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from gt4py.cartesian.frontend.defir_to_gtir import DataDimensionsChecker, UnrollVectorAssignments
from gt4py.cartesian.frontend.nodes import (
    ArgumentInfo,
    ComputationBlock,
    Domain,
    FieldDecl,
    Location,
    StencilDefinition,
    VarDecl,
)


class DefIRBuilder:
    """Assemble DefinitionIR from DefIR nodes build from the Python AST parsing"""

    def __init__(self, stencil_name: str) -> None:
        self.stencil_name = stencil_name

    def build(
        self,
        domain: Domain,
        api_signature: list[ArgumentInfo],
        fields_decls: dict[str, FieldDecl],
        parameter_decls: dict[str, VarDecl],
        computations: list[ComputationBlock],
        externals: dict[str, Any] | None = None,
        sources: dict[str, str] | None = None,
        docstring: str = "",
        loc: Location | None = None,
    ) -> StencilDefinition:
        """Assemble signature, fields and computations nodes into a StencilDefinition"""
        api_fields = [
            fields_decls[item.name] for item in api_signature if item.name in fields_decls
        ]
        parameters = [
            parameter_decls[item.name] for item in api_signature if item.name in parameter_decls
        ]
        definition_ir = StencilDefinition(
            name=self.stencil_name,
            domain=domain,
            api_signature=api_signature,
            api_fields=api_fields,
            parameters=parameters,
            computations=computations,
            externals=externals,
            docstring=docstring,
            loc=loc,
        )

        definition_ir = UnrollVectorAssignments.apply(
            definition_ir,
            fields_decls=fields_decls,
        )

        # We check fields with data dimensions are all fully indexed
        DataDimensionsChecker.apply(definition_ir, fields_decls)

        return definition_ir
