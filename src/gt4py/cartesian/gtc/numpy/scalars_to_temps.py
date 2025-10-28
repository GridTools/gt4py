# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""An optimization to convert npir.LocalScalarDecl to npir.TemporaryDecl."""

from dataclasses import dataclass
from typing import Dict

from gt4py import eve
from gt4py.cartesian import utils
from gt4py.cartesian.gtc import common
from gt4py.cartesian.gtc.definitions import Extent
from gt4py.cartesian.gtc.numpy import npir


@dataclass
class Temporary:
    name: str
    dtype: common.DataType
    extent: Extent
    dimensions: tuple[bool, bool, bool]


def _all_local_scalars_are_unique_type(stencil: npir.Computation) -> bool:
    all_declarations = utils.flatten(
        stencil.walk_values().if_isinstance(npir.HorizontalBlock).getattr("declarations").to_list()
    )

    name_to_dtype: Dict[str, common.DataType] = {}
    for decl in all_declarations:
        if decl.name in name_to_dtype:
            if decl.dtype != name_to_dtype[decl.name]:
                return False
        else:
            name_to_dtype[decl.name] = decl.dtype

    return True


class ScalarsToTemporaries(eve.NodeTranslator):
    def visit_LocalScalarAccess(
        self, node: npir.LocalScalarAccess, *, temps_from_scalars: Dict[str, Temporary]
    ) -> npir.FieldSlice:
        return npir.FieldSlice(
            name=node.name,
            i_offset=0,
            j_offset=0,
            k_offset=0,
            dtype=temps_from_scalars[node.name].dtype,
        )

    def visit_HorizontalBlock(
        self, node: npir.HorizontalBlock, *, temps_from_scalars: Dict[str, Temporary]
    ) -> npir.HorizontalBlock:
        for decl in node.declarations:
            if decl.name not in temps_from_scalars:
                temps_from_scalars[decl.name] = Temporary(
                    name=decl.name,
                    dtype=decl.dtype,
                    extent=node.extent,
                    dimensions=(True, True, True),
                )
            else:
                temps_from_scalars[decl.name].extent |= node.extent

        body = self.visit(node.body, temps_from_scalars=temps_from_scalars)

        # Block without declarations
        return node.copy(update={"body": body, "declarations": []})

    def visit_Computation(self, node: npir.Computation) -> npir.Computation:
        if not _all_local_scalars_are_unique_type(node):
            raise TypeError(
                "Local scalars exist with different types and same name. "
                "The numpy backend currently assumes this is not the case."
            )

        temps_from_scalars: Dict[str, Temporary] = {}

        vertical_passes = self.visit(node.vertical_passes, temps_from_scalars=temps_from_scalars)

        scalar_temp_decls = [
            npir.TemporaryDecl(
                name=d.name,
                offset=(-d.extent[0][0], -d.extent[1][0]),
                padding=(d.extent[0][1] - d.extent[0][0], d.extent[1][1] - d.extent[1][0]),
                dimensions=d.dimensions,
                dtype=d.dtype,
            )
            for d in temps_from_scalars.values()
        ]

        return node.copy(
            update={
                "vertical_passes": vertical_passes,
                "temp_decls": node.temp_decls + scalar_temp_decls,
            }
        )
