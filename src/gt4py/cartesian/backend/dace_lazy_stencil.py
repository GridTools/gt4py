# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set, Tuple

import dace
from dace.frontend.python.common import SDFGConvertible

from gt4py.cartesian.backend.dace_backend import SDFGManager
from gt4py.cartesian.backend.dace_stencil_object import DaCeStencilObject, add_optional_fields
from gt4py.cartesian.backend.module_generator import make_args_data_from_gtir
from gt4py.cartesian.lazy_stencil import LazyStencil


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_builder import StencilBuilder


class DaCeLazyStencil(LazyStencil, SDFGConvertible):
    def __init__(self, builder: StencilBuilder):
        if "dace" not in builder.backend.name:
            raise ValueError("Trying to build a DaCeLazyStencil for non-dace backend.")
        super().__init__(builder=builder)

    @property
    def field_info(self) -> Dict[str, Any]:
        """
        Return same value as compiled stencil object's `field_info` attribute.

        Does not trigger a build.
        """
        return make_args_data_from_gtir(self.builder.gtir_pipeline).field_info

    def closure_resolver(
        self,
        constant_args: Dict[str, Any],
        given_args: Set[str],
        parent_closure: Optional["dace.frontend.python.common.SDFGClosure"] = None,
    ) -> "dace.frontend.python.common.SDFGClosure":
        return dace.frontend.python.common.SDFGClosure()

    def __sdfg__(self, *args, **kwargs) -> dace.SDFG:
        sdfg_manager = SDFGManager(self.builder)
        args_data = make_args_data_from_gtir(self.builder.gtir_pipeline)
        arg_names = [arg.name for arg in self.builder.gtir.api_signature]
        assert args_data.domain_info is not None
        norm_kwargs = DaCeStencilObject.normalize_args(
            *args,
            backend=self.backend.name,
            arg_names=arg_names,
            domain_info=args_data.domain_info,
            field_info=args_data.field_info,
            **kwargs,
        )
        sdfg = sdfg_manager.frozen_sdfg(origin=norm_kwargs["origin"], domain=norm_kwargs["domain"])
        sdfg = add_optional_fields(
            sdfg,
            field_info=args_data.field_info,
            parameter_info=args_data.parameter_info,
            **norm_kwargs,
        )
        return sdfg

    def __sdfg_closure__(self, reevaluate: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        return {}

    def __sdfg_signature__(self) -> Tuple[Sequence[str], Sequence[str]]:
        args = [arg.name for arg in self.builder.gtir.api_signature]
        return (args, [])
