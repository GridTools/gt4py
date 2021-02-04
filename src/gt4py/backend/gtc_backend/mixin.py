# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING

from gt4py.backend.gtc_backend.defir_to_gtir import DefIRToGTIR
from gtc import gtir, gtir_to_oir, oir
from gtc.passes.gtir_dtype_resolver import resolve_dtype
from gtc.passes.gtir_prune_unused_parameters import prune_unused_parameters
from gtc.passes.gtir_upcaster import upcast


if TYPE_CHECKING:
    from gt4py.stencil_builder import StencilBuilder


class GTCBackendMixin:
    GTIR_KEY = "gtc:gtir"
    OIR_KEY = "gtc:oir"
    builder: "StencilBuilder"

    def _make_gtir(self) -> gtir.Stencil:
        gtir = DefIRToGTIR.apply(self.builder.definition_ir)
        gtir_without_unused_params = prune_unused_parameters(gtir)
        dtype_deduced = resolve_dtype(gtir_without_unused_params)
        upcasted = upcast(dtype_deduced)
        return upcasted

    @property
    def gtir(self) -> gtir.Stencil:
        key = self.GTIR_KEY
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_gtir()})
        return self.builder.backend_data[key]

    def _make_oir(self) -> oir.Stencil:
        oir = gtir_to_oir.GTIRToOIR().visit(self.gtir)
        return oir

    @property
    def oir(self) -> oir.Stencil:
        key = self.OIR_KEY
        if key not in self.builder.backend_data:
            self.builder.with_backend_data({key: self._make_oir()})
        return self.builder.backend_data[key]
