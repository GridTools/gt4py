from typing import Dict, List, Optional

from eve.visitors import NodeTranslator

from .. import oir
from . import npir


class OirToNpir(NodeTranslator):
    """Lower from optimizable IR (OIR) to numpy IR (NPIR)."""

    def visit_Stencil(self, node: oir.Stencil) -> npir.Computation:
        domain_padding = {"lower": [0, 0, 0], "upper": [0, 0, 0]}
        vertical_passes = [
            self.visit(vloop, domain_padding=domain_padding) for vloop in node.vertical_loops
        ]

        return npir.Computation(
            field_params=[decl.name for decl in node.params if isinstance(decl, oir.FieldDecl)],
            params=[decl.name for decl in node.params],
            vertical_passes=vertical_passes,
            domain_padding=npir.DomainPadding(
                lower=domain_padding["lower"],
                upper=domain_padding["upper"],
            ),
        )

    def visit_VerticalLoop(
        self, node: oir.VerticalLoop, domain_padding: Optional[Dict[str, List]]
    ):
        return npir.VerticalPass(
            body=[],
            lower=self.visit(node.interval.start),
            upper=self.visit(node.interval.end),
            direction=self.visit(node.loop_order),
        )
