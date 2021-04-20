from dace.transformation.transformation import Transformation

from gtc import dace_to_oir, oir
from gtc.dace.utils import iter_vertical_loop_section_sub_sdfgs
from gtc.oir_to_dace import OirSDFGBuilder


def optimize_horizontal_executions(
    stencil: oir.Stencil, transformation: Transformation
) -> oir.Stencil:
    sdfg = OirSDFGBuilder.build(stencil.name, stencil)
    for subgraph in iter_vertical_loop_section_sub_sdfgs(sdfg):
        subgraph.apply_transformations_repeated(transformation, validate_all=True)
    return dace_to_oir.convert(sdfg)
