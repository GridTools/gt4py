from dace.transformation.transformation import Transformation

from gtc import oir
from gtc.dace import dace_to_oir
from gtc.dace.oir_to_dace import OirSDFGBuilder
from gtc.dace.utils import iter_vertical_loop_section_sub_sdfgs


def optimize_horizontal_executions(
    stencil: oir.Stencil, transformation: Transformation
) -> oir.Stencil:
    sdfg = OirSDFGBuilder.build(stencil.name, stencil)
    for subgraph in iter_vertical_loop_section_sub_sdfgs(sdfg):
        subgraph.apply_transformations_repeated(transformation, validate_all=True)
    return dace_to_oir.convert(sdfg)
