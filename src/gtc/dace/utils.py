from typing import TYPE_CHECKING

from dace import SDFG, InterstateEdge

from .nodes import HorizontalExecutionLibraryNode


if TYPE_CHECKING:
    from gtc.oir import VerticalLoopSection


def get_vertical_loop_section_sdfg(section: "VerticalLoopSection") -> SDFG:
    sdfg = SDFG(section.id_)
    old_state = sdfg.add_state("start_state", is_start_state=True)
    for he in section.horizontal_executions:
        new_state = sdfg.add_state(he.id_ + "_state")
        sdfg.add_edge(old_state, new_state, InterstateEdge())
        new_state.add_node(HorizontalExecutionLibraryNode(oir_node=he))

        old_state = new_state
    return sdfg
