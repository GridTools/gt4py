# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple

import dace
import numpy as np
from dace import SDFG, InterstateEdge, SDFGState
from dace.sdfg.nodes import AccessNode
from dace.subsets import Subset

import gtc.common as common
import gtc.oir as oir
from eve import field
from eve.visitors import NodeVisitor
from gtc.dace.nodes import VerticalLoopLibraryNode


class OirToSDFGVisitor(NodeVisitor):
    class Context:
        sdfg: SDFG = field()
        state: SDFGState = field()
        accesses: Dict[str, Tuple[AccessNode, List[Subset], List[Subset]]] = field(
            default_factory=dict
        )
        is_ij_flat: bool = False
        is_k_flat: bool = True

        def __init__(self, label: str):
            self.sdfg = SDFG(label)
            self.state = self.sdfg.add_state(label + "_init_state")
            self.is_ij_flat = False
            self.is_k_flat = False

        def new_state(self, label: str):
            last_state = self.state
            self.state = self.sdfg.add_state(label)
            self.sdfg.add_edge(last_state, self.state, InterstateEdge())
            return self.state

    def visit_FieldDecl(self, node: oir.FieldDecl, ctx: "OirToSDFGVisitor.Context", **kwargs):
        dtype = np.dtype(common.data_type_to_typestr(node.dtype))
        ctx.sdfg.add_array(name=node.name, shape=(), dtype=dace.typeclass(dtype.name))

    def visit_ScalarDecl(self, node: oir.ScalarDecl, ctx: "OirToSDFGVisitor.Context", **kwargs):
        dtype = np.dtype(common.data_type_to_typestr(node.dtype))
        ctx.sdfg.add_scalar(name=node.name, dtype=dace.typeclass(dtype.name))

    def visit_VerticalLoop(self, node: oir.VerticalLoop, ctx: "OirToSDFGVisitor.Context", **kwargs):
        # wrap self as library node,
        # wire up result
        # new context with sliced arrays

        ctx.new_state(node.id_).add_node(VerticalLoopLibraryNode(oir_node=node))

    def visit_HorizontalExecution(
        self, node: oir.HorizontalExecution, ctx: "OirToSDFGVisitor.Context", **kwargs
    ):
        pass

    def visit_Stencil(self, node: oir.Stencil, **kwargs: Any) -> SDFG:
        context = self.Context(node.name)
        # add arrays
        # add temporary arrays

        self.generic_visit(node, ctx=context, **kwargs)
        return context.sdfg
