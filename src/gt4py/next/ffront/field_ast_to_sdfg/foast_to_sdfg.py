import dace

import gt4py.eve as eve
from gt4py.next.ffront import program_ast as past
from gt4py.next.type_system import type_specifications as ts



class PastToSDFG(eve.NodeVisitor):
    sdfg: dace.SDFG