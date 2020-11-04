# -*- coding: utf-8 -*-
#
# Eve Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2020, CSCS - Swiss National Supercomputing Center, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import enum
from typing import List, Optional, Tuple, Union

from devtools import debug
from eve import StrEnum  # noqa: F401
from pydantic import root_validator, validator

import eve
from eve import Node, Str
from . import common
from .stencil_ast_nodes import *

from pydantic.tools import parse_file_as

# GT prefixed nodes are nodes representing the GridTools C++ entities


class VerticalDimension(Node):
    pass


# class ApiField(Node):
#     name: Str
#     vtype: common.DataType
#     dimensions: List[Union[common.LocationType, NeighborChain, VerticalDimension]]  # Set?


class Temporary(Node):
    name: Str
    vtype: common.DataType


class GTGrid(Node):
    pass


class GTInterval(Node):
    pass


class GTApplyMethod(Node):
    interval: GTInterval
    body: List[Stmt]


@enum.unique
class Intent(StrEnum):
    IN = "in"
    INOUT = "inout"


class GTExtent(Node):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    @classmethod
    def zero(cls):
        return cls(i=(0, 0), j=(0, 0), k=(0, 0))


class GTAccessor(Node):
    name: Str
    id: int
    intent: Intent
    extent: GTExtent


class GTParamList(Node):
    accessors: List[GTAccessor]


class GTFunctor(Node):
    name: Str
    applies: List[GTApplyMethod]
    param_list: GTParamList


# A ParamArg is an argument that maps to a parameter of something with the same name.
# Because all things are called exactly once there is a one-to-one mapping.
class ParamArg(Node):
    name: Str


class GTStage(Node):
    functor: str  # symbol ref
    args: List[ParamArg]  # symbol ref to GTComputation params


class IJCache(Node):
    name: Str  # symbol ref to GTComputation params or temporaries


class GTMultiStage(Node):
    loop_order: common.LoopOrder
    stages: List[GTStage]  # TODO at least one
    caches: List[Union[IJCache]]


class AccessorRef(Expr):
    name: Str  # symbol ref to param list
    offset: Offset


class AssignStmt(Stmt):
    left: Union[AccessorRef, VarAccess]
    right: Expr


# A GridTools computation object
class GTComputation(Node):
    name: Str
    parameters: List[ParamArg]  # ?
    temporaries: List[Temporary]
    multistages: List[GTMultiStage]  # TODO at least one


#     parameters: List[UField]
#     temporaries: List[Temporary]
#     kernels: List[Kernel]
#     ctrlflow_ast: List[KernelCall]


class CtrlFlowStmt(Node):
    pass


class Computation(Node):
    name: Str
    # The ParamArg here, doesn't fully work as we need the type for template instantiation.
    # But maybe the module instantiation code is actually generated from a different IR?
    parameters: List[ParamArg]
    functors: List[GTFunctor]
    ctrl_flow_ast: List[GTComputation]


#     parameters: List[UField]
#     temporaries: List[Temporary]
#     kernels: List[Kernel]
#     ctrlflow_ast: List[KernelCall]
