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


from typing import List, Optional, Tuple, Union

from devtools import debug  # noqa: F401
from pydantic import validator
from pydantic.class_validators import root_validator

from eve import FrozenNode, Node, Str, SymbolTableTrait
from eve.type_definitions import SymbolName, SymbolRef
from gtc_unstructured.irs import common
from gtc import common as stable_gtc_common


TAG_APPENDIX: str = "_tag"


class Expr(Node):
    location_type: common.LocationType


class Stmt(Node):
    location_type: common.LocationType


class FieldAccess(Expr):
    name: SymbolRef  # symbol ref to SidCompositeEntry
    sid: SymbolRef  # symbol ref


class VarDecl(Stmt):
    name: SymbolName
    init: Expr
    vtype: common.DataType


class Literal(Expr):
    value: Union[common.BuiltInLiteral, Str]
    vtype: common.DataType


class VarAccess(Expr):
    name: SymbolRef


class AssignStmt(common.AssignStmt[Union[FieldAccess, VarAccess], Expr], Stmt):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    pass


class Connectivity(FrozenNode):
    name: SymbolName
    max_neighbors: int
    has_skip_values: bool

    @property
    def tag(self):
        return self.name + TAG_APPENDIX

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash((self.name, self.chain))

    def __eq__(self, other):
        return self.name == other.name and self.chain == other.chain


class SidCompositeEntry(FrozenNode):
    ref: SymbolRef  # ref to field
    name: SymbolName  # (don't set: generated from ref)

    @root_validator(pre=True)
    def set_name(cls, values):
        assert "name" not in values
        values["name"] = values["ref"] + TAG_APPENDIX
        return values

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SidCompositeSparseEntry(SidCompositeEntry):
    connectivity: SymbolRef


class SidComposite(Node):
    name: SymbolName
    entries: List[Union[SidCompositeEntry, SidCompositeSparseEntry]]

    @property
    def ptr_name(self):
        return self.name + "_ptrs"

    @property
    def strides_name(self):
        return self.name + "_strides"

    @validator("entries")
    def not_empty(cls, entries):
        if len(entries) < 1:
            raise ValueError("SidComposite must contain at least one entry")
        return entries


class PtrRef(Node):
    name: SymbolName

    @property
    def ptr_name(self):
        return self.name


class NeighborLoop(Stmt, SymbolTableTrait):
    primary_sid: SymbolRef
    secondary_sid: SymbolRef
    connectivity: SymbolRef
    primary: PtrRef
    secondary: PtrRef
    body: List[Stmt]


class Kernel(Node, SymbolTableTrait):
    name: SymbolName
    primary_location: common.LocationType  # TODO probably replace by domain for this location
    primary_composite: SidComposite
    secondary_composites: List[SidComposite]
    body: List[Stmt]


class KernelCall(Node):
    name: SymbolRef


class VerticalDimension(Node):
    pass


class UField(Node):
    name: SymbolName
    vtype: common.DataType
    dimensions: List[Union[common.LocationType, VerticalDimension]]  # Set?

    @property
    def tag(self):
        return self.name + TAG_APPENDIX


class SparseField(Node):
    name: SymbolName
    vtype: common.DataType
    connectivity: SymbolRef
    dimensions: List[Union[common.LocationType, VerticalDimension]]

    @property
    def tag(self):
        return self.name + TAG_APPENDIX


class Temporary(UField):
    pass


class Computation(Node, SymbolTableTrait):
    name: Str
    connectivities: List[Connectivity]
    parameters: List[Union[UField, SparseField]]
    temporaries: List[Temporary]
    kernels: List[Kernel]
    ctrlflow_ast: List[KernelCall]

    _validate_symbol_refs = stable_gtc_common.validate_symbol_refs()
