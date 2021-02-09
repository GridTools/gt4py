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


class Expr(Node):
    location_type: common.LocationType


class Stmt(Node):
    location_type: common.LocationType


class FieldAccess(Expr):
    name: SymbolRef  # symbol ref to SidCompositeEntry
    sid: SymbolRef  # symbol ref


class VarDecl(Stmt):
    name: Str
    init: Expr
    vtype: common.DataType


class Literal(Expr):
    value: Union[common.BuiltInLiteral, Str]
    vtype: common.DataType


class VarAccess(Expr):
    name: Str  # via symbol table
    dummy: Optional[
        Str
    ]  # to distinguish from FieldAccess, see https://github.com/eth-cscs/eve_toolchain/issues/34


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
        return self.name + "_tag"

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash((self.name, self.chain))

    def __eq__(self, other):
        return self.name == other.name and self.chain == other.chain


class SidCompositeEntry(FrozenNode):
    ref: SymbolRef  # ref to field
    name: SymbolName  # generated from ref

    @root_validator(pre=True)
    def set_name(cls, values):
        values["name"] = values["ref"] + "_tag"
        return values

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SidCompositeSparseEntry(SidCompositeEntry):
    connectivity: SymbolRef


class SidCompositeNeighborTableEntry(FrozenNode):
    connectivity: SymbolRef
    connectivity_deref_: Optional[
        Connectivity
    ]  # TODO temporary workaround for symbol tbl reference

    @property
    def tag_name(self):
        return self.connectivity + "_tag"  # TODO
        # return self.connectivity_deref_.neighbor_tbl_tag

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.connectivity)

    def __eq__(self, other):
        return self.connectivity == other.connectivity


class SidComposite(Node):
    name: SymbolName
    # location: NeighborChain
    entries: List[
        Union[SidCompositeEntry, SidCompositeSparseEntry, SidCompositeNeighborTableEntry]
    ]  # TODO ensure tags are unique

    @property
    def with_connectivity(self) -> bool:
        for e in self.entries:
            if isinstance(e, SidCompositeNeighborTableEntry):
                return True
        return False

    # node private symbol table to entries
    @property
    def symbol_tbl(self):
        return {e.name: e for e in self.entries if isinstance(e, SidCompositeEntry)}

    @property
    def field_name(self):
        return self.name + "_fields"

    @property
    def ptr_name(self):
        return self.name + "_ptrs"

    @property
    def origin_name(self):
        return self.name + "_origins"

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
    primary_location: common.LocationType  # TODO probably replace by domain for this location or not needed?
    primary_composite: SidComposite  # TODO maybe the composites should live in the Call
    secondary_composites: List[SidComposite]
    body: List[Stmt]

    # private symbol table
    @property
    def symbol_tbl(self):
        return {**{s.name: s for s in self.sids}, **{c.name: c for c in self.connectivities}}


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
        return self.name + "_tag"


class SparseField(Node):
    name: SymbolName
    vtype: common.DataType
    connectivity: SymbolRef
    dimensions: List[Union[common.LocationType, VerticalDimension]]

    @property
    def tag(self):
        return self.name + "_tag"


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
