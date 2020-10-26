# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Definitions and utilities used by all the analysis pipeline components.
"""

import abc

from gt4py import ir as gt_ir
from gt4py import utils as gt_utils
from gt4py.definitions import BuildOptions, CartesianSpace, Extent, NumericTuple
from gt4py.utils.attrib import Any
from gt4py.utils.attrib import Dict as DictOf
from gt4py.utils.attrib import List as ListOf
from gt4py.utils.attrib import Optional
from gt4py.utils.attrib import Set as SetOf
from gt4py.utils.attrib import Tuple as TupleOf
from gt4py.utils.attrib import attribclass, attribute


@attribclass
class SymbolInfo:
    """`AttribClass` class containing all the data related to a symbol.

    Parameters
    ----------
    decl : `gridtools.ir.Decl`
        Definition statement.
    has_redundancy : `bool`
        True if the symbol is a data field allocated with redundancy.
    in_use : `bool`
        True if the symbol is used.
    """

    decl = attribute(of=gt_ir.Decl)
    has_redundancy = attribute(of=bool, default=False)
    in_use = attribute(of=bool, default=False)

    @property
    def is_field(self):
        return isinstance(self.decl, gt_ir.FieldDecl)

    @property
    def is_api(self):
        return self.decl.is_api

    @property
    def is_parameter(self):
        return self.is_api and not self.is_field


@attribclass(frozen=True)
class IntervalInfo:
    """`AttribClass` class representing an AxisInterval definition.

    Interval specification: (start_level, start_offset), (end_level, end_offset)
    End is not included

    Parameters
    ----------
    start : `tuple` of `int`
        Start level and offset (included).
    end : `tuple` of `int`
        End level and offset (not included).
    """

    start = attribute(of=TupleOf[int, int])
    end = attribute(of=TupleOf[int, int])

    def as_tuple(self, k_interval_sizes: list) -> NumericTuple:
        start = sum(k_interval_sizes[: self.start[0]]) + self.start[1]
        end = sum(k_interval_sizes[: self.end[0]]) + self.end[1]
        return NumericTuple(start, end)

    def overlaps(self, other, k_interval_sizes: list):
        actual_self = self.as_tuple(k_interval_sizes)
        if isinstance(other, IntervalInfo):
            other = other.as_tuple(k_interval_sizes)

        if actual_self[0] < other[0]:
            # Self starts at lower level
            result = True if actual_self[1] > other[0] else False
        else:
            # Other starts at lower level
            result = True if other[1] > actual_self[0] else False

        return result

    def precedes(self, other, k_interval_sizes: list, order: gt_ir.IterationOrder):
        actual_self = self.as_tuple(k_interval_sizes)
        if isinstance(other, IntervalInfo):
            other = other.as_tuple(k_interval_sizes)
        if order == gt_ir.IterationOrder.FORWARD:
            result = actual_self[0] < other[0]
        else:
            result = actual_self[1] > other[1]
        return result


@attribclass
class StatementInfo:
    """`AttribClass` class defining a single operation in a stencil computation.

    This is the smallest piece of a computation considered in the
    analysis pipeline.

    Parameters
    ----------
    id : `int`
        Operation ID.
    stmt : `gridtools.ir.Statement`
        Statement containing the description of the computation
    inputs : `dict` [`str`, `gt4py.definitions.Extent`]
        Each input to this statement and extent
    outputs : `set` [`str`]
        Outputs from this statement (with zero extent)
    """

    id = attribute(of=int)
    stmt = attribute(of=gt_ir.Statement)
    inputs = attribute(of=DictOf[str, Extent], factory=dict)
    outputs = attribute(of=SetOf[str], factory=set)


@attribclass
class IntervalBlockInfo:
    """`AttribClass` class defining a vertical region computation.

    Parameters
    ----------
    id : `int`
        Unique identifier.
    intervals : IntervalInfo`
        Sequential-axis interval to which this block is applied.
    stmts : `list` [`StatementInfo`]
        List of operations.
    inputs : `dict` [`str`, `gt4py.definitions.Extent`]
        Inputs (with extent) to these operations.
    outputs : `set` [`str`]
        Outputs from these operations (with zero extent).
    """

    id = attribute(of=int)
    interval = attribute(of=IntervalInfo)
    stmts = attribute(of=ListOf[StatementInfo], factory=list)
    inputs = attribute(of=DictOf[str, Extent], factory=dict)
    outputs = attribute(of=SetOf[str], factory=set)


@attribclass
class IJBlockInfo:
    """`AttribClass` class defining a vertical region computation.

    Parameters
    ----------
    id : `int`
        Unique identifier.
    intervals : `set` [`IntervalInfo`]
        Set of sequential-axis intervals over which this block iterates
    interval_blocks : `list` [`IntervalBlockInfo`]
        List of blocks (each has a list of statements) that this block executes.
    inputs : `dict` [`str`, `gt4py.definitions.Extent`]
        Each input to this block with extent.
    outputs : `set` [`str`]
        Outputs from this block (with zero extent).
    compute_extent : `gt4py.definitions.Extent`
        Compute extent for this block.
    """

    id = attribute(of=int)
    intervals = attribute(of=SetOf[IntervalInfo])
    interval_blocks = attribute(of=ListOf[IntervalBlockInfo], factory=list)
    inputs = attribute(of=DictOf[str, Extent], factory=dict)
    outputs = attribute(of=SetOf[str], factory=set)
    compute_extent = attribute(of=Extent, optional=True)


@attribclass
class DomainBlockInfo:
    """`AttribClass` class defining a vertical region computation.

    Domain blocks become multi-stages in the IIR.

    Parameters
    ----------
    id : `int`
        Unique identifier.
    iteration_order : `gt4py.ir.IterationOrder`
        The iteration order of the resulting multistage.
    intervals : `set` [`IntervalInfo`]
        Set of sequential-axis intervals over which this block iterates.
    ij_blocks : `list` [`IJBlockInfo`]
        List of stage blocks.
    inputs : `dict` [`str`, `gt4py.definitions.Extent`]
        Each input to this block with extent.
    outputs : `set` [`str`]
        Outputs from this block (with zero extent).
    """

    id = attribute(of=int)
    iteration_order = attribute(of=gt_ir.IterationOrder)
    intervals = attribute(of=SetOf[IntervalInfo])
    ij_blocks = attribute(of=ListOf[IJBlockInfo], factory=list)
    inputs = attribute(of=DictOf[str, Extent], factory=dict)
    outputs = attribute(of=SetOf[str], factory=set)


@attribclass
class TransformData:
    """`AttribClass` class containing all the data structures used in the analysis pipeline.

    Parameters
    ----------
    definition_ir : `gridtools.ir.StencilDefinition`
        High-level IR with the definition of the stencil.
    implementation_ir : `gridtools.ir.StencilImplementation`
        Implementation IR with the final implementation of the stencil.
    options : `gt4py.definitions.Options`
        Build options provided by the users.
    splitters_var : `str`
        Used in IntervalMaker when parsing variable splitters.
    min_k_interval_sizes : `list` [`int`]
        Used in IntervalMaker for storing the interval sizes.
    symbols : `dict` [`str`, `SymbolInfo`]
        Symbols table.
    blocks : `list` [`DomainBlockInfo`]
        List of domain blocks.
    id_generator : `gt4py.utils.UniqueIdGenerator`
        Generates unique IDs.
    """

    definition_ir = attribute(of=gt_ir.StencilDefinition)
    implementation_ir = attribute(of=gt_ir.StencilImplementation)
    options = attribute(of=BuildOptions)

    splitters_var = attribute(of=str, optional=True)
    min_k_interval_sizes = attribute(of=ListOf[int], factory=list)
    symbols = attribute(of=DictOf[str, SymbolInfo], factory=dict)
    blocks = attribute(of=ListOf[DomainBlockInfo], factory=list)

    id_generator = attribute(
        of=gt_utils.UniqueIdGenerator, init=False, factory=gt_utils.UniqueIdGenerator
    )

    @property
    def ndims(self):
        return self.definition_ir.domain.ndims

    @property
    def nk_intervals(self):
        return len(self.min_k_interval_sizes)

    @property
    def axes_names(self):
        return self.definition_ir.domain.axes_names

    @property
    def sequential_axis(self):
        return str(CartesianSpace.Axis.K)

    @property
    def has_sequential_axis(self):
        return self.sequential_axis in self.axes_names


class TransformPass(abc.ABC):
    """Abstract base class defining the interface of an analysis pass."""

    @property
    def defaults(self):
        return {}

    @abc.abstractmethod
    def apply(self, transform_data: TransformData):
        """Run the transformation pass.

        Parameters
        ----------
        transform_data : `TransformData`
            Transformation data (modified in place).

        Returns
        -------
        transform_data : `TransformData`
            Transformation data (modified in place).
        """
        pass
