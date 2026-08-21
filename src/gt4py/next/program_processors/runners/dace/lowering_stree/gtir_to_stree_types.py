# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Data types used by the GTIR-to-ScheduleTree lowering.

This module is the schedule-tree analogue of ``lowering.gtir_to_sdfg_types``.
The key simplification relative to the SDFG lowering is that data is
referenced *by name* (a ``str``) rather than by a ``dace.nodes.AccessNode``
instance, because the schedule tree uses a single flat descriptor repository
at the root (``ScheduleTreeRoot.containers``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, TypeAlias

import dace
from dace import subsets as dace_subsets

from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next.type_system import type_specifications as ts


@dataclasses.dataclass(frozen=True)
class DataRef:
    """Reference (by name) to a data container on the ``ScheduleTreeRoot``.

    A ``DataRef`` couples the name under which a value is stored in the
    schedule-tree context with the GT4Py knowledge needed to index it: the
    GT4Py type (which carries the field domain) and the per-dimension origin.
    Note this class is *not* the data itself — the actual container
    descriptors live in ``root.containers`` (or ``root.symbols``) and the
    schedule tree creates the access nodes on demand.  In the schedule-tree
    lowering, data is referenced by name (``str``) rather than by a
    ``dace.nodes.AccessNode`` instance, because the tree uses a single flat
    descriptor repository at the root (``ScheduleTreeRoot.containers``).

    Attributes:
        name: Name of the data container (or SDFG symbol) on ``root.containers``
            or ``root.symbols``.
        gt_type: The GT4Py data type: a ``FieldType``, ``ScalarType``, or
            ``ListType`` (for neighbor-list values).
        origin: The per-dimension origin of the field.  For scalar data this
            is an empty tuple.
    """

    name: str
    gt_type: ts.FieldType | ts.ScalarType | ts.ListType
    origin: tuple[dace.symbolic.SymbolicType, ...]


DataRefTree: TypeAlias = MaybeNestedInTuple[DataRef | None]
"""A tree of ``DataRef`` leaves, mirroring the type of the lowered GTIR expression.

A scalar, field or local-list expression lowers to a single ``DataRef``; a
tuple-typed expression lowers to a (nested) tuple of these. A leaf may be
``None`` when the field is not used and therefore is not computed (per the
domain information attached to GTIR nodes by domain inference).
"""


@dataclasses.dataclass(frozen=True)
class SymbolicData:
    """A symbolic expression passed as an argument to a lambda expression.

    When a scalar argument's dependencies are all SDFG symbols, the argument
    can be passed as a symbolic expression rather than as a data container.
    In the schedule-tree lowering (which has no nested SDFGs), this is less
    common than in the SDFG lowering, but still useful for let-lambda
    arguments that are pure symbolic expressions.
    """

    gt_type: ts.ScalarType
    value: dace.symbolic.SymbolicType


@dataclasses.dataclass(frozen=True)
class SubgraphContext:
    """The context in which a GTIR expression is lowered to the schedule tree.

    The schedule-tree analogue of ``lowering.gtir_to_sdfg_fieldview.SubgraphContext``.
    Instead of carrying an ``SDFG`` and an ``SDFGState``, it carries the
    ``ScheduleTreeRoot`` (for storage allocation and the flat descriptor
    repository) and the ``current_scope`` (the tree node to which children
    are appended).

    The ``data_nodes`` dict plays the same role as in the
    ``dace_lowering_without_nesting`` branch: it maps GTIR symbol names to
    their ``DataRef`` handles, and let-lambdas are inlined by creating
    a child context with ``data_nodes | args`` (dict-union = lexical
    shadowing, Python call stack = implicit push/pop).
    """

    root: dace.sdfg.analysis.schedule_tree.treenodes.ScheduleTreeRoot
    current_scope: dace.sdfg.analysis.schedule_tree.treenodes.ScheduleTreeScope
    data_nodes: dict[str, DataRef | None]

    def get_symbol_type(self, symbol_name: str) -> ts.DataType:
        """Retrieve the GT4Py type of a symbol in the current scope."""
        data = self.data_nodes[symbol_name]
        assert data is not None
        return data.gt_type

    def get_input_data(self) -> set[str]:
        """Retrieve the names of arrays and scalars which are input data.

        Only non-transient data containers that are currently in scope are
        considered input data.
        """
        flat_symbols = []
        for _sym_name, data in self.data_nodes.items():
            if data is None:
                continue
            if isinstance(data.gt_type, ts.TupleType):
                # TODO(edopao): handle tuple types
                raise NotImplementedError("Tuple types not yet supported in stree lowering.")
            else:
                flat_symbols.append(data.name)
        input_data = {name for name in flat_symbols if name in self.root.containers}
        return {name for name in input_data if not self.root.containers[name].transient}

    def copy_data(
        self,
        sdfg_builder: Any,
        src: DataRef,
        domain: Any = None,
    ) -> DataRef:
        """Copy data from a source field into a new transient buffer.

        This is the schedule-tree analogue of
        ``lowering.gtir_to_sdfg_fieldview.SubgraphContext.copy_data``.
        Instead of creating SDFG edges, it creates a ``CopyNode`` and
        appends it to ``current_scope``.
        """
        from dace.sdfg.analysis.schedule_tree import treenodes as tn

        from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_utils import (
            get_field_layout,
        )

        src_desc = self.root.containers[src.name]
        if isinstance(src.gt_type, ts.FieldType):
            if domain is None:
                out_name, out_desc = sdfg_builder.add_temp_array_like(self.root, src_desc)
                out_origin = list(src.origin)
                src_subset = ",".join(f"0:{size}" for size in src_desc.shape)
            else:
                out_dims, out_origin, out_shape = get_field_layout(domain)
                assert out_dims == src.gt_type.dims
                out_name, out_desc = sdfg_builder.add_temp_array(
                    self.root, out_shape, src_desc.dtype
                )
                src_subset = ",".join(
                    f"{dst_o - src_o}:{dst_o - src_o + size}"
                    for dst_o, src_o, size in zip(out_origin, src.origin, out_shape, strict=True)
                )
        else:
            assert domain is None
            assert isinstance(src_desc, dace.data.Scalar)
            out_name, out_desc = sdfg_builder.add_temp_array_like(self.root, src_desc)
            out_origin = []
            src_subset = "0"

        copy_node = tn.CopyNode(
            target=out_name,
            memlet=dace.Memlet(
                data=src.name,
                subset=dace_subsets.Range.from_string(src_subset),
                other_subset=dace.subsets.Range.from_array(out_desc),
            ),
        )
        self.current_scope.add_child(copy_node)
        return DataRef(out_name, src.gt_type, tuple(out_origin))
