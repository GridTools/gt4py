# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contains the visitor to lower GTIR to a DaCe Schedule Tree.

This module is the schedule-tree analogue of
``lowering.gtir_to_sdfg_fieldview``.  Instead of building an SDFG directly,
it builds a ``ScheduleTreeRoot`` that can be converted to an SDFG via
``ScheduleTreeRoot.as_sdfg()``.

Key design decisions (confirmed in the grilling session):

- **Direct lowering** (no intermediate Next-TreeIR) in a new ``lowering_stree``
  module alongside ``lowering``.
- **Two-layer split**: ``GTIRToScheduleTree`` (fieldview — statements,
  storage, let-lambda, dispatch) + ``StreePythonCodegen`` (iterator —
  stencil body as one Python code string inside a ``TaskletNode``).
- **Pre-allocate** all program-level storage onto ``root.containers`` and
  ``root.symbols`` before visiting statements.
- **Let-lambdas inlined** into the same ``ScheduleTreeRoot`` via
  ``data_nodes | args`` (dict-union = lexical shadowing, Python call stack
  = implicit push/pop).  No nested SDFGs.
- **No library nodes**: reductions lowered directly to Python code inside
  the tasklet.
- **Scan** as a ``ForScope`` inside a ``MapScope``; the scalar carry
  transient is initialized with ``init`` before the loop and read/written
  each iteration.  List-typed leaves keep a ``carry if k > start else init``
  ternary because their carry is read from the result array at the previous
  index, which is out of bounds on the first iteration.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import dace
from dace import nodes as dace_nodes, subsets as dace_subsets
from dace.frontend.python import astutils as dace_astutils
from dace.sdfg import state as dace_state
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import builtins as itir_builtins, ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import (
    infer_domain,
    inline_lambdas,
    prune_casts as ir_prune_casts,
)
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_codegen import (
    generate_list_tasklet_code,
    generate_tasklet_code,
)
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_types import (
    DataRef,
    DataRefTree,
    SubgraphContext,
)
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_utils import (
    _CONST_DIM,
    FieldopDomain,
    extract_target_domain,
    flatten_tuple_fields,
    get_field_domain,
    get_field_layout,
    get_map_variable,
    get_source,
    get_symbolic,
    make_symbol_tree,
    make_tasklet_connector_for,
    replace_invalid_symbols,
)
from gt4py.next.type_system import (
    type_info as ti,
    type_specifications as ts,
    type_translation as tt,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _root_symbol_names(ctx: SubgraphContext) -> frozenset[str]:
    """Names of SDFG-level symbols and data containers (codegen ``SymRef`` fallback)."""
    return frozenset(ctx.root.symbols) | frozenset(ctx.root.containers)


def _replace_connectors_in_code_string(
    code: str, language: dace.dtypes.Language, connector_mapping: Mapping[str, str]
) -> str:
    """Replace connector names in the code of a Python tasklet."""
    code_block = dace.properties.CodeBlock(code, language)
    transformed_code_stmts = [
        dace_astutils.ASTFindReplace(connector_mapping).visit(stmt) for stmt in code_block.code
    ]
    return dace.properties.CodeBlock(transformed_code_stmts, language).as_string


def _flatten_tuple_symbols(symbols: Iterable[gtir.Sym]) -> list[gtir.Sym]:
    """Flatten tuple symbols, recursively in case of nested tuples."""
    flat_symbols: list[gtir.Sym] = []
    for sym in symbols:
        if isinstance(sym.type, ts.TupleType):
            flat_symbols.extend(f for f in flatten_tuple_fields(sym.id, sym.type))
        else:
            flat_symbols.append(sym)
    return flat_symbols


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class DataflowBuilder(Protocol):
    """Visitor interface to build schedule-tree dataflow nodes."""

    @abc.abstractmethod
    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem: ...

    @abc.abstractmethod
    def unique_nsdfg_name(self, prefix: str) -> str: ...

    @abc.abstractmethod
    def unique_map_name(self, name: str) -> str: ...

    @abc.abstractmethod
    def unique_tasklet_name(self, name: str) -> str: ...

    @abc.abstractmethod
    def unique_temp_name(self) -> str: ...

    @abc.abstractmethod
    def unique_lib_node_name(self, lib_node_type: str) -> str: ...

    def add_temp_array(
        self, root: tn.ScheduleTreeRoot, shape: Sequence[Any], dtype: dace.dtypes.typeclass
    ) -> tuple[str, dace.data.Array]:
        """Add a temporary array to the root's containers."""
        temp_name = self.unique_temp_name()
        array = dace.data.Array(dtype, shape, transient=True)
        root.containers[temp_name] = array
        # Ensure free symbols in shape are on root.symbols
        _collect_free_symbols(shape, root)
        return temp_name, array

    def add_temp_array_like(
        self, root: tn.ScheduleTreeRoot, datadesc: dace.data.Data
    ) -> tuple[str, dace.data.Array]:
        """Add a temporary array with the same descriptor as the given data."""
        temp_name = self.unique_temp_name()
        array = datadesc.clone()
        array.transient = True
        root.containers[temp_name] = array
        # Ensure free symbols in shape are on root.symbols
        _collect_free_symbols(array.shape, root)
        return temp_name, array

    def add_temp_scalar(
        self, root: tn.ScheduleTreeRoot, dtype: dace.dtypes.typeclass
    ) -> tuple[str, dace.data.Scalar]:
        """Add a temporary scalar to the root's containers."""
        temp_name = self.unique_temp_name()
        scalar = dace.data.Scalar(dtype, transient=True)
        root.containers[temp_name] = scalar
        return temp_name, scalar

    def add_map(
        self,
        name: str,
        ndrange: Union[
            Dict[str, Union[str, dace.subsets.Subset]],
            List[Tuple[str, Union[str, dace.subsets.Subset]]],
        ],
        **kwargs: Any,
    ) -> Tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
        """Create a free-standing ``MapEntry``/``MapExit`` pair (not in any state)."""
        unique_name = self.unique_map_name(name)
        params, map_range = dace_state._make_iterators(ndrange)
        map_obj = dace.nodes.Map(unique_name, params, map_range, **kwargs)
        map_entry = dace.nodes.MapEntry(map_obj)
        map_exit = dace.nodes.MapExit(map_obj)
        return map_entry, map_exit

    def add_tasklet(
        self,
        name: str,
        inputs: set[str] | Mapping[str, dace.dtypes.typeclass | None],
        outputs: set[str] | Mapping[str, dace.dtypes.typeclass | None],
        code: str,
        language: dace.dtypes.Language = dace.dtypes.Language.Python,
        **kwargs: Any,
    ) -> tuple[dace_nodes.Tasklet, dict[str, str]]:
        """Create a free-standing ``Tasklet`` (not in any state).

        Also modifies the tasklet connectors by adding a prefix string (see
        ``gtir_to_sdfg_utils.get_tasklet_connector()``), in order to avoid name
        conflicts with SDFG data.

        Returns:
            The created tasklet node and the mapping from original connector
            names to modified connector names.
        """
        if isinstance(inputs, set):
            inputs = {k: None for k in sorted(inputs)}
        if isinstance(outputs, set):
            outputs = {k: None for k in sorted(outputs)}
        assert inputs.keys().isdisjoint(outputs.keys())

        connector_mapping = {
            conn: make_tasklet_connector_for(conn) for conn in (inputs.keys() | outputs.keys())
        }
        new_code = _replace_connectors_in_code_string(code, language, connector_mapping)

        inputs = {connector_mapping[k]: v for k, v in inputs.items()}
        outputs = {connector_mapping[k]: v for k, v in outputs.items()}
        unique_name = self.unique_tasklet_name(name)
        tasklet = dace.nodes.Tasklet(
            unique_name, inputs=inputs, outputs=outputs, code=new_code, language=language, **kwargs
        )
        return tasklet, connector_mapping


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def make_field(
        self,
        name: str,
        data_type: ts.FieldType,
    ) -> DataRef: ...

    @abc.abstractmethod
    def is_column_axis(self, dim: gtx_common.Dimension) -> bool: ...

    @abc.abstractmethod
    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Field operator translation (stree-level primitives)
# ---------------------------------------------------------------------------


def _collect_free_symbols(shape_or_strides: Any, root: tn.ScheduleTreeRoot) -> None:
    """Ensure all free symbols in shape/stride expressions are on root.symbols.

    Free symbols in array shapes/strides are ``dace.symbol`` objects (a
    ``sympy.Symbol`` subclass).  ``SDFG.symbols`` expects ``typeclass``
    values, so we store the field-symbol dtype (``int64``) for each.
    """
    if not isinstance(shape_or_strides, (list, tuple)):
        shape_or_strides = [shape_or_strides]
    for expr in shape_or_strides:
        if hasattr(expr, "free_symbols"):
            for sym in expr.free_symbols:
                if hasattr(sym, "name") and sym.name not in root.symbols:
                    root.symbols[sym.name] = gtx_dace_args.FIELD_SYMBOL_DTYPE


def _make_access_index_for_field(domain: FieldopDomain, data: DataRef) -> dace.subsets.Range:
    """Build a memlet subset of a field over the given domain."""
    if isinstance(data.gt_type, ts.FieldType) and len(data.gt_type.dims) != 0:
        assert data.origin is not None
        domain_ranges = {
            domain_range.dim: (domain_range.start, domain_range.stop) for domain_range in domain
        }
        return dace.subsets.Range(
            (domain_ranges[dim][0] - origin, domain_ranges[dim][1] - origin - 1, 1)
            for dim, origin in zip(data.gt_type.dims, data.origin, strict=True)
        )
    else:
        assert len(domain) == 0
        return dace.subsets.Range.from_string("0")


def _make_full_array_memlet(name: str, root: tn.ScheduleTreeRoot) -> dace.Memlet:
    """Create a memlet that passes the full array to a tasklet."""
    desc = root.containers[name]
    return dace.Memlet(data=name, subset=dace.subsets.Range.from_array(desc))


def _get_scalar_field_dims(
    data_args: Mapping[str, DataRef], ctx: SubgraphContext
) -> dict[str, frozenset[gtx_common.Dimension]]:
    """Compute the scalar (size-one) dimensions of each field in ``data_args``.

    DaCe requires the subscript dimensionality of a tasklet connector to
    match the number of non-scalar (non-singleton) dimensions of the memlet
    subset (see ``dace.sdfg.tasklet_validation``), so the code generator
    squeezes container dimensions whose extent is statically one out of the
    subscripts it emits (see ``scalar_field_dims`` in
    ``gtir_to_stree_codegen.CodegenContext``).
    """
    result: dict[str, frozenset[gtx_common.Dimension]] = {}
    for data in data_args.values():
        if not isinstance(data.gt_type, ts.FieldType) or data.name in result:
            continue
        desc = ctx.root.containers.get(data.name, None)
        if not isinstance(desc, dace.data.Array):
            continue
        dims = data.gt_type.dims
        if len(dims) != len(desc.shape):
            continue
        squeezed = frozenset(dim for dim, size in zip(dims, desc.shape) if size == 1)
        if squeezed:
            result[data.name] = squeezed
    return result


def translate_as_fieldop(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for the ``as_fieldop`` builtin.

    The stencil expression is lowered to a single Python code string
    (via ``StreePythonCodegen``) and embedded in a ``TaskletNode`` inside
    a ``MapScope`` that ranges over the field domain.
    """
    assert cpm.is_applied_as_fieldop(node)
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    fieldop_expr, fieldop_domain_expr = fun_node.args

    if cpm.is_call_to(fieldop_expr, "scan"):
        return translate_scan(node, ctx, sdfg_builder)

    if not isinstance(node.type, ts.FieldType):
        assert isinstance(node.type, ts.TupleType)
        if not isinstance(fieldop_expr, gtir.Lambda):
            raise NotImplementedError(
                "'as_fieldop' with tuple output and non-lambda stencil expression."
            )
        if any(isinstance(el_type, ts.TupleType) for el_type in node.type.types):
            raise NotImplementedError("Nested tuple output of 'as_fieldop' not supported.")
        # Lower each tuple element independently: extract the element from
        # the lambda body with `tuple_get` and translate it as a separate
        # single-output field operator.
        results = []
        for index, element_type in enumerate(node.type.types):
            element_expr = im.tuple_get(index, fieldop_expr.expr)
            element_expr.type = element_type
            element_lambda = im.lambda_(*fieldop_expr.params)(element_expr)
            element_call = im.as_fieldop(element_lambda, fieldop_domain_expr)(*node.args)
            element_call.type = element_type
            results.append(translate_as_fieldop(element_call, ctx, sdfg_builder))
        return tuple(results)

    # Parse the domain of the field operator.
    assert isinstance(fieldop_domain_expr.type, ts.DomainType)
    field_domain = get_field_domain(domain_utils.SymbolicDomain.from_expr(fieldop_domain_expr))

    # Handle special case: deref with field argument (copy)
    if cpm.is_ref_to(fieldop_expr, "deref"):
        arg_type = node.args[0].type
        assert isinstance(arg_type, (ts.FieldType, ts.ScalarType))
        if isinstance(arg_type, ts.ScalarType) or arg_type.dims != node.type.dims:
            stencil_expr = im.lambda_("a")(im.deref("a"))
            stencil_expr.expr.type = node.type.dtype
        else:
            arg = sdfg_builder.visit(node.args[0], ctx=ctx)
            assert isinstance(arg, DataRef)
            return ctx.copy_data(sdfg_builder, arg, domain=field_domain)
    elif isinstance(fieldop_expr, gtir.Lambda):
        stencil_expr = fieldop_expr
    else:
        raise NotImplementedError(
            f"Expression type '{type(fieldop_expr)}' not supported as argument to 'as_fieldop' node."
        )

    # Visit the arguments to be passed to the lambda expression.
    # Each argument is visited to obtain a DataRef handle.
    data_args: dict[str, DataRef] = {}
    for i, arg_expr in enumerate(node.args):
        arg = sdfg_builder.visit(arg_expr, ctx=ctx)
        if isinstance(arg, DataRef):
            param_name = (
                str(stencil_expr.params[i].id) if i < len(stencil_expr.params) else f"__arg{i}"
            )
            data_args[param_name] = arg

    # Build the map_indices mapping (Dimension -> map variable name).  Only
    # the field-domain (global) dimensions; when the result field has a
    # ``ListType`` dtype, the local dimension gets its own map variable,
    # handled elementwise in the tasklet (see below).
    map_indices: dict[gtx_common.Dimension, str] = {}
    for r in field_domain:
        map_indices[r.dim] = get_map_variable(r.dim)

    field_dims, field_origin, field_shape = get_field_layout(field_domain)

    offset_provider_type = (
        sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
        if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
        else ctx.root._offset_provider_type
    )
    args_map: dict[str, gtir.Node] = {
        str(p.id): node.args[i] for i, p in enumerate(stencil_expr.params) if i < len(node.args)
    }
    output_subset_parts = [
        f"{map_indices[dim]} - ({origin})"
        for dim, origin in zip(field_dims, field_origin, strict=True)
    ]

    if isinstance(node.type.dtype, ts.ListType) and node.type.dtype.offset_type not in (
        None,
        _CONST_DIM,
    ):
        # Field operator computing a field with a local (neighbor) dimension,
        # e.g. ``as_fieldop(λ(it) → neighbors(V2E, it), V_domain)(edge_f)``.
        # Lower it elementwise: the map scope ranges over the field domain
        # extended with the local dimension, and the tasklet writes one
        # element of the local list (with the skip-value guard).  This mirrors
        # ``_visit_neighbors`` in the SDFG iterator lowering.
        list_dtype = node.type.dtype
        assert list_dtype.offset_type is not None
        local_dim = list_dtype.offset_type
        conn_type = offset_provider_type.get(local_dim.value)
        if not isinstance(conn_type, gtx_common.NeighborConnectivityType):
            raise NotImplementedError(
                f"'as_fieldop' with 'ListType' dtype over offset '{local_dim.value}' "
                "is not supported."
            )
        local_var = get_map_variable(local_dim)

        # Build the map over the field domain extended with the local dim.
        map_range = (
            {"__gt4py_zerodim": "0"}
            if len(field_domain) == 0
            else {get_map_variable(r.dim): f"{r.start}:{r.stop}" for r in field_domain}
        )
        map_range[local_var] = f"0:{conn_type.max_neighbors}"
        map_entry, _map_exit = sdfg_builder.add_map("fieldop", map_range)

        # Generate the elementwise tasklet code: one element of the local
        # dimension, with the skip-value guard.
        expr_code, pre_statements, used_connectivities, field_inputs = generate_list_tasklet_code(
            stencil_expr.expr,
            data_args=data_args,
            map_indices=map_indices,
            offset_provider_type=offset_provider_type,
            args_map=args_map,
            local_index=local_var,
            root_symbols=_root_symbol_names(ctx),
            scalar_field_dims=_get_scalar_field_dims(data_args, ctx),
        )

        # Build the full tasklet code: pre-statements + final assignment.
        tasklet_code = (
            "\n".join(pre_statements) + f"\n__out = {expr_code}"
            if pre_statements
            else f"__out = {expr_code}"
        )

        # Allocate the result array: the domain shape extended with the
        # local dimension at its canonical position (local origin is 0).
        extended_dims = gtx_common.order_dimensions([*field_dims, local_dim])
        local_dim_index = extended_dims.index(local_dim)
        field_shape.insert(local_dim_index, conn_type.max_neighbors)
        assert isinstance(list_dtype.element_type, ts.ScalarType)
        result_name, _ = sdfg_builder.add_temp_array(
            ctx.root, field_shape, gtx_dace_args.as_dace_type(list_dtype.element_type)
        )
        output_subset_parts.insert(local_dim_index, local_var)
    else:
        # Build the map over the field domain.
        if len(field_domain) == 0:
            map_range = {"__gt4py_zerodim": "0"}
        else:
            map_range = {get_map_variable(r.dim): f"{r.start}:{r.stop}" for r in field_domain}
        map_entry, _map_exit = sdfg_builder.add_map("fieldop", map_range)

        # Generate the tasklet code using StreePythonCodegen.
        expr_code, pre_statements, used_connectivities, field_inputs = generate_tasklet_code(
            stencil_expr.expr,
            data_args=data_args,
            map_indices=map_indices,
            offset_provider_type=offset_provider_type,
            args_map=args_map,
            root_symbols=_root_symbol_names(ctx),
            scalar_field_dims=_get_scalar_field_dims(data_args, ctx),
        )

        # Build the full tasklet code: pre-statements + final assignment.
        tasklet_code = (
            "\n".join(pre_statements) + f"\n__out = {expr_code}"
            if pre_statements
            else f"__out = {expr_code}"
        )

        # Allocate the result array.  A ``ListType`` dtype without a local
        # offset dimension is a broadcast (const) list — all elements are
        # equal, so the value is stored without the local dimension.
        output_dtype: ts.DataType = node.type.dtype
        if isinstance(output_dtype, ts.ListType):
            output_dtype = output_dtype.element_type
        assert isinstance(output_dtype, ts.ScalarType)
        if len(field_shape) == 0:
            # Zero-dimensional field: a scalar container, matching
            # ``_create_field_operator_data`` in the SDFG lowering.
            result_name, _ = sdfg_builder.add_temp_scalar(
                ctx.root, gtx_dace_args.as_dace_type(output_dtype)
            )
        else:
            result_name, _ = sdfg_builder.add_temp_array(
                ctx.root, field_shape, gtx_dace_args.as_dace_type(output_dtype)
            )

    # Connectivity tables referenced by ``neighbors`` in the stencil body must
    # be passed to the tasklet as inputs (full-array memlets); otherwise the
    # connectivity access inside the tasklet code would be a dangling
    # reference.  The connector names are the connectivity identifiers, which
    # ``add_tasklet`` prefixes and replaces in the code string.  In addition,
    # the connectivity arrays (allocated transient by ``_add_sdfg_params``) are
    # marked non-transient here so that (a) the caller provides them as SDFG
    # arguments and (b) the unused-connectivity cleanup in ``visit_Program``
    # — which only inspects top-level children and would miss tasklets nested
    # in ``MapScope`` — does not drop them.
    connectivity_inputs = {aname: None for aname in used_connectivities}

    # When a field is accessed through ``field_inputs`` (scalar memlet
    # on the input edge), the field parameter name does not appear in
    # the tasklet code (only the ``field_inputs`` connector name does).
    # In that case, skip the full-array memlet to avoid duplicate
    # memlets for the same data, which can cause codegen issues.
    _field_names_in_inputs = (
        {fname for fname, _ in field_inputs.items() if isinstance(fname, str)}
        if field_inputs
        else set()
    )
    # Only add a full-array connector if the field name appears in the
    # tasklet code (i.e. the field is accessed directly, not through
    # ``field_inputs``).
    import re

    _full_array_param_names = {
        param_name
        for param_name, data in data_args.items()
        if not field_inputs
        or data.name not in _field_names_in_inputs
        or re.search(r"\b" + re.escape(data.name) + r"\b", tasklet_code)
        or re.search(r"\b" + re.escape(param_name) + r"\b", tasklet_code)
    }

    # Create the Tasklet and TaskletNode.
    tasklet, _connector_mapping = sdfg_builder.add_tasklet(
        name="fieldop",
        inputs={inp: None for inp in _full_array_param_names}
        | connectivity_inputs
        | {inp: None for inp in field_inputs.keys()},
        outputs={"__out"},
        code=tasklet_code,
    )

    # Setup input memlets.
    in_memlets: dict[str, dace.Memlet] = {
        _connector_mapping[param_name]: _make_full_array_memlet(data.name, ctx.root)
        for param_name, data in data_args.items()
        if param_name in _full_array_param_names
    }
    for aname in used_connectivities:
        ctx.root.containers[aname].transient = False
        in_memlets[_connector_mapping[aname]] = _make_full_array_memlet(aname, ctx.root)
    # Scalar / 1D field accesses on the input edge: the memlet subset
    # encodes all symbolic dimensions as exact indices (and dynamic
    # dimensions as full ranges), so the tasklet receives a scalar or
    # 1D array rather than the full multi-dimensional array.
    for connector, (field_name, subset) in field_inputs.items():
        in_memlets[_connector_mapping[connector]] = dace.Memlet(
            data=field_name, subset=dace_subsets.Range.from_string(subset)
        )

    # Output memlet: single element at the map indices (plus the local
    # dimension map variable for fields with a ``ListType`` dtype).
    output_subset = ",".join(output_subset_parts) if output_subset_parts else "0"
    out_memlet = dace.Memlet(data=result_name, subset=output_subset)
    out_memlets = {_connector_mapping["__out"]: out_memlet}

    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets=in_memlets,
        out_memlets=out_memlets,
    )

    # Create the MapScope and add the TaskletNode as a child.
    map_scope = tn.MapScope(node=map_entry, children=[tasklet_node])
    ctx.current_scope.add_child(map_scope)

    return DataRef(result_name, node.type, tuple(field_origin))


def translate_map_list(
    node: gtir.FunCall,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRef:
    """Lower ``map_list(op)(args...)`` to a ``MapScope`` + ``TaskletNode``.

    The map iterates over the local (neighbor) dimension; each argument is
    accessed at the current map index inside the tasklet.  This mirrors
    ``_visit_map`` in ``lowering.gtir_to_sdfg_iterator``.

    Args:
        node: An applied ``map_list`` ``FunCall`` (``map_list(op)(args...)``).
        ctx: The subgraph context.
        sdfg_builder: The SDFG builder.

    Returns:
        A ``DataRef`` for the 1-D result array (``ListType``).
    """
    assert cpm.is_applied_map(node)
    assert isinstance(node.type, ts.ListType)
    assert isinstance(node.fun, gtir.FunCall) and len(node.fun.args) == 1
    map_op = node.fun.args[0]

    # Determine the local dimension and its size from the input arguments.
    # Skip ``make_const_list`` args — their offset type is ``_CONST_DIM``.
    local_offset_type: gtx_common.Dimension | None = None
    local_size: int | None = None
    has_skip_values = False
    for arg in node.args:
        arg_type = arg.type
        if isinstance(arg_type, ts.ListType) and arg_type.offset_type is not None:
            offset_type = arg_type.offset_type
            if offset_type == _CONST_DIM:
                continue
            offset_provider_t = sdfg_builder.get_offset_provider_type.__self__.offset_provider_type[  # type: ignore[attr-defined]
                offset_type.value
            ]
            assert isinstance(offset_provider_t, gtx_common.NeighborConnectivityType)
            local_offset_type = offset_type
            local_size = offset_provider_t.max_neighbors
            has_skip_values = offset_provider_t.has_skip_values
            break

    if local_offset_type is None or local_size is None:
        raise ValueError(f"Missing information on local dimension for map node {node}.")

    assert local_offset_type is not None
    map_var = get_map_variable(local_offset_type)

    # Visit the arguments to obtain DataRef handles.  Arguments that
    # don't lower to DataRef (e.g. ``make_const_list``) are passed
    # through ``args_map`` so the codegen can handle them.
    data_args: dict[str, DataRef] = {}
    args_map: dict[str, gtir.Node] = {}
    for i, arg_expr in enumerate(node.args):
        param_name = f"__arg{i}"
        arg = sdfg_builder.visit(arg_expr, ctx=ctx)
        if isinstance(arg, DataRef):
            data_args[param_name] = arg
        else:
            args_map[param_name] = arg_expr

    # Build the expression: ``map_op(__arg0, __arg1, ...)``.
    map_expr = im.call(map_op)(*[im.ref(p) for p in [*(data_args.keys()), *(args_map.keys())]])

    # Generate the tasklet code.  The map variable acts as the loop variable
    # for element access (neighbors indexing, local field indexing, etc.).
    map_indices = {local_offset_type: map_var}
    expr_code, pre_statements, used_connectivities, field_inputs = generate_tasklet_code(
        map_expr,
        data_args=data_args,
        map_indices=map_indices,
        offset_provider_type=sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
        if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
        else ctx.root._offset_provider_type,
        args_map=args_map,
        root_symbols=_root_symbol_names(ctx),
        scalar_field_dims=_get_scalar_field_dims(data_args, ctx),
    )

    # Build the mapped tasklet expression.  When the connectivity has skip
    # values, the tasklet writes a dummy value for invalid neighbors (matching
    # the SDFG lowering's `map` with `__neighbor_idx` masking).
    if has_skip_values:
        conn_name = gtx_dace_args.connectivity_identifier(local_offset_type.value)
        ctx.root.containers[conn_name].transient = False
        used_connectivities.add(conn_name)
        element_type = node.type.element_type
        assert isinstance(element_type, ts.ScalarType)
        skip_value = (
            "math.nan"
            if ti.is_floating_point(element_type)
            else str(dace.dtypes.max_value(gtx_dace_args.as_dace_type(element_type)))
        )
        expr_code = (
            f"({expr_code}) if {map_var} != {gtx_common._DEFAULT_SKIP_VALUE} else {skip_value}"
        )

    tasklet_code = (
        "\n".join(pre_statements) + f"\n__out = {expr_code}"
        if pre_statements
        else f"__out = {expr_code}"
    )

    # Allocate the result array (1-D over the local dimension).
    element_type = node.type.element_type
    assert isinstance(element_type, ts.ScalarType)
    dc_dtype = gtx_dace_args.as_dace_type(element_type)
    result_name, _ = sdfg_builder.add_temp_array(ctx.root, (local_size,), dc_dtype)

    # Connectivity tables referenced by ``neighbors`` must be passed as
    # inputs (full-array memlets); see ``translate_as_fieldop`` for details.
    connectivity_inputs = {aname: None for aname in used_connectivities}

    # Determine which field args need a full-array memlet (vs. scalar memlet
    # through ``field_inputs``).
    import re

    _field_names_in_inputs = (
        {fname for fname, _ in field_inputs.items() if isinstance(fname, str)}
        if field_inputs
        else set()
    )
    _full_array_param_names = {
        param_name
        for param_name, data in data_args.items()
        if not field_inputs
        or data.name not in _field_names_in_inputs
        or re.search(r"\b" + re.escape(data.name) + r"\b", tasklet_code)
        or re.search(r"\b" + re.escape(param_name) + r"\b", tasklet_code)
    }

    # Create the Tasklet and TaskletNode.
    tasklet, _connector_mapping = sdfg_builder.add_tasklet(
        name="map",
        inputs={inp: None for inp in _full_array_param_names}
        | connectivity_inputs
        | {inp: None for inp in field_inputs.keys()},
        outputs={"__out"},
        code=tasklet_code,
    )

    # Setup input memlets.
    in_memlets: dict[str, dace.Memlet] = {
        _connector_mapping[param_name]: _make_full_array_memlet(data.name, ctx.root)
        for param_name, data in data_args.items()
        if param_name in _full_array_param_names
    }
    for aname in used_connectivities:
        ctx.root.containers[aname].transient = False
        in_memlets[_connector_mapping[aname]] = _make_full_array_memlet(aname, ctx.root)
    for connector, (field_name, subset) in field_inputs.items():
        in_memlets[_connector_mapping[connector]] = dace.Memlet(
            data=field_name, subset=dace_subsets.Range.from_string(subset)
        )

    # Output memlet: single element at the map index.
    out_memlet = dace.Memlet(data=result_name, subset=map_var)
    out_memlets = {_connector_mapping["__out"]: out_memlet}

    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets=in_memlets,
        out_memlets=out_memlets,
    )

    # Create the MapScope over the local dimension.
    map_entry, _map_exit = sdfg_builder.add_map("map", {map_var: f"0:{local_size}"})
    map_scope = tn.MapScope(node=map_entry, children=[tasklet_node])
    ctx.current_scope.add_child(map_scope)

    return DataRef(result_name, node.type, origin=())


def translate_scan(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for a scan field operator.

    The scan is lowered as a ``ForScope`` inside a ``MapScope``.  The
    horizontal domain is mapped, and the vertical (column) dimension is
    iterated by the ``ForScope``.  For scalar leaves the carry is stored in
    a scalar transient container that is initialized with the scan's
    ``init`` value before the loop and then written at the end of each
    iteration; it is read directly by the next iteration.  For list-typed
    leaves the carry is read from the result array at the previous index
    (``O[k-1] if k > start else A`` for forward scans), because the init
    value cannot be written into the result array without going out of
    bounds.
    """
    assert cpm.is_applied_as_fieldop(node)
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, scan_domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # Parse the domain of the scan field operator.
    assert isinstance(scan_domain_expr.type, ts.DomainType)
    field_domain = get_field_domain(domain_utils.SymbolicDomain.from_expr(scan_domain_expr))

    # Parse scan parameters.
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # scan args[1]: boolean flag for forward/backward scan.
    assert isinstance(scan_expr.args[1], gtir.Literal)
    scan_forward = scan_expr.args[1].value == "True"

    # scan args[2]: the expression that computes the value for scan initialization.
    init_expr = scan_expr.args[2]

    # The value produced by each scan step (and returned by the scanned
    # lambda) may be a — possibly nested — tuple of values; the scan output
    # is accordingly a (nested) tuple of fields.  The lowering flattens the
    # value type into scalar leaves: one result array and one carry connector
    # per leaf, with all leaves computed together in a single scan tasklet.
    # ``state``-accesses in the stencil body (``tuple_get`` chains on the
    # carry parameter) are resolved by the code generator, which routes each
    # leaf to its carry variable.
    def _scan_value_dtype(scan_type: ts.TypeSpec) -> ts.DataType:
        """Extract the per-position value dtype of a scan output field.

        Nested tuples of fields (e.g. a scan returning ``tuple[tuple[Field,
        Field], Field]``) are traversed recursively, mirroring the nesting.
        """
        if isinstance(scan_type, ts.TupleType):
            return ts.TupleType(types=[_scan_value_dtype(t) for t in scan_type.types])
        if not isinstance(scan_type, ts.FieldType):
            raise NotImplementedError(f"Unsupported scan output type: {scan_type}.")
        dtype = scan_type.dtype
        if not isinstance(dtype, ts.DataType):
            raise NotImplementedError(f"Unsupported scan output dtype: {dtype}.")
        return dtype

    dtype_tree: ts.DataType = _scan_value_dtype(node.type)

    def _flatten_leaves(
        dtype: ts.DataType, path: tuple[int, ...] = ()
    ) -> list[tuple[tuple[int, ...], ts.ScalarType | ts.ListType]]:
        if isinstance(dtype, ts.TupleType):
            flattened: list[tuple[tuple[int, ...], ts.ScalarType | ts.ListType]] = []
            for i, subtype in enumerate(dtype.types):
                if not isinstance(subtype, ts.DataType):
                    raise NotImplementedError(f"Unsupported scan output type: {subtype}.")
                flattened.extend(_flatten_leaves(subtype, (*path, i)))
            return flattened
        if not isinstance(dtype, (ts.ScalarType, ts.ListType)):
            raise NotImplementedError(f"Unsupported scan output type: {dtype}.")
        return [(path, dtype)]

    leaves = _flatten_leaves(dtype_tree)

    def _index_path(expr: gtir.Expr, path: tuple[int, ...]) -> gtir.Expr:
        expr_ = expr
        for index in path:
            expr_ = im.tuple_get(index, expr_)
        return expr_

    # Compute the init data (name of a scalar container or SDFG symbol) per
    # leaf.  NOTE: this must happen *before* the horizontal MapScope is
    # created and added to the tree, so that producer nodes appear in the
    # tree before their consumer scope — DaCe's `as_sdfg()` conversion drops
    # a top-level node placed after the scope that consumes its data.
    init_leaves_data = []
    for path, _leaf_dtype in leaves:
        init_data = sdfg_builder.visit(_index_path(init_expr, path), ctx=ctx)
        assert isinstance(init_data, DataRef)
        init_leaves_data.append(init_data)

    # Visit the field arguments (excluding the carry, which is handled
    # specially as the output array at the previous index).  Same tree
    # ordering constraint as above.
    data_args: dict[str, DataRef] = {}
    for i, arg_expr in enumerate(node.args):
        arg = sdfg_builder.visit(arg_expr, ctx=ctx)
        if isinstance(arg, DataRef):
            param_name = (
                str(stencil_expr.params[i + 1].id)
                if i + 1 < len(stencil_expr.params)
                else f"__arg{i}"
            )
            data_args[param_name] = arg

    # Find the column dimension.
    scan_dim_index = [sdfg_builder.is_column_axis(r.dim) for r in field_domain].index(True)
    scan_domain_range = field_domain[scan_dim_index]
    scan_dim = scan_domain_range.dim
    scan_loop_var = get_map_variable(scan_dim)

    # Determine the horizontal dimensions (all except the column axis).
    horizontal_dims = [r for r in field_domain if not sdfg_builder.is_column_axis(r.dim)]

    # Allocate one result array per leaf (on the full scan domain).
    field_dims, field_origin, field_shape = get_field_layout(field_domain)

    # Shape of the per-column scan carry for scalar leaves: one slot per
    # horizontal map point, without the scan dimension.  When the scan runs
    # inside a horizontal ``MapScope`` every column must keep its own carry;
    # a single shared scalar (as used previously) races across columns and
    # produces nondeterministic results.  For 1D scans (no horizontal map)
    # ``horizontal_carry_shape`` is empty and the carry stays a scalar.
    horizontal_carry_shape = [field_shape[i] for i, dim in enumerate(field_dims) if dim != scan_dim]

    result_names = []
    # For scalar leaves, the scan carry is stored in a scalar transient container,
    # written at the end of each scan iteration and read (as a whole scalar) at
    # the beginning of the next one.  This mirrors the lowering of the legacy
    # fieldview/iview scan (``gtir_to_sdfg_scan``) and avoids reading the result
    # array at the previous scan index — such an in-edge would statically extend
    # one element past the allocated scan domain, producing an out-of-bounds
    # memlet on the aggregated scope edges once the scan domain is static.
    carry_container_names: list[str | None] = []
    for _path, leaf_dtype in leaves:
        if isinstance(leaf_dtype, ts.ListType):
            assert isinstance(leaf_dtype.element_type, ts.ScalarType)
            dtype = gtx_dace_args.as_dace_type(leaf_dtype.element_type)
        else:
            dtype = gtx_dace_args.as_dace_type(leaf_dtype)
        result_name, _result_desc = sdfg_builder.add_temp_array(ctx.root, field_shape, dtype)
        result_names.append(result_name)
        if isinstance(leaf_dtype, ts.ScalarType):
            if horizontal_carry_shape:
                carry_name, _carry_desc = sdfg_builder.add_temp_array(
                    ctx.root, horizontal_carry_shape, dtype
                )
            else:
                carry_name, _carry_desc = sdfg_builder.add_temp_scalar(ctx.root, dtype)
            carry_container_names.append(carry_name)
        else:
            carry_container_names.append(None)

    # Free SDFG symbols can be used directly in the tasklet code; scalar
    # containers are read through an additional scalar input edge per leaf.
    init_codes: list[str] = []
    init_connectors: dict[str, None] = {}
    for i, init_data in enumerate(init_leaves_data):
        if init_data.name in ctx.root.symbols:
            init_codes.append(init_data.name)
        else:
            init_code = f"__init_{i}"
            init_codes.append(init_code)
            init_connectors[init_code] = None

    # Build the MapScope over horizontal dimensions.
    if len(horizontal_dims) == 0:
        # 1D scan: no horizontal map needed.
        map_scope = None
        map_indices: dict[gtx_common.Dimension, str] = {}
    else:
        map_range = {get_map_variable(r.dim): f"{r.start}:{r.stop}" for r in horizontal_dims}
        map_entry, _map_exit = sdfg_builder.add_map("fieldop", map_range)
        map_indices = {r.dim: get_map_variable(r.dim) for r in horizontal_dims}
        map_scope = tn.MapScope(node=map_entry, children=[])
        ctx.current_scope.add_child(map_scope)

    # Horizontal subset (no scan dimension) used to index the per-column
    # scalar carry containers (init before the loop, read/write each
    # iteration); mirrors the horizontal part of ``carry_subset``.  Empty for
    # 1D scans, where the carry is a single scalar accessed at ``"0"``.
    horizontal_carry_subset_parts = [
        f"({map_indices[dim]}) - ({origin})"
        for dim, origin in zip(field_dims, field_origin, strict=True)
        if dim != scan_dim
    ]
    horizontal_carry_subset = (
        dace_subsets.Range.from_string(",".join(horizontal_carry_subset_parts))
        if horizontal_carry_subset_parts
        else "0"
    )

    # Initialize the carry container of each scalar leaf with the scan's
    # ``init`` value before entering the scan loop.  Without this, the first
    # scan iteration reads the carry container before it is ever written;
    # the value is discarded by a ``carry if k > start else init`` ternary
    # inside the tasklet, but the read itself is genuine uninitialized
    # memory (flagged by DaCe's SDFG validator on every stree scan compile).
    # Pre-initializing the carry lets the scan tasklet read it directly and
    # makes the ternary (and its ``init`` input connector) redundant for
    # scalar leaves.  List-typed leaves keep the ternary because their carry
    # is read from the result array at the previous scan index, which would
    # be out of bounds on the first iteration.
    scalar_carry_init_lines = [
        f"__carry_w_{i} = {init_codes[i]}"
        for i in range(len(leaves))
        if carry_container_names[i] is not None
    ]
    if scalar_carry_init_lines:
        scalar_init_connectors = {
            init_codes[i]: None
            for i in range(len(leaves))
            if carry_container_names[i] is not None and init_codes[i] in init_connectors
        }
        carry_init_tasklet, carry_init_mapping = sdfg_builder.add_tasklet(
            name="scan_carry_init",
            inputs=scalar_init_connectors,
            outputs={
                f"__carry_w_{i}" for i in range(len(leaves)) if carry_container_names[i] is not None
            },
            code="\n".join(scalar_carry_init_lines),
        )
        carry_init_in_memlets: dict[str, dace.Memlet] = {}
        for i in range(len(leaves)):
            if carry_container_names[i] is None:
                continue
            init_code = init_codes[i]
            if init_code in scalar_init_connectors:
                carry_init_in_memlets[carry_init_mapping[init_code]] = dace.Memlet(
                    data=init_leaves_data[i].name, subset="0"
                )
        carry_init_out_memlets: dict[str, dace.Memlet] = {}
        for i in range(len(leaves)):
            if carry_container_names[i] is not None:
                # Write the init value into the current map point's slot of the
                # per-column carry (not the shared "0" slot): the init tasklet
                # runs inside the horizontal MapScope, so each column must
                # initialize its own carry element.  1D scans have a scalar
                # carry accessed at "0".
                carry_init_out_memlets[carry_init_mapping[f"__carry_w_{i}"]] = dace.Memlet(
                    data=carry_container_names[i],
                    subset=copy.deepcopy(horizontal_carry_subset)
                    if horizontal_carry_subset_parts
                    else "0",
                )
        carry_init_node = tn.TaskletNode(
            node=carry_init_tasklet,
            in_memlets=carry_init_in_memlets,
            out_memlets=carry_init_out_memlets,
        )
        if map_scope is not None:
            map_scope.add_child(carry_init_node)
        else:
            ctx.current_scope.add_child(carry_init_node)

    # Add the scan dimension to map_indices (used by the code generator
    # to generate the carry read and output write).
    map_indices[scan_dim] = scan_loop_var

    # Build the carry expression.
    # Forward carry: O[k-1] if k > start else A
    # Backward carry: O[k+1] if k < stop-1 else A
    if scan_forward:
        carry_prev = f"{scan_loop_var} - 1"
        carry_cond = f"{scan_loop_var} > {scan_domain_range.start}"
    else:
        carry_prev = f"{scan_loop_var} + 1"
        carry_cond = f"{scan_loop_var} < {scan_domain_range.stop} - 1"

    # On the input edge, each carry is read as a scalar (the leaf's result
    # array at the previous scan position along the column): `__carry_<i>` is
    # a connector with a scalar memlet on `result_name_i` at
    # `O[horizontal_indices, k-1]` (or `k+1` for backward scans).  Inside the
    # tasklet, a pre-statement selects between the carry and the scan init
    # value; the lambda carry parameter (``params[0]``) resolves to a nested
    # `make_tuple` of per-leaf pre-statement variables via `args_map`, whose
    # `tuple_get` accesses are collapsed by the code generator.
    carry_conn_of = [f"__carry_{i}" for i in range(len(leaves))]
    carry_subset_parts = []
    for dim, origin in zip(field_dims, field_origin, strict=True):
        map_var = map_indices.get(dim)
        assert map_var is not None, f"No map variable for dimension {dim}."
        if dim == scan_dim:
            carry_subset_parts.append(f"({carry_prev}) - ({origin})")
        else:
            carry_subset_parts.append(f"({map_var}) - ({origin})")
    carry_subset = dace_subsets.Range.from_string(",".join(carry_subset_parts))

    carry_vars = [f"__gtir_scan_carry_{i}" for i in range(len(leaves))]
    # Scalar leaves: the carry container was initialized before the loop, so
    # the tasklet reads it directly.  List leaves: the carry is read from the
    # result array at the previous index, which is out of bounds on the first
    # iteration, so the ternary (selecting ``init`` on the first iteration)
    # is still needed.
    pre_statements_scan = [
        f"{carry_var} = {carry_conn}"
        if carry_container_names[i] is not None
        else f"{carry_var} = {carry_conn} if {carry_cond} else {init_code}"
        for i, (carry_var, carry_conn, init_code) in enumerate(
            zip(carry_vars, carry_conn_of, init_codes, strict=True)
        )
    ]

    # Set up the args_map for the lambda parameters: the carry parameter
    # (``params[0]``) maps to a nested `make_tuple` of the per-leaf carry
    # SymRefs (mirroring the scan value structure); ``params[1:]`` map to the
    # scan arguments.
    carry_leaf_iter = iter(carry_vars)

    def _build_carry_tree(dtype: ts.DataType) -> gtir.Expr:
        if isinstance(dtype, ts.TupleType):
            subtree_args = []
            for subtype in dtype.types:
                if not isinstance(subtype, ts.DataType):
                    raise NotImplementedError(f"Unsupported scan output type: {subtype}.")
                subtree_args.append(_build_carry_tree(subtype))
            return im.call("make_tuple")(*subtree_args)
        return gtir.SymRef(id=next(carry_leaf_iter))

    scan_args_map: dict[str, gtir.Node] = {
        str(stencil_expr.params[0].id): _build_carry_tree(dtype_tree)
    }
    scan_args_map |= {
        str(stencil_expr.params[i].id): node.args[i - 1]
        for i in range(1, len(stencil_expr.params))
        if i - 1 < len(node.args)
    }

    # The per-leaf carry SymRefs above resolve to the tasklet code variables
    # holding the carry-or-init selection.
    string_args = {carry_var: carry_var for carry_var in carry_vars}

    # Generate the tasklet code per leaf using StreePythonCodegen.  All
    # leaves share the temp counter, since their code (and the connector
    # names generated in the process) is combined into a single tasklet.
    temp_counter = itertools.count(1)
    expr_codes: list[str] = []
    pre_statements_body: list[str] = []
    used_connectivities: set[str] = set()
    field_inputs: dict[str, tuple[str, str]] = {}
    for path, _leaf_dtype in leaves:
        expr_code, pre_stmts, used_conns, tasklet_field_inputs = generate_tasklet_code(
            _index_path(stencil_expr.expr, path),
            data_args=data_args,
            map_indices=map_indices,
            offset_provider_type=sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
            if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
            else ctx.root._offset_provider_type,
            args_map=scan_args_map,
            string_args=string_args,
            temp_counter=temp_counter,
            root_symbols=_root_symbol_names(ctx),
            scalar_field_dims=_get_scalar_field_dims(data_args, ctx),
        )
        expr_codes.append(expr_code)
        pre_statements_body.extend(pre_stmts)
        used_connectivities |= set(used_conns)
        field_inputs |= tasklet_field_inputs

    # Combine all pre-statements and the final (per-leaf) expressions.
    out_conns = [f"__out_{i}" for i in range(len(leaves))]
    # For scalar leaves, the scan tasklet additionally writes the computed value
    # into the carry container, where it is read back by the next scan iteration.
    carry_write_conns = [
        f"__carry_w_{i}" if carry_container_names[i] is not None else None
        for i in range(len(leaves))
    ]
    assignment_lines = [
        f"{out_conn} = {expr_code}"
        for out_conn, expr_code in zip(out_conns, expr_codes, strict=True)
    ]
    assignment_lines += [
        f"{carry_w_conn} = {expr_code}"
        for carry_w_conn, expr_code in zip(carry_write_conns, expr_codes, strict=True)
        if carry_w_conn is not None
    ]
    tasklet_code = (
        "\n".join(pre_statements_scan + pre_statements_body) + "\n" + "\n".join(assignment_lines)
    )

    # Connectivity tables referenced by ``neighbors`` in the stencil body must
    # be passed to the tasklet as inputs (full-array memlets); see the comment
    # in ``translate_as_fieldop`` for details.
    connectivity_inputs = {aname: None for aname in used_connectivities}

    # Skip the full-array memlet for fields that are only accessed through
    # scalar memlets on the input edge (``field_inputs``), see the comment in
    # ``translate_as_fieldop`` for details.
    import re

    _field_names_in_inputs = (
        {fname for fname, _ in field_inputs.values()} if field_inputs else set()
    )
    _full_array_param_names = {
        param_name
        for param_name, data in data_args.items()
        if not field_inputs
        or data.name not in _field_names_in_inputs
        or re.search(r"\b" + re.escape(data.name) + r"\b", tasklet_code)
        or re.search(r"\b" + re.escape(param_name) + r"\b", tasklet_code)
    }

    # Init connectors for list-leaf inits only — scalar-leaf inits are now
    # written to the carry container before the loop by a separate tasklet.
    list_init_connectors = {
        init_codes[i]: None
        for i in range(len(leaves))
        if carry_container_names[i] is None and init_codes[i] in init_connectors
    }

    # Create the Tasklet and TaskletNode.
    tasklet, connector_mapping = sdfg_builder.add_tasklet(
        name="scan",
        inputs={param_name: None for param_name in _full_array_param_names}
        | connectivity_inputs
        | {connector: None for connector in field_inputs}
        | {carry_conn: None for carry_conn in carry_conn_of}
        | list_init_connectors,
        outputs=set(out_conns)
        | {carry_w_conn for carry_w_conn in carry_write_conns if carry_w_conn is not None},
        code=tasklet_code,
    )

    # Input connectors: the field args (full arrays) + connectivity tables +
    # scalar field inputs + the carry reads (scalar memlets on the leaf result
    # arrays).
    in_memlets: dict[str, dace.Memlet] = {
        connector_mapping[param_name]: _make_full_array_memlet(data.name, ctx.root)
        for param_name, data in data_args.items()
        if param_name in _full_array_param_names
    }
    for aname in used_connectivities:
        ctx.root.containers[aname].transient = False
        in_memlets[connector_mapping[aname]] = _make_full_array_memlet(aname, ctx.root)
    for connector, (field_name, subset) in field_inputs.items():
        in_memlets[connector_mapping[connector]] = dace.Memlet(
            data=field_name, subset=dace_subsets.Range.from_string(subset)
        )
    for i, (carry_conn, result_name) in enumerate(zip(carry_conn_of, result_names, strict=True)):
        # NOTE: the subset objects must not be shared across memlets — DaCe
        # validation rejects a subset instance referenced by multiple edges.
        carry_container_name = carry_container_names[i]
        if carry_container_name is not None:
            # Scalar leaf: read the carry from the (per-column when a
            # horizontal map is present) carry container.  Indexing by the
            # horizontal map variable avoids every column racing on a single
            # shared scalar slot.
            in_memlets[connector_mapping[carry_conn]] = dace.Memlet(
                data=carry_container_name,
                subset=copy.deepcopy(horizontal_carry_subset)
                if horizontal_carry_subset_parts
                else "0",
            )
        else:
            # List leaf: read the carry from the result array at the previous
            # scan position (see ``carry_subset`` construction above).
            in_memlets[connector_mapping[carry_conn]] = dace.Memlet(
                data=result_name, subset=copy.deepcopy(carry_subset)
            )
    for i, init_data in enumerate(init_leaves_data):
        init_code = init_codes[i]
        if init_code in list_init_connectors:
            in_memlets[connector_mapping[init_code]] = dace.Memlet(data=init_data.name, subset="0")

    # Output memlets: one element per leaf at (horizontal_indices, k).
    output_indices = []
    for dim, origin in zip(field_dims, field_origin, strict=True):
        map_var = map_indices.get(dim)
        assert map_var is not None
        output_indices.append(f"{map_var} - ({origin})")
    output_subset = dace.subsets.Range.from_string(",".join(output_indices))
    out_memlets = {
        connector_mapping[out_conn]: dace.Memlet(
            data=result_name, subset=copy.deepcopy(output_subset)
        )
        for out_conn, result_name in zip(out_conns, result_names, strict=True)
    }
    # The scan carry output of scalar leaves is written to the (per-column when
    # a horizontal map is present) carry container, to be read by the next scan
    # iteration of the same column.
    for i, carry_w_conn in enumerate(carry_write_conns):
        if carry_w_conn is not None:
            out_memlets[connector_mapping[carry_w_conn]] = dace.Memlet(
                data=carry_container_names[i],
                subset=copy.deepcopy(horizontal_carry_subset)
                if horizontal_carry_subset_parts
                else "0",
            )

    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets=in_memlets,
        out_memlets=out_memlets,
    )

    # Build the ForScope.
    if scan_forward:
        loop = dace.sdfg.state.LoopRegion(
            label="scan",
            loop_var=scan_loop_var,
            condition_expr=f"{scan_loop_var} < {scan_domain_range.stop}",
            initialize_expr=f"{scan_loop_var} = {scan_domain_range.start}",
            update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
        )
    else:
        loop = dace.sdfg.state.LoopRegion(
            label="scan",
            loop_var=scan_loop_var,
            condition_expr=f"{scan_loop_var} >= {scan_domain_range.start}",
            initialize_expr=f"{scan_loop_var} = {scan_domain_range.stop} - 1",
            update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
        )

    for_scope = tn.ForScope(loop=loop, children=[tasklet_node])

    if map_scope is not None:
        map_scope.add_child(for_scope)
    else:
        ctx.current_scope.add_child(for_scope)

    # Rebuild the (possibly nested) tuple of leaf results, mirroring the
    # structure of the scan value type.
    def _gt_type_at_path(path: tuple[int, ...]) -> ts.FieldType | ts.ScalarType | ts.ListType:
        # Navigate `node.type`, consuming path indices through tuple nesting;
        # entering a `FieldType`'s dtype does not consume a path element.
        root_type = node.type
        assert isinstance(root_type, (ts.FieldType, ts.TupleType))
        leaf_type: ts.TypeSpec = root_type
        for index in path:
            if isinstance(leaf_type, ts.FieldType):
                leaf_dtype = leaf_type.dtype
                assert isinstance(leaf_dtype, ts.TypeSpec)
                leaf_type = leaf_dtype
            assert isinstance(leaf_type, ts.TupleType)
            child = leaf_type.types[index]
            assert isinstance(child, ts.TypeSpec)
            leaf_type = child
        if isinstance(root_type, ts.FieldType):
            if not path:
                # Scalar (non-tuple) scan output — return the field as-is.
                return root_type
            # Leaf reached through the field dtype: wrap it back in a field.
            assert isinstance(leaf_type, (ts.ScalarType, ts.ListType))
            return ts.FieldType(dims=root_type.dims, dtype=leaf_type)
        assert isinstance(leaf_type, ts.FieldType)
        return leaf_type

    leaf_results = [
        DataRef(result_name, _gt_type_at_path(path), tuple(field_origin))
        for result_name, (path, _leaf_dtype) in zip(result_names, leaves, strict=True)
    ]
    leaf_results_iter = iter(leaf_results)

    def _rebuild_result(dtype: ts.DataType) -> DataRefTree:
        if isinstance(dtype, ts.TupleType):
            subresults = []
            for subtype in dtype.types:
                if not isinstance(subtype, ts.DataType):
                    raise NotImplementedError(f"Unsupported scan output type: {subtype}.")
                subresults.append(_rebuild_result(subtype))
            return tuple(subresults)
        return next(leaf_results_iter)

    return _rebuild_result(dtype_tree)


def translate_if(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for the ``if_`` builtin (statement level).

    Uses ``IfScope``/``ElseScope`` at the statement level.  Inside the stencil
    body, ``if_`` is handled by the code generator as a ternary expression.
    """
    assert cpm.is_call_to(node, "if_")
    assert len(node.args) == 3
    cond_expr, true_expr, false_expr = node.args

    if isinstance(node.type, ts.TupleType):
        # Lower each tuple element independently: build a scalar-typed
        # `if_` on the extracted elements of the true and false branches
        # and translate it separately.  The domain annexes of tuple-typed
        # expressions hold one domain per element.
        results = []
        for index, element_type in enumerate(node.type.types):

            def _extract(
                branch: gtir.Expr,
                index: int = index,
                element_type: ts.TypeSpec = element_type,
            ) -> gtir.Expr:
                element_branch = im.tuple_get(index, branch)
                element_branch.type = element_type
                if hasattr(branch.annex, "domain"):
                    branch_domain = branch.annex.domain
                    element_branch.annex.domain = (
                        branch_domain[index] if isinstance(branch_domain, tuple) else branch_domain
                    )
                return element_branch

            element_node = im.if_(cond_expr, _extract(true_expr), _extract(false_expr))
            element_node.type = element_type
            if hasattr(node.annex, "domain"):
                element_node.annex.domain = (
                    node.annex.domain[index]
                    if isinstance(node.annex.domain, tuple)
                    else node.annex.domain
                )
            results.append(translate_if(element_node, ctx, sdfg_builder))
        return tuple(results)

    # Generate the condition code.
    cond_code = get_source(cond_expr)

    # Visit the true and false branches.
    SubgraphContext(root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes)
    SubgraphContext(root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes)

    # Allocate the output field (as a temp on root.containers).
    assert isinstance(node.type, ts.FieldType)
    field_domain = get_field_domain(node.annex.domain)
    _field_dims, field_origin, field_shape = get_field_layout(field_domain)
    if isinstance(node.type.dtype, ts.ScalarType):
        dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    else:
        assert isinstance(node.type.dtype, ts.ListType)
        assert isinstance(node.type.dtype.element_type, ts.ScalarType)
        dtype = gtx_dace_args.as_dace_type(node.type.dtype.element_type)
    out_name, _ = sdfg_builder.add_temp_array(ctx.root, field_shape, dtype)
    out_data = DataRef(out_name, node.type, tuple(field_origin))

    # Visit the true branch — results are added to the IfScope.
    if_scope = tn.IfScope(
        condition=dace.properties.CodeBlock(cond_code, dace.dtypes.Language.Python),
        children=[],
    )
    ctx.current_scope.add_child(if_scope)
    true_ctx_inner = SubgraphContext(
        root=ctx.root, current_scope=if_scope, data_nodes=ctx.data_nodes
    )
    true_result = sdfg_builder.visit(true_expr, ctx=true_ctx_inner)

    # Add a CopyNode to write the true result to the output.
    if true_result is not None and isinstance(true_result, DataRef):
        src_subset = _make_access_index_for_field(
            get_field_domain(true_expr.annex.domain), true_result
        )
        dst_subset = _make_access_index_for_field(field_domain, out_data)
        if_scope.add_child(
            tn.CopyNode(
                target=out_name,
                memlet=dace.Memlet(
                    data=true_result.name, subset=src_subset, other_subset=dst_subset
                ),
            )
        )

    # Visit the false branch — results are added to the ElseScope.
    else_scope = tn.ElseScope(children=[])
    ctx.current_scope.add_child(else_scope)
    false_ctx_inner = SubgraphContext(
        root=ctx.root, current_scope=else_scope, data_nodes=ctx.data_nodes
    )
    false_result = sdfg_builder.visit(false_expr, ctx=false_ctx_inner)

    # Add a CopyNode to write the false result to the output.
    if false_result is not None and isinstance(false_result, DataRef):
        src_subset = _make_access_index_for_field(
            get_field_domain(false_expr.annex.domain), false_result
        )
        dst_subset = _make_access_index_for_field(field_domain, out_data)
        else_scope.add_child(
            tn.CopyNode(
                target=out_name,
                memlet=dace.Memlet(
                    data=false_result.name, subset=src_subset, other_subset=dst_subset
                ),
            )
        )

    return out_data


def translate_symbol_ref(
    node: gtir.SymRef,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for a ``ir.SymRef`` node."""

    # Check if the symbol is unused (domain is NEVER).  Only field-typed
    # symbols can be pruned this way: scalar arguments of field operators
    # have no spatial access domain, so domain inference marks them NEVER
    # even when they are actually used (e.g. scalar let-bound arguments,
    # like the CollapseTuple ``__ct_el_N`` temporaries).
    node_domain = getattr(node.annex, "domain", infer_domain.DomainAccessDescriptor.UNKNOWN)
    if node_domain == infer_domain.DomainAccessDescriptor.NEVER and not isinstance(
        node.type, ts.ScalarType
    ):
        return None

    def _translate_symbol_ref_inner(node: gtir.SymRef) -> DataRef:
        symbol_name = str(node.id)
        if symbol_name in ctx.data_nodes:
            data = ctx.data_nodes[symbol_name]
            assert data is not None
            return data
        elif symbol_name in ctx.root.containers:
            assert isinstance(node.type, ts.FieldType)
            return sdfg_builder.make_field(symbol_name, node.type)
        elif symbol_name in ctx.root.symbols:
            return translate_scalar_expr(node, ctx, sdfg_builder)

        raise ValueError(f"Symbol '{symbol_name}' not found in scope.")

    if isinstance(node.type, ts.TupleType):
        symbol_name = str(node.id)
        # A tuple may have been let-bound: it is stored in ``data_nodes``
        # under its un-flattened name as a nested tuple of results (e.g. the
        # result of a tuple-output scan consumed by a later statement).
        if symbol_name in ctx.data_nodes:
            data = ctx.data_nodes[symbol_name]
            assert data is not None
            return data
        # Tuple program parameters are flattened into per-element
        # symbols/containers (``__tuple_tmp_0`` → ``__tuple_tmp_0_0``, ...).
        sym_tree = make_symbol_tree(symbol_name, node.type)
        return gtx_utils.tree_map(lambda x: _translate_symbol_ref_inner(im.ref(x.id, x.type)))(
            sym_tree
        )
    else:
        return _translate_symbol_ref_inner(node)


def translate_scalar_expr(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRef:
    """Generates a tasklet for a scalar-valued expression.

    Symbols bound by let-lambdas (present in ``ctx.data_nodes``) are passed
    to the tasklet as scalar input connectors; free SDFG symbols (in
    ``ctx.root.symbols``) are emitted verbatim in the tasklet code.  Blindly
    printing the expression with ``get_source`` would leave let-bound IR
    symbols as dangling free symbols in the generated code.
    """
    assert isinstance(node.type, ts.ScalarType)
    assert isinstance(node, (gtir.FunCall, gtir.Literal, gtir.SymRef))

    # Collect free symbols and resolve them to data containers or SDFG
    # symbols.
    free_syms = {str(sym.id) for sym in eve.walk_values(node).if_isinstance(gtir.SymRef).to_set()}
    data_args: dict[str, DataRef] = {}
    for sym in free_syms:
        if sym in ctx.data_nodes:
            data = ctx.data_nodes[sym]
            if isinstance(data, DataRef):
                assert isinstance(data.gt_type, ts.ScalarType)
                data_args[sym] = data
    string_args = {sym: sym for sym in free_syms if sym in ctx.root.symbols}

    # Generate the Python code for the scalar expression.
    expr_code, pre_statements, used_connectivities, field_inputs = generate_tasklet_code(
        node,
        data_args=data_args,
        map_indices={},
        offset_provider_type=sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
        if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
        else ctx.root._offset_provider_type,
        string_args=string_args,
        root_symbols=_root_symbol_names(ctx),
    )
    assert not pre_statements, f"Unexpected pre-statements in scalar expression '{node}'."
    assert not field_inputs, f"Unexpected field inputs in scalar expression '{node}'."
    tasklet_code = f"out = {expr_code}"

    # Create a temporary scalar for the result.
    result_name, _ = sdfg_builder.add_temp_scalar(ctx.root, gtx_dace_args.as_dace_type(node.type))

    # Create the TaskletNode.
    tasklet, connector_mapping = sdfg_builder.add_tasklet(
        name="scalar_expr",
        inputs={sym: None for sym in data_args} | {c: None for c in used_connectivities},
        outputs={"out"},
        code=tasklet_code,
    )
    in_memlets: dict[str, dace.Memlet] = {
        connector_mapping[sym]: dace.Memlet(data=data.name, subset="0")
        for sym, data in data_args.items()
    }
    for aname in used_connectivities:
        ctx.root.containers[aname].transient = False
        in_memlets[connector_mapping[aname]] = _make_full_array_memlet(aname, ctx.root)
    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets=in_memlets,
        out_memlets={connector_mapping["out"]: dace.Memlet(data=result_name, subset="0")},
    )
    ctx.current_scope.add_child(tasklet_node)

    return DataRef(result_name, node.type, origin=())


def translate_index(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRef:
    """Generates the schedule-tree nodes for the ``index`` builtin.

    ``index(dim)`` produces a 1-D field whose value at each grid point is
    the index along ``dim``.  It is lowered to a mapped tasklet writing the
    map index to a temporary 1-D array over the node domain (taken from the
    domain-inference annex), matching ``translate_index`` in the SDFG
    lowering.
    """
    assert cpm.is_call_to(node, "index")
    assert isinstance(node.type, ts.FieldType)

    field_domain = get_field_domain(node.annex.domain)
    assert len(field_domain) == 1
    domain_range = field_domain[0]
    map_index = get_map_variable(domain_range.dim)

    # Allocate the result array.
    _field_dims, field_origin, field_shape = get_field_layout(field_domain)
    result_scalar_type = node.type.dtype
    assert isinstance(result_scalar_type, ts.ScalarType)
    result_name, _ = sdfg_builder.add_temp_array(
        ctx.root, field_shape, gtx_dace_args.as_dace_type(result_scalar_type)
    )

    # Build the map over the index domain.
    map_entry, _map_exit = sdfg_builder.add_map(
        "index", {map_index: f"{domain_range.start}:{domain_range.stop}"}
    )

    tasklet, connector_mapping = sdfg_builder.add_tasklet(
        name="index",
        inputs={},
        outputs={"__out"},
        code=f"__out = {map_index}",
    )
    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets={},
        out_memlets={
            connector_mapping["__out"]: dace.Memlet(
                data=result_name, subset=f"{map_index} - ({field_origin[0]})"
            )
        },
    )
    map_scope = tn.MapScope(node=map_entry, children=[tasklet_node])
    ctx.current_scope.add_child(map_scope)

    return DataRef(result_name, node.type, origin=tuple(field_origin))


def translate_make_tuple(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for ``make_tuple``."""
    assert cpm.is_call_to(node, "make_tuple")
    return tuple(sdfg_builder.visit(arg, ctx=ctx) for arg in node.args)


def translate_tuple_get(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for ``tuple_get``."""
    assert cpm.is_call_to(node, "tuple_get")
    assert len(node.args) == 2

    if not isinstance(node.args[0], gtir.Literal):
        raise ValueError("Tuple can only be subscripted with compile-time constants.")
    assert ti.is_integral(node.args[0].type)
    index = int(node.args[0].value)

    data_nodes = sdfg_builder.visit(node.args[1], ctx=ctx)
    if not isinstance(data_nodes, tuple):
        raise ValueError(f"Invalid tuple expression {node}.")
    return data_nodes[index]


def _translate_concat_where_branch(
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
    source_expr: gtir.Expr,
    is_lower: bool,
    concat_dim: gtx_common.Dimension,
    concat_dim_bound_expr: gtir.Expr,
    branch_cond_domain: domain_utils.SymbolicDomain,
    output_domain: domain_utils.SymbolicDomain,
    output_type: ts.FieldType,
    output_desc: dace.data.Array,
    output_name: str,
    output_origin: Sequence[dace.symbolic.SymbolicType],
) -> None:
    """
    Translate one of the two branches of a 'concat_where' expression.

    The 'concat_where' expression requires two input fields, one is written on the
    lower part of the result domain, with respect to the domain boundary extracted
    from the first argument; the other input field is written on the upper domain.
    The handling of the two branches is similar, there is only one small difference
    in case the input field does not contain the 'concat_where' dimension, which
    is the case of scalar values or horizontal slice fields. In this case, the input
    is broadcast on the full output domain by translating it into a 'deref' field
    operator on the promoted domain, and the `is_lower` argument is used to select
    which part of the output domain is written.

    This is a direct port of the corresponding function in `gtir_to_sdfg_concat_where.py`.
    """
    assert isinstance(source_expr.type, (ts.FieldType, ts.ScalarType))

    # The domain annex on program parameter references is the union of all
    # accesses, which can be wider than the region actually read in this
    # branch (the SDFG lowering path relies on per-branch `__cwcda` let
    # bindings, see `canonicalize_domain_argument`, to obtain tight domains,
    # but the stree pipeline inlines those lets again via `FuseAsFieldOp`).
    # Intersect with the branch condition domain to tighten the read domain.
    source_domain = source_expr.annex.domain
    cond_dims_in_source = [
        dim for dim in source_domain.ranges.keys() if dim in branch_cond_domain.ranges
    ]
    if cond_dims_in_source:
        source_domain = domain_utils.domain_intersection(
            source_domain,
            domain_utils.promote_domain(
                domain_utils.SymbolicDomain(
                    branch_cond_domain.grid_type,
                    {dim: branch_cond_domain.ranges[dim] for dim in cond_dims_in_source},
                ),
                source_domain.ranges.keys(),
            ),
        )
    if isinstance(source_expr.type, ts.ScalarType) or len(source_expr.type.dims) < len(
        output_type.dims
    ):
        # We promote the input expression to a field defined on the output domain:
        # either a scalar value, broadcast on all dimensions, or a field defined
        # on a slice of the output domain (e.g. a 2D boundary field broadcast on
        # all levels of a 3D 'concat_where' result).
        if concat_dim not in source_domain.ranges:
            source_domain.ranges[concat_dim] = (
                domain_utils.SymbolicRange(
                    start=output_domain.ranges[concat_dim].start,
                    stop=concat_dim_bound_expr,
                )
                if is_lower
                else domain_utils.SymbolicRange(
                    start=concat_dim_bound_expr,
                    stop=output_domain.ranges[concat_dim].stop,
                )
            )
        source_domain = domain_utils.promote_domain(source_domain, output_type.dims)
        source_domain = domain_utils.domain_intersection(source_domain, output_domain)

        # Use a 'deref' field operator to broadcast the input expression on the target domain.
        bcast_expr = im.as_fieldop("deref", source_domain.as_expr())(source_expr)
        bcast_expr.type = output_type

        source = sdfg_builder.visit(bcast_expr, ctx=ctx)
    else:
        # The input field is defined on all dimensions of the result field.
        source = sdfg_builder.visit(source_expr, ctx=ctx)

    assert isinstance(source, DataRef) and source.gt_type == output_type
    source_domain_range = source_domain.ranges[concat_dim]
    source_range_0 = get_symbolic(source_domain_range.start)
    source_range_1 = get_symbolic(im.maximum(source_domain_range.start, source_domain_range.stop))
    source_range_size = source_range_1 - source_range_0

    assert isinstance(output_type.dtype, ts.ScalarType)
    all_dims = gtx_common.order_dimensions(output_type.dims)

    source_subset = []
    output_subset = []
    for dim, size, src_origin, dst_origin in zip(
        all_dims,
        output_desc.shape,
        source.origin,
        output_origin,
        strict=True,
    ):
        if dim == concat_dim:
            # Write only the subset corresponding to the range of lower or upper branch.
            source_subset.append(
                (
                    source_range_0 - src_origin,
                    source_range_1 - src_origin - 1,
                    1,
                )
            )
            output_subset.append(
                (
                    source_range_0 - dst_origin,
                    source_range_0 - dst_origin + source_range_size - 1,
                    1,
                )
            )
        else:
            # Write the full subset which covers the array size in this dimension.
            source_subset.append(
                (
                    dst_origin - src_origin,
                    dst_origin - src_origin + size - 1,
                    1,
                )
            )
            output_subset.append((0, size - 1, 1))

    ctx.current_scope.add_child(
        tn.CopyNode(
            target=output_name,
            memlet=dace.Memlet(
                data=source.name,
                subset=dace_subsets.Range(source_subset),
                other_subset=dace_subsets.Range(output_subset),
            ),
        )
    )


def translate_concat_where(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> DataRefTree:
    """Generates the schedule-tree nodes for ``concat_where`` using CopyNode.

    Lowers a `concat_where` expression to a dataflow where two memlets write
    disjoint subsets, for the lower and upper domain, on one temporary field.
    Direct port of the corresponding lowering in `gtir_to_sdfg_concat_where.py`.
    """
    assert cpm.is_call_to(node, "concat_where")
    assert len(node.args) == 3
    assert isinstance(node.type, ts.FieldType)

    # First argument is a domain expression that defines the mask.
    mask_domain = domain_utils.SymbolicDomain.from_expr(node.args[0])
    if len(mask_domain.ranges) != 1:
        raise NotImplementedError("Expected `concat_where` along single axis.")

    concat_dim = next(iter(mask_domain.ranges.keys()))

    # Determine lower and upper branches.
    infinity_literals = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)
    if mask_domain.ranges[concat_dim].start in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].stop
        lower_expr, upper_expr = node.args[1:]
        lower_cond = mask_domain
        upper_cond = domain_utils.domain_complement(mask_domain)
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_expr, lower_expr = node.args[1:]
        upper_cond = mask_domain
        lower_cond = domain_utils.domain_complement(mask_domain)
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    if not isinstance(node.type.dtype, ts.ScalarType):
        # TODO(edopao): Refactor allocation of fields with local dimension and enable this.
        raise NotImplementedError("'concat_where' with list output is not supported")

    # Allocate the output field.
    output_domain = get_field_domain(node.annex.domain)
    output_dims, output_origin, output_shape = get_field_layout(output_domain)
    assert output_dims == node.type.dims
    dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    output_name, output_desc = sdfg_builder.add_temp_array(ctx.root, output_shape, dtype)

    # Translate the input expression on the lower domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=lower_expr,
        is_lower=True,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        branch_cond_domain=lower_cond,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_name=output_name,
        output_origin=output_origin,
    )

    # Translate the input expression on the upper domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=upper_expr,
        is_lower=False,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        branch_cond_domain=upper_cond,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_name=output_name,
        output_origin=output_origin,
    )

    return DataRef(output_name, node.type, tuple(output_origin))


# ---------------------------------------------------------------------------
# Main visitor
# ---------------------------------------------------------------------------


class GTIRToScheduleTree(eve.NodeVisitor, SDFGBuilder):
    """Translates a GTIR program to a DaCe Schedule Tree.

    A single instance of this visitor is used for the entire lowering.
    The level-specific information, including the data symbols available
    in the lowering scope, is stored inside a ``SubgraphContext`` object
    that can be accessed by the visitor methods.
    """

    offset_provider_type: gtx_common.OffsetProviderType
    column_axis: Optional[gtx_common.Dimension]
    uids: gtx_utils.IDGeneratorPool = dataclasses.field(
        init=False, repr=False, default_factory=lambda: gtx_utils.IDGeneratorPool()
    )

    def __init__(
        self,
        offset_provider_type: gtx_common.OffsetProviderType,
        column_axis: Optional[gtx_common.Dimension] = None,
    ):
        self.offset_provider_type = offset_provider_type
        self.column_axis = column_axis
        self.uids = gtx_utils.IDGeneratorPool()

    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem:
        return gtx_common.get_offset_type(self.offset_provider_type, offset)

    def make_field(
        self,
        name: str,
        data_type: ts.FieldType,
    ) -> DataRef:
        """Retrieve the field descriptor for a data container by name."""
        local_dims = [dim for dim in data_type.dims if dim.kind == gtx_common.DimensionKind.LOCAL]
        if len(local_dims) == 0:
            field_type = data_type
        elif len(local_dims) == 1:
            local_dim = local_dims[0]
            if not isinstance(data_type.dtype, ts.ScalarType):
                raise ValueError(f"Invalid field type {data_type}.")
            if not gtx_common.has_offset(self.offset_provider_type, local_dim.value):
                raise ValueError(
                    f"The provided local dimension {local_dim} does not match any offset provider type."
                )
            local_type = ts.ListType(element_type=data_type.dtype, offset_type=local_dim)
            field_type = ts.FieldType(
                dims=[dim for dim in data_type.dims if dim != local_dim], dtype=local_type
            )
        else:
            raise NotImplementedError(
                "Fields with more than one local dimension are not supported."
            )
        field_origin = tuple(gtx_dace_args.range_start_symbol(name, dim) for dim in field_type.dims)
        return DataRef(name, field_type, field_origin)

    def is_column_axis(self, dim: gtx_common.Dimension) -> bool:
        assert self.column_axis
        return dim == self.column_axis

    def unique_nsdfg_name(self, prefix: str) -> str:
        return next(self.uids[prefix])

    def unique_map_name(self, name: str) -> str:
        return f"{next(self.uids['map'])}_{name}"

    def unique_tasklet_name(self, name: str) -> str:
        return f"{next(self.uids['tlet'])}_{name}"

    def unique_temp_name(self) -> str:
        return f"{next(self.uids['gtir_tmp'])}"

    def unique_lib_node_name(self, lib_node_type: str) -> str:
        return f"{next(self.uids[lib_node_type])}"

    def _make_array_shape_and_strides(
        self, name: str, dims: Sequence[gtx_common.Dimension]
    ) -> tuple[list[dace.symbolic.SymbolicType], list[dace.symbolic.SymbolicType]]:
        """Parse field dimensions and allocate symbols for array shape and strides."""
        neighbor_table_types = gtx_dace_args.filter_connectivity_types(self.offset_provider_type)
        shape = []
        for dim in dims:
            if dim.kind == gtx_common.DimensionKind.LOCAL:
                shape.append(neighbor_table_types[dim.value].max_neighbors)
            elif gtx_dace_args.is_connectivity_identifier(name, self.offset_provider_type):
                shape.append(gtx_dace_args.field_size_symbol(name, dim, neighbor_table_types))
            else:
                shape.append(
                    dace.symbolic.pystr_to_symbolic(
                        "{} - {}".format(
                            gtx_dace_args.range_stop_symbol(name, dim),
                            gtx_dace_args.range_start_symbol(name, dim),
                        )
                    )
                )
        strides = [
            gtx_dace_args.field_stride_symbol(name, dim, neighbor_table_types) for dim in dims
        ]
        return shape, strides

    def _add_storage(
        self,
        root: tn.ScheduleTreeRoot,
        symbolic_params: set[str] | None,
        name: str,
        gt_type: ts.DataType,
        transient: bool,
    ) -> list[tuple[str, ts.DataType]]:
        """Add storage to the root's containers and symbols."""
        if isinstance(gt_type, ts.TupleType):
            tuple_fields = []
            for sym in flatten_tuple_fields(name, gt_type):
                assert isinstance(sym.type, ts.DataType)
                tuple_fields.extend(
                    self._add_storage(
                        root=root,
                        symbolic_params=symbolic_params,
                        name=str(sym.id),
                        gt_type=sym.type,
                        transient=transient,
                    )
                )
            return tuple_fields

        elif isinstance(gt_type, ts.FieldType):
            if len(gt_type.dims) == 0:
                return self._add_storage(
                    root=root,
                    symbolic_params=set(),
                    name=name,
                    gt_type=gt_type.dtype,
                    transient=transient,
                )
            if isinstance(gt_type.dtype, ts.ScalarType):
                dc_dtype = gtx_dace_args.as_dace_type(gt_type.dtype)
                all_dims = gt_type.dims
            else:
                assert gt_type.dtype.offset_type is not None
                assert gt_type.dtype.offset_type.kind == gtx_common.DimensionKind.LOCAL
                assert isinstance(gt_type.dtype.element_type, ts.ScalarType)
                dc_dtype = gtx_dace_args.as_dace_type(gt_type.dtype.element_type)
                all_dims = gtx_common.order_dimensions([*gt_type.dims, gt_type.dtype.offset_type])

            sym_shape, sym_strides = self._make_array_shape_and_strides(name, all_dims)
            array = dace.data.Array(dc_dtype, sym_shape, strides=sym_strides, transient=transient)
            root.containers[name] = array
            # Ensure free symbols in shape/strides are on root.symbols
            _collect_free_symbols(sym_shape, root)
            _collect_free_symbols(sym_strides, root)
            return [(name, gt_type)]

        elif isinstance(gt_type, ts.ScalarType):
            dc_dtype = gtx_dace_args.as_dace_type(gt_type)
            if symbolic_params is None or name in symbolic_params:
                root.symbols[name] = dc_dtype
            else:
                root.containers[name] = dace.data.Scalar(dc_dtype, transient=transient)
            return [(name, gt_type)]

        raise RuntimeError(f"Data type '{type(gt_type)}' not supported.")

    def _add_sdfg_params(
        self,
        root: tn.ScheduleTreeRoot,
        node_params: Sequence[gtir.Sym],
        symbolic_params: set[str] | None,
        use_transient_storage: bool,
    ) -> list[str]:
        """Add storage for node parameters and connectivity tables."""
        sdfg_args = []
        for param in node_params:
            gt_symbol_name = str(param.id)
            assert isinstance(param.type, ts.DataType)
            sdfg_args += self._add_storage(
                root=root,
                symbolic_params=symbolic_params,
                name=gt_symbol_name,
                gt_type=param.type,
                transient=use_transient_storage,
            )

        # Add storage for connectivity tables.
        for offset, connectivity_type in gtx_dace_args.filter_connectivity_types(
            self.offset_provider_type
        ).items():
            gt_type = ts.FieldType(
                dims=[connectivity_type.source_dim, connectivity_type.neighbor_dim],
                dtype=tt.from_dtype(connectivity_type.dtype),
            )
            self._add_storage(
                root=root,
                symbolic_params=symbolic_params,
                name=gtx_dace_args.connectivity_identifier(offset),
                gt_type=gt_type,
                transient=True,
            )

        return [arg_name for arg_name, _ in sdfg_args]

    def visit_Program(self, node: gtir.Program) -> tn.ScheduleTreeRoot:
        """Translates ``ir.Program`` to a ``ScheduleTreeRoot``."""
        root = tn.ScheduleTreeRoot(
            name=str(node.id),
            children=[],
            containers={},
            symbols={},
            constants={},
            callback_mapping={},
            arg_names=[],
        )
        root._offset_provider_type = self.offset_provider_type

        # Start: children are added directly to root (no GBlock —
        # from_schedule_tree does not support GBlock yet).
        # Pre-allocate storage for all program parameters.
        sdfg_arg_names = self._add_sdfg_params(
            root, node.params, symbolic_params=None, use_transient_storage=False
        )
        root.arg_names = sdfg_arg_names

        # Visit one statement at a time.
        data_nodes: dict[str, DataRef | None] = {}
        for p in node.params:
            sym_type = p.type
            if (sym_name := str(p.id)) in root.containers:
                assert isinstance(sym_type, ts.FieldType)
                data_nodes[sym_name] = self.make_field(sym_name, sym_type)

        for i, stmt in enumerate(node.body):
            # Insert a StateBoundaryNode between statements to ensure
            # they are lowered to separate SDFG states.
            if i > 0:
                root.add_child(tn.StateBoundaryNode())
            ctx = SubgraphContext(root=root, current_scope=root, data_nodes=data_nodes)
            self.visit(stmt, ctx=ctx)

        # Remove unused connectivity tables.
        used_data = set()
        for child in root.children:
            if isinstance(child, tn.CopyNode):
                used_data.add(child.memlet.data)
                used_data.add(child.target)
            elif isinstance(child, tn.TaskletNode):
                for m in child.in_memlets.values():
                    used_data.add(m.data)
                for m in child.out_memlets.values():
                    used_data.add(m.data)
        for data_name in list(root.containers.keys()):
            if (
                gtx_dace_args.is_connectivity_identifier(data_name, self.offset_provider_type)
                and root.containers[data_name].transient
                and data_name not in used_data
            ):
                root.containers.pop(data_name)

        return root

    def visit_SetAt(self, stmt: gtir.SetAt, ctx: SubgraphContext) -> None:
        """Visits a ``SetAt`` statement and writes the result to the target."""
        # Visit the field operator expression.
        source_tree = self._visit_expression(stmt.expr, ctx)

        # Visit the target expression.  It can be a ``SymRef`` to an output
        # field (including a tuple-typed symbol, handled by
        # ``translate_symbol_ref``) or a ``make_tuple`` expression when the
        # statement returns more than one field — same handling as the SDFG
        # lowering.
        target_tree = self._visit_expression(
            stmt.target,
            ctx=SubgraphContext(
                root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes
            ),
            use_temp=False,
        )

        # Write the result to the target using CopyNode(s).
        domain = extract_target_domain(stmt.domain)

        # If the write target is also read by the statement expression (i.e. the
        #  field is updated "in place"), the write-back copies have to be placed
        #  in a separate, later state: inside one SDFG state the copy would race
        #  with the reads of the old value, since the dataflow graph only orders
        #  read-after-write dependencies (WAR hazards are not tracked).
        expr_reads = {str(sym.id) for sym in stmt.expr.pre_walk_values().if_isinstance(gtir.SymRef)}
        target_writes = {
            str(sym.id) for sym in stmt.target.pre_walk_values().if_isinstance(gtir.SymRef)
        }
        if expr_reads & target_writes:
            ctx.current_scope.add_child(tn.StateBoundaryNode())

        def _write_target(
            source: DataRef | None,
            target: DataRef | None,
            target_domain: Any,
        ) -> None:
            if source is None or target is None:
                return
            assert source.gt_type == target.gt_type
            field_domain = get_field_domain(target_domain)
            src_subset = _make_access_index_for_field(field_domain, source)
            dst_subset = _make_access_index_for_field(field_domain, target)
            ctx.current_scope.add_child(
                tn.CopyNode(
                    target=target.name,
                    memlet=dace.Memlet(
                        data=source.name, subset=src_subset, other_subset=dst_subset
                    ),
                )
            )

        gtx_utils.tree_map(_write_target)(source_tree, target_tree, domain)

    def _visit_expression(
        self,
        node: gtir.Expr,
        ctx: SubgraphContext,
        use_temp: bool = True,
    ) -> DataRefTree:
        """Specialized visit method for fieldview expressions."""
        result = self.visit(node, ctx=ctx)

        if use_temp and result is not None:
            # Copy the full shape of global data to temporary storage.
            def _maybe_copy(x: DataRef | None) -> DataRef | None:
                if x is None:
                    return None
                if x.name in ctx.root.containers and not ctx.root.containers[x.name].transient:
                    return ctx.copy_data(self, x, domain=None)
                return x

            return gtx_utils.tree_map(_maybe_copy)(result)
        else:
            return result

    def visit_FunCall(
        self,
        node: gtir.FunCall,
        ctx: SubgraphContext,
    ) -> DataRefTree:
        # Let-lambda: inline via data_nodes | args (no nested SDFG).
        if isinstance(node.fun, gtir.Lambda):
            args = {
                str(param.id): self.visit(arg, ctx=ctx)
                for param, arg in zip(node.fun.params, node.args, strict=True)
            }
            let_ctx = SubgraphContext(
                root=ctx.root,
                current_scope=ctx.current_scope,
                data_nodes=ctx.data_nodes | args,
            )
            return self.visit(node.fun.expr, ctx=let_ctx)

        # Pattern-matched builtins (structural).
        if cpm.is_applied_as_fieldop(node):
            return translate_as_fieldop(node, ctx, self)

        if cpm.is_applied_map(node):
            return translate_map_list(node, ctx, self)

        # ``tuple_get`` is handled at the IR level (indexing the nested result
        # of the tuple expression) also when the result is a scalar — needed
        # for tuples of scalars, e.g. flattened named-collection arguments.
        if isinstance(node.fun, gtir.SymRef) and str(node.fun.id) == "tuple_get":
            return translate_tuple_get(node, ctx, self)

        if isinstance(node.type, ts.ScalarType):
            return translate_scalar_expr(node, ctx, self)

        if isinstance(node.fun, gtir.SymRef):
            # Name-matched builtins.
            name = str(node.fun.id)
            if name == "if_":
                return translate_if(node, ctx, self)
            if name == "make_tuple":
                return translate_make_tuple(node, ctx, self)
            if name == "concat_where":
                return translate_concat_where(node, ctx, self)
            if name == "index":
                return translate_index(node, ctx, self)

        raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Literal(
        self,
        node: gtir.Literal,
        ctx: SubgraphContext,
    ) -> DataRefTree:
        return translate_scalar_expr(node, ctx, self)

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        ctx: SubgraphContext,
    ) -> DataRefTree:
        return translate_symbol_ref(node, ctx, self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


class _InlineSymbolicScalarLetArgs(eve.NodeTranslator):
    """Inline scalar let-arguments that are pure symbolic expressions.

    A scalar let-argument is *symbolic* when it only depends on scalar
    program parameters and builtin operators — e.g. ``n_lev - 1`` where
    ``n_lev`` is an ``int32`` program parameter.  Such expressions are
    rendered as symbolic expressions (domain bounds, memlet subset sizes)
    by the schedule-tree lowering, which flattens everything into a single
    SDFG and therefore cannot bind lambda-local scalar names through
    nested-SDFG symbol mapping (unlike the SDFG lowering, cf.
    ``SymbolicData`` in ``lowering.gtir_to_sdfg_types``).  Substituting them
    removes otherwise dangling symbol names from domain expressions and
    scalar expressions.

    Unlike the ``InlineScalar`` pass (``iterator.transforms.inline_scalar``),
    this pass also descends into ``as_fieldop`` expressions, since scalar
    lets inside field-operator bodies (e.g. created by ``CollapseTuple`` for
    tuple ``astype`` calls) are subject to the same dangling-symbol problem.

    Like ``_visit_let`` in the SDFG lowering, ``cast_`` expressions are not
    inlined: they cannot be represented as dace symbolic expressions.

    Note: nodes rebuilt by this pass drop their (stale) domain annexes;
    domain inference must be re-run afterwards.
    """

    def __init__(self, scalar_params: set[str]):
        self._symbol_universe = scalar_params | itir_builtins.BUILTINS

    def _is_symbolic_scalar_arg(self, arg: gtir.Expr) -> bool:
        if not isinstance(arg.type, ts.ScalarType):
            return False
        if any(eve.walk_values(arg).map(lambda n: cpm.is_call_to(n, "cast_"))):
            return False
        free_syms = {
            str(sym.id) for sym in eve.walk_values(arg).if_isinstance(gtir.SymRef).to_set()
        }
        return all(s in self._symbol_universe for s in free_syms)

    def visit_FunCall(self, node: gtir.FunCall) -> gtir.Node:
        node = self.generic_visit(node)
        if not cpm.is_let(node):
            return node
        eligible_params = [self._is_symbolic_scalar_arg(arg) for arg in node.args]
        if any(eligible_params):
            return inline_lambdas.inline_lambda(node, eligible_params=eligible_params)
        return node


def inline_symbolic_scalar_let_args(ir: gtir.Program) -> gtir.Program:
    """Inline symbolic scalar let-arguments (see :class:`_InlineSymbolicScalarLetArgs`).

    Applied to convergence in order to resolve chains of scalar bindings,
    where the argument of one let refers to the parameter of another one.
    """
    scalar_params = {str(p.id) for p in ir.params if isinstance(p.type, ts.ScalarType)}
    for _ in range(50):
        inlined = _InlineSymbolicScalarLetArgs(scalar_params).visit(ir)
        if inlined == ir:
            return inlined
        ir = inlined
    raise RuntimeError("Inlining of symbolic scalar let-arguments did not converge.")


def lower_program_to_stree(
    ir: gtir.Program,
    offset_provider_type: gtx_common.OffsetProviderType,
    column_axis: Optional[gtx_common.Dimension] = None,
) -> tn.ScheduleTreeRoot:
    """Receives a GTIR program and lowers it to a DaCe Schedule Tree.

    The schedule tree can be converted to an SDFG via
    ``ScheduleTreeRoot.as_sdfg(skip=...)``.

    Args:
        ir: The GTIR program node to be lowered.
        offset_provider_type: The definitions of offset providers used by the program.
        column_axis: Vertical dimension used for column scan expressions.

    Returns:
        A ``ScheduleTreeRoot`` representing the lowered program.
    """
    if ir.function_definitions:
        raise NotImplementedError("Functions expected to be inlined as lambda calls.")
    if ir.declarations:
        raise NotImplementedError("Temporaries not supported yet by GTIR DaCe stree backend.")

    ir = gtir_type_inference.infer(ir, offset_provider_type=offset_provider_type)
    ir = ir_prune_casts.PruneCasts().visit(ir)

    # DaCe requires C-compatible strings for the names of data containers,
    # such as arrays and scalars. GT4Py uses a unicode symbols ('ᐞ') as name
    # separator in the SSA pass, which generates invalid symbols for DaCe.
    # Here we find new names for invalid symbols present in the IR.
    ir = replace_invalid_symbols(ir)

    visitor = GTIRToScheduleTree(offset_provider_type, column_axis)
    import os

    if os.environ.get("GT4PY_STREE_IR_DUMP"):
        with open(f"/tmp/stree_ir_{ir.id}.txt", "w") as _f:
            _f.write(str(ir))
    stree = visitor.visit(ir)
    assert isinstance(stree, tn.ScheduleTreeRoot)

    if os.environ.get("GT4PY_STREE_IR_DUMP"):
        with open(f"/tmp/stree_tree_{ir.id}.txt", "w") as _f:
            _f.write(str(stree))

    return stree
