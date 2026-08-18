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
- **Scan** as a ``ForScope`` inside a ``MapScope``; carry =
  ``O[i-1] if i > 0 else A`` (no separate transient).
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import dace
from dace import nodes as dace_nodes, subsets as dace_subsets
from dace.frontend.python import astutils as dace_astutils
from dace.sdfg import state as dace_state
from dace.sdfg.analysis.schedule_tree import treenodes as tn

from gt4py import eve
from gt4py.eve import concepts
from gt4py.next import common as gtx_common, utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import inline_literal, prune_casts as ir_prune_casts
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_codegen import (
    generate_tasklet_code,
)
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_types import (
    FieldopData,
    FieldopResult,
    SubgraphContext,
)
from gt4py.next.program_processors.runners.dace.lowering_stree.gtir_to_stree_utils import (
    FieldopDomain,
    extract_target_domain,
    flatten_tuple_fields,
    get_field_domain,
    get_field_layout,
    get_map_variable,
    get_source,
    get_symbolic,
    make_tasklet_connector_for,
    replace_invalid_symbols,
)
from gt4py.next.type_system import type_specifications as ts, type_translation as tt


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


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
            flat_symbols.extend(
                f for f in flatten_tuple_fields(sym.id, sym.type)
            )
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
        array = dace.data.Array(dtype, shape)
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
        scalar = dace.data.Scalar(dtype)
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
            conn: make_tasklet_connector_for(conn)
            for conn in (inputs.keys() | outputs.keys())
        }
        new_code = _replace_connectors_in_code_string(code, language, connector_mapping)

        inputs = {connector_mapping[k]: v for k, v in inputs.items()}
        outputs = {connector_mapping[k]: v for k, v in outputs.items()}
        unique_name = self.unique_tasklet_name(name)
        tasklet = dace.nodes.Tasklet(unique_name, inputs=inputs, outputs=outputs, code=new_code, language=language, **kwargs)
        return tasklet, connector_mapping


class SDFGBuilder(DataflowBuilder, Protocol):
    """Visitor interface available to GTIR-primitive translators."""

    @abc.abstractmethod
    def make_field(
        self,
        name: str,
        data_type: ts.FieldType,
    ) -> FieldopData: ...

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


def _make_access_index_for_field(
    domain: FieldopDomain, data: FieldopData
) -> dace.subsets.Range:
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


def translate_as_fieldop(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for the ``as_fieldop`` builtin.

    The stencil expression is lowered to a single Python code string
    (via ``StreePythonCodegen``) and embedded in a ``TaskletNode`` inside
    a ``MapScope`` that ranges over the field domain.
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    fieldop_expr, fieldop_domain_expr = fun_node.args

    if cpm.is_call_to(fieldop_expr, "scan"):
        return translate_scan(node, ctx, sdfg_builder)

    if not isinstance(node.type, ts.FieldType):
        raise NotImplementedError("Unexpected 'as_fieldop' with tuple output in stree lowering.")

    # Parse the domain of the field operator.
    assert isinstance(fieldop_domain_expr.type, ts.DomainType)
    field_domain = get_field_domain(
        domain_utils.SymbolicDomain.from_expr(fieldop_domain_expr)
    )

    # Handle special case: deref with field argument (copy)
    if cpm.is_ref_to(fieldop_expr, "deref"):
        arg_type = node.args[0].type
        assert isinstance(arg_type, (ts.FieldType, ts.ScalarType))
        if isinstance(arg_type, ts.ScalarType) or arg_type.dims != node.type.dims:
            stencil_expr = im.lambda_("a")(im.deref("a"))
            stencil_expr.expr.type = node.type.dtype
        else:
            arg = sdfg_builder.visit(node.args[0], ctx=ctx)
            assert isinstance(arg, FieldopData)
            return ctx.copy_data(sdfg_builder, arg, domain=field_domain)
    elif isinstance(fieldop_expr, gtir.Lambda):
        stencil_expr = fieldop_expr
    else:
        raise NotImplementedError(
            f"Expression type '{type(fieldop_expr)}' not supported as argument to 'as_fieldop' node."
        )

    # Visit the arguments to be passed to the lambda expression.
    # Each argument is visited to obtain a FieldopData handle.
    field_args: dict[str, FieldopData] = {}
    for i, arg_expr in enumerate(node.args):
        arg = sdfg_builder.visit(arg_expr, ctx=ctx)
        if isinstance(arg, FieldopData):
            param_name = str(stencil_expr.params[i].id) if i < len(stencil_expr.params) else f"__arg{i}"
            field_args[param_name] = arg

    # Build the map over the field domain.
    if len(field_domain) == 0:
        map_range = {"__gt4py_zerodim": "0"}
    else:
        map_range = {
            get_map_variable(r.dim): f"{r.start}:{r.stop}"
            for r in field_domain
        }
    map_entry, _map_exit = sdfg_builder.add_map("fieldop", map_range)

    # Build the map_indices mapping (Dimension -> map variable name).
    map_indices: dict[gtx_common.Dimension, str] = {}
    for r in field_domain:
        map_indices[r.dim] = get_map_variable(r.dim)

    # Generate the tasklet code using StreePythonCodegen.
    expr_code, pre_statements = generate_tasklet_code(
        stencil_expr.expr,
        field_args=field_args,
        map_indices=map_indices,
        offset_provider_type=sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
        if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
        else ctx.root._offset_provider_type,
        args_map={p.id: node.args[i] for i, p in enumerate(stencil_expr.params) if i < len(node.args)},
    )

    # Build the full tasklet code: pre-statements + final assignment.
    tasklet_code = "\n".join(pre_statements) + f"\n__out = {expr_code}" if pre_statements else f"__out = {expr_code}"

    # Create the TaskletNode.
    # Input connectors: one per field argument (full arrays).
    # Output connector: __out (writes to the result array).
    input_connectors = set()
    in_memlets: dict[str, dace.Memlet] = {}
    for param_name, field_data in field_args.items():
        conn = f"__field_{param_name}"
        input_connectors.add(conn)
        in_memlets[make_tasklet_connector_for(conn)] = _make_full_array_memlet(
            field_data.name, ctx.root
        )

    # Allocate the result array.
    _field_dims, field_origin, field_shape = get_field_layout(field_domain)
    result_name, _result_desc = sdfg_builder.add_temp_array(ctx.root, field_shape, gtx_dace_args.as_dace_type(node.type.dtype) if isinstance(node.type.dtype, ts.ScalarType) else gtx_dace_args.as_dace_type(node.type.dtype.element_type))

    # Output memlet: single element at the map indices.
    output_subset = _make_access_index_for_field(field_domain, FieldopData(result_name, node.type, tuple(field_origin)))
    out_memlet = dace.Memlet(data=result_name, subset=output_subset)
    out_memlets = {make_tasklet_connector_for("__out"): out_memlet}

    # Create the Tasklet and TaskletNode.
    tasklet, _connector_mapping = sdfg_builder.add_tasklet(
        name="fieldop",
        inputs=input_connectors,
        outputs={"__out"},
        code=tasklet_code,
    )

    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets=in_memlets,
        out_memlets=out_memlets,
    )

    # Create the MapScope and add the TaskletNode as a child.
    map_scope = tn.MapScope(node=map_entry, children=[tasklet_node])
    ctx.current_scope.add_child(map_scope)

    return FieldopData(result_name, node.type, tuple(field_origin))


def translate_scan(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for a scan field operator.

    The scan is lowered as a ``ForScope`` inside a ``MapScope``.  The
    horizontal domain is mapped, and the vertical (column) dimension is
    iterated by the ``ForScope``.  The carry is read from the output array
    at the previous index: ``O[k-1] if k > start else A`` (forward) or
    ``O[k+1] if k < stop-1 else A`` (backward).
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, scan_domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # Parse the domain of the scan field operator.
    assert isinstance(scan_domain_expr.type, ts.DomainType)
    field_domain = get_field_domain(
        domain_utils.SymbolicDomain.from_expr(scan_domain_expr)
    )

    # Parse scan parameters.
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # params[0]: the lambda parameter to propagate the scan carry.
    str(stencil_expr.params[0].id)

    # params[1]: boolean flag for forward/backward scan.
    assert isinstance(scan_expr.args[1], gtir.Literal)
    scan_forward = scan_expr.args[1].value == "True"

    # params[2]: the expression that computes the value for scan initialization.
    init_expr = scan_expr.args[2]
    init_data = sdfg_builder.visit(init_expr, ctx=ctx)
    assert isinstance(init_data, FieldopData)
    init_value = init_data.name  # the name of the init data container or symbol

    # Find the column dimension.
    scan_dim_index = [sdfg_builder.is_column_axis(r.dim) for r in field_domain].index(True)
    scan_domain_range = field_domain[scan_dim_index]
    scan_dim = scan_domain_range.dim
    scan_loop_var = get_map_variable(scan_dim)

    # Determine the horizontal dimensions (all except the column axis).
    horizontal_dims = [r for r in field_domain if not sdfg_builder.is_column_axis(r.dim)]

    # Allocate the result array.
    field_dims, field_origin, field_shape = get_field_layout(field_domain)
    assert isinstance(node.type, ts.FieldType)
    if isinstance(node.type.dtype, ts.ScalarType):
        dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    else:
        assert isinstance(node.type.dtype, ts.ListType)
        assert isinstance(node.type.dtype.element_type, ts.ScalarType)
        dtype = gtx_dace_args.as_dace_type(node.type.dtype.element_type)

    result_name, _result_desc = sdfg_builder.add_temp_array(ctx.root, field_shape, dtype)

    # Build the MapScope over horizontal dimensions.
    if len(horizontal_dims) == 0:
        # 1D scan: no horizontal map needed.
        map_scope = None
        map_indices: dict[gtx_common.Dimension, str] = {}
    else:
        map_range = {
            get_map_variable(r.dim): f"{r.start}:{r.stop}"
            for r in horizontal_dims
        }
        map_entry, _map_exit = sdfg_builder.add_map("fieldop", map_range)
        map_indices = {
            r.dim: get_map_variable(r.dim) for r in horizontal_dims
        }
        map_scope = tn.MapScope(node=map_entry, children=[])
        ctx.current_scope.add_child(map_scope)

    # Add the scan dimension to map_indices (used by the code generator
    # to generate the carry read and output write).
    map_indices[scan_dim] = scan_loop_var

    # Visit the field arguments (excluding the carry, which is handled
    # specially as the output array at the previous index).
    field_args: dict[str, FieldopData] = {}
    for i, arg_expr in enumerate(node.args):
        arg = sdfg_builder.visit(arg_expr, ctx=ctx)
        if isinstance(arg, FieldopData):
            param_name = str(stencil_expr.params[i + 1].id) if i + 1 < len(stencil_expr.params) else f"__arg{i}"
            field_args[param_name] = arg

    # Build the carry expression.
    # Forward carry: O[k-1] if k > start else A
    # Backward carry: O[k+1] if k < stop-1 else A
    if scan_forward:
        carry_prev = f"{scan_loop_var} - 1"
        carry_cond = f"{scan_loop_var} > {scan_domain_range.start}"
    else:
        carry_prev = f"{scan_loop_var} + 1"
        carry_cond = f"{scan_loop_var} < {scan_domain_range.stop} - 1"

    # Build the array access for the carry: O[horizontal_indices, k-1]
    # (or O[horizontal_indices, k+1] for backward).
    carry_indices = []
    for dim, origin in zip(field_dims, field_origin, strict=True):
        map_var = map_indices.get(dim)
        assert map_var is not None, f"No map variable for dimension {dim}."
        if dim == scan_dim:
            carry_indices.append(f"({carry_prev}) - {origin}")
        else:
            carry_indices.append(f"{map_var} - {origin}")
    carry_expr = f"{result_name}[{', '.join(carry_indices)}] if {carry_cond} else {init_value}"

    # Generate the tasklet code using StreePythonCodegen.
    # The carry is passed as the first lambda parameter.
    {stencil_expr.params[0].id: gtir.SymRef(id=stencil_expr.params[0].id)}
    # We need to set up the args_map so that the carry parameter maps
    # to the carry expression. We do this by creating a temporary SymRef
    # that the code generator will resolve to the carry expression.
    # Actually, the code generator resolves through args_map, so we
    # need to add the carry expression to the args_map.
    # But args_map maps to gtir.Node, not to a code string. So we need
    # a different approach: we add the carry expression as a pre-statement
    # and use the variable name in the args_map.

    # Alternative approach: generate the stencil code with the carry
    # as a regular field argument, then replace the carry parameter
    # with the carry expression.

    # For now, let's generate the code by adding the carry to the
    # args_map as a special "symbolic" value that the code generator
    # can handle. We'll use a SymRef that the code generator will
    # resolve to the carry expression.

    # Actually, the simplest approach is to generate the code without
    # the carry (as if the carry is a regular variable), then prepend
    # the carry expression as a pre-statement.

    # Let's use a different approach: generate the full code including
    # the carry, by passing the carry expression through the args_map
    # as a pre-statement.

    # The cleanest approach: add the carry as a "field_arg" with a
    # special name, and have the code generator generate the carry
    # expression inline.

    # For now, let's just generate the code manually:
    # 1. Generate the stencil body (without the carry)
    # 2. Add the carry as a pre-statement
    # 3. Combine them

    # Generate the stencil body with the carry as a placeholder.
    # The carry is the first parameter of the stencil lambda.
    # We'll add it to args_map as a SymRef to itself, and then
    # replace it with the carry expression in the generated code.

    # Actually, the simplest approach is to add the carry to args_map
    # as a literal SymRef, and then have the code generator resolve
    # it to the carry expression.

    # Let me just set up the args_map properly:
    # - The carry parameter maps to a SymRef that the code generator
    #   will resolve to the carry expression.
    # - The other parameters map to their field arguments.

    # But the code generator's visit_SymRef resolves through args_map
    # by visiting the mapped expression. So if we map the carry parameter
    # to a SymRef with a special name, the code generator will return
    # that name.

    # The cleanest approach: add the carry expression as a pre-statement
    # that assigns it to a temporary variable, then use that variable
    # in the args_map.

    # Let me implement this:
    carry_var = "__gtir_scan_carry"
    pre_statements_scan = [f"{carry_var} = {carry_expr}"]

    # Set up the args_map: carry parameter -> SymRef(carry_var)
    # and other parameters -> their arguments.
    scan_args_map_final = {stencil_expr.params[0].id: gtir.SymRef(id=carry_var)}
    for i, p in enumerate(stencil_expr.params[1:], start=1):
        if i - 1 < len(node.args):
            scan_args_map_final[p.id] = node.args[i - 1]

    # Generate the tasklet code.
    expr_code, pre_statements_body = generate_tasklet_code(
        stencil_expr.expr,
        field_args=field_args,
        map_indices=map_indices,
        offset_provider_type=sdfg_builder.get_offset_provider_type.__self__.offset_provider_type
        if hasattr(sdfg_builder.get_offset_provider_type, "__self__")
        else ctx.root._offset_provider_type,
        args_map=scan_args_map_final,
    )

    # Combine all pre-statements and the final expression.
    all_pre_statements = pre_statements_scan + pre_statements_body
    tasklet_code = "\n".join(all_pre_statements) + f"\n__out = {expr_code}"

    # Create the TaskletNode.
    # Input connectors: the field args (full arrays) + the output array O (full array, for carry read).
    input_connectors = set()
    in_memlets: dict[str, dace.Memlet] = {}
    for param_name, field_data in field_args.items():
        conn = f"__field_{param_name}"
        input_connectors.add(conn)
        in_memlets[make_tasklet_connector_for(conn)] = _make_full_array_memlet(
            field_data.name, ctx.root
        )
    # Add the output array as an input (for the carry read).
    carry_conn = "__O_carry"
    input_connectors.add(carry_conn)
    in_memlets[make_tasklet_connector_for(carry_conn)] = _make_full_array_memlet(
        result_name, ctx.root
    )

    # Output memlet: single element at (horizontal_indices, k).
    output_indices = []
    for dim, origin in zip(field_dims, field_origin, strict=True):
        map_var = map_indices.get(dim)
        assert map_var is not None
        output_indices.append(f"{map_var} - {origin}")
    output_subset = dace.subsets.Range.from_string(",".join(output_indices))
    out_memlet = dace.Memlet(data=result_name, subset=output_subset)
    out_memlets = {make_tasklet_connector_for("__out"): out_memlet}

    # Create the Tasklet and TaskletNode.
    tasklet, _connector_mapping = sdfg_builder.add_tasklet(
        name="scan",
        inputs=input_connectors,
        outputs={"__out"},
        code=tasklet_code,
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

    return FieldopData(result_name, node.type, tuple(field_origin))


def translate_if(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for the ``if_`` builtin (statement level).

    Uses ``IfScope``/``ElseScope`` at the statement level.  Inside the stencil
    body, ``if_`` is handled by the code generator as a ternary expression.
    """
    assert cpm.is_call_to(node, "if_")
    assert len(node.args) == 3
    cond_expr, true_expr, false_expr = node.args

    # Generate the condition code.
    cond_code = get_source(cond_expr)

    # Visit the true and false branches.
    SubgraphContext(
        root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes
    )
    SubgraphContext(
        root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes
    )

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
    out_data = FieldopData(out_name, node.type, tuple(field_origin))

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
    if true_result is not None and isinstance(true_result, FieldopData):
        src_subset = _make_access_index_for_field(
            get_field_domain(true_expr.annex.domain), true_result
        )
        dst_subset = _make_access_index_for_field(field_domain, out_data)
        if_scope.add_child(
            tn.CopyNode(
                target=out_name,
                memlet=dace.Memlet(data=true_result.name, subset=src_subset, other_subset=dst_subset),
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
    if false_result is not None and isinstance(false_result, FieldopData):
        src_subset = _make_access_index_for_field(
            get_field_domain(false_expr.annex.domain), false_result
        )
        dst_subset = _make_access_index_for_field(field_domain, out_data)
        else_scope.add_child(
            tn.CopyNode(
                target=out_name,
                memlet=dace.Memlet(data=false_result.name, subset=src_subset, other_subset=dst_subset),
            )
        )

    return out_data


def translate_symbol_ref(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for a ``ir.SymRef`` node."""
    assert isinstance(node, gtir.SymRef)

    # Check if the symbol is unused (domain is NEVER).
    from gt4py.next.iterator.transforms import infer_domain
    node_domain = getattr(node.annex, "domain", infer_domain.DomainAccessDescriptor.UNKNOWN)
    if node_domain == infer_domain.DomainAccessDescriptor.NEVER:
        return None

    symbol_name = str(node.id)
    if symbol_name in ctx.data_nodes:
        return ctx.data_nodes[symbol_name]

    # Look up in root.containers and create a FieldopData.
    if symbol_name in ctx.root.containers:
        ctx.root.containers[symbol_name]
        # Determine the GT4Py type from the scope_symbols.
        gt_symbol_type = ctx.get_symbol_type(symbol_name)
        return sdfg_builder.make_field(symbol_name, gt_symbol_type)

    # If the symbol is a scalar symbol on root.symbols.
    if symbol_name in ctx.root.symbols:
        gt_symbol_type = ctx.get_symbol_type(symbol_name)
        return FieldopData(symbol_name, gt_symbol_type, origin=())

    raise ValueError(f"Symbol '{symbol_name}' not found in scope.")


def translate_scalar_expr(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates a tasklet for a scalar-valued expression."""
    assert isinstance(node, gtir.FunCall)
    assert isinstance(node.type, ts.ScalarType)

    # Generate the Python code for the scalar expression.
    scalar_code = get_source(node)

    # Create a temporary scalar for the result.
    result_name, _ = sdfg_builder.add_temp_scalar(ctx.root, gtx_dace_args.as_dace_type(node.type))

    # Create the TaskletNode.
    tasklet, connector_mapping = sdfg_builder.add_tasklet(
        name="scalar_expr",
        inputs=set(),
        outputs={"out"},
        code=f"out = {scalar_code}",
    )

    tn.TaskletNode(
        node=tasklet,
        in_memlets={},
        out_memlets={connector_mapping["out"]: dace.Memlet(data=result_name, subset="0")},
    )
    # Note: The TaskletNode needs to be added to the current scope.
    # But we can't add it here because we need to return it.
    # Actually, the TaskletNode is created by the caller (visit_FunCall).
    # Let me fix this by creating the TaskletNode here and adding it to the current scope.
    # Actually, looking at the existing code, the scalar_expr translator creates
    # the tasklet and adds it to the state. In the new code, I need to create
    # the TaskletNode and add it to the current scope.
    # Let me redo this properly.
    tasklet_node = tn.TaskletNode(
        node=tasklet,
        in_memlets={},
        out_memlets={connector_mapping["out"]: dace.Memlet(data=result_name, subset="0")},
    )
    ctx.current_scope.add_child(tasklet_node)

    return FieldopData(result_name, node.type, origin=())


def translate_make_tuple(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for ``make_tuple``."""
    assert cpm.is_call_to(node, "make_tuple")
    return tuple(sdfg_builder.visit(arg, ctx=ctx) for arg in node.args)


def translate_tuple_get(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for ``tuple_get``."""
    assert cpm.is_call_to(node, "tuple_get")
    assert len(node.args) == 2

    if not isinstance(node.args[0], gtir.Literal):
        raise ValueError("Tuple can only be subscripted with compile-time constants.")
    from gt4py.next.type_system import type_info as ti
    assert ti.is_integral(node.args[0].type)
    index = int(node.args[0].value)

    data_nodes = sdfg_builder.visit(node.args[1], ctx=ctx)
    if not isinstance(data_nodes, tuple):
        raise ValueError(f"Invalid tuple expression {node}.")
    return data_nodes[index]


def translate_concat_where(
    node: gtir.Node,
    ctx: SubgraphContext,
    sdfg_builder: SDFGBuilder,
) -> FieldopResult:
    """Generates the schedule-tree nodes for ``concat_where`` using CopyNode."""
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
    elif mask_domain.ranges[concat_dim].stop in infinity_literals:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_expr, lower_expr = node.args[1:]
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    # Allocate the output field.
    output_domain = get_field_domain(node.annex.domain)
    output_dims, output_origin, output_shape = get_field_layout(output_domain)
    assert output_dims == node.type.dims
    assert isinstance(node.type.dtype, ts.ScalarType)
    dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    output_name, output_desc = sdfg_builder.add_temp_array(ctx.root, output_shape, dtype)

    # Translate the lower branch and create a CopyNode.
    lower_data = sdfg_builder.visit(lower_expr, ctx=ctx)
    assert isinstance(lower_data, FieldopData)
    lower_domain = get_field_domain(lower_expr.annex.domain)
    lower_subset = _make_access_index_for_field(lower_domain, lower_data)
    # Compute the output subset for the lower branch.
    bound_symbolic = get_symbolic(bound_expr)
    lower_output_subset_parts = []
    for dim, size, src_origin, dst_origin in zip(
        output_dims, output_desc.shape, lower_data.origin, output_origin, strict=True
    ):
        if dim == concat_dim:
            lower_output_subset_parts.append((
                dst_origin - src_origin,
                dst_origin - src_origin + bound_symbolic - src_origin - 1,
                1,
            ))
        else:
            lower_output_subset_parts.append((0, size - 1, 1))
    lower_output_subset = dace_subsets.Range(lower_output_subset_parts)
    ctx.current_scope.add_child(
        tn.CopyNode(
            target=output_name,
            memlet=dace.Memlet(data=lower_data.name, subset=lower_subset, other_subset=lower_output_subset),
        )
    )

    # Translate the upper branch and create a CopyNode.
    upper_data = sdfg_builder.visit(upper_expr, ctx=ctx)
    assert isinstance(upper_data, FieldopData)
    upper_domain = get_field_domain(upper_expr.annex.domain)
    upper_subset = _make_access_index_for_field(upper_domain, upper_data)
    upper_output_subset_parts = []
    for dim, size, _src_origin, dst_origin in zip(
        output_dims, output_desc.shape, upper_data.origin, output_origin, strict=True
    ):
        if dim == concat_dim:
            upper_output_subset_parts.append((
                bound_symbolic - dst_origin,
                bound_symbolic - dst_origin + size - 1,
                1,
            ))
        else:
            upper_output_subset_parts.append((0, size - 1, 1))
    upper_output_subset = dace_subsets.Range(upper_output_subset_parts)
    ctx.current_scope.add_child(
        tn.CopyNode(
            target=output_name,
            memlet=dace.Memlet(data=upper_data.name, subset=upper_subset, other_subset=upper_output_subset),
        )
    )

    return FieldopData(output_name, node.type, tuple(output_origin))


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

    def __init__(self, offset_provider_type: gtx_common.OffsetProviderType, column_axis: Optional[gtx_common.Dimension] = None):
        self.offset_provider_type = offset_provider_type
        self.column_axis = column_axis
        self.uids = gtx_utils.IDGeneratorPool()

    def get_offset_provider_type(self, offset: str) -> gtx_common.OffsetProviderTypeElem:
        return gtx_common.get_offset_type(self.offset_provider_type, offset)

    def make_field(
        self,
        name: str,
        data_type: ts.FieldType,
    ) -> FieldopData:
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
        field_origin = tuple(
            gtx_dace_args.range_start_symbol(name, dim) for dim in field_type.dims
        )
        return FieldopData(name, field_type, field_origin)

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
        root._offset_provider_type = self.offset_provider_type  # type: ignore[attr-defined]

        # Start: children are added directly to root (no GBlock —
        # from_schedule_tree does not support GBlock yet).
        # Pre-allocate storage for all program parameters.
        scope_symbols = {str(p.id): p.type for p in node.params if isinstance(p.type, ts.DataType)}
        sdfg_arg_names = self._add_sdfg_params(
            root, node.params, symbolic_params=None, use_transient_storage=False
        )
        root.arg_names = sdfg_arg_names

        # Visit one statement at a time.
        data_nodes: dict[str, FieldopData | None] = {}
        for sym_name, sym_type in scope_symbols.items():
            if isinstance(sym_type, ts.ScalarType) and sym_name in root.symbols:
                data_nodes[sym_name] = FieldopData(sym_name, sym_type, origin=())
            elif sym_name in root.containers:
                data_nodes[sym_name] = self.make_field(sym_name, sym_type)
            pass  # else: symbol not in scope, skip

        for i, stmt in enumerate(node.body):
            # Insert a StateBoundaryNode between statements to ensure
            # they are lowered to separate SDFG states.
            if i > 0:
                root.add_child(tn.StateBoundaryNode())
            ctx = SubgraphContext(
                root=root, current_scope=root, data_nodes=data_nodes
            )
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

    def visit_SetAt(
        self, stmt: gtir.SetAt, ctx: SubgraphContext
    ) -> None:
        """Visits a ``SetAt`` statement and writes the result to the target."""
        # Visit the field operator expression.
        source_tree = self._visit_expression(stmt.expr, ctx)

        # Visit the target expression.
        target_tree = self._visit_expression(
            stmt.target,
            ctx=SubgraphContext(
                root=ctx.root, current_scope=ctx.current_scope, data_nodes=ctx.data_nodes
            ),
            use_temp=False,
        )

        # Write the result to the target using CopyNode(s).
        domain = extract_target_domain(stmt.domain)

        def _write_target(
            source: FieldopData | None,
            target: FieldopData | None,
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
                    memlet=dace.Memlet(data=source.name, subset=src_subset, other_subset=dst_subset),
                )
            )

        gtx_utils.tree_map(_write_target)(source_tree, target_tree, domain)

    def _visit_expression(
        self,
        node: gtir.Expr,
        ctx: SubgraphContext,
        use_temp: bool = True,
    ) -> FieldopResult:
        """Specialized visit method for fieldview expressions."""
        result = self.visit(node, ctx=ctx)

        if use_temp and result is not None:
            # Copy the full shape of global data to temporary storage.
            def _maybe_copy(x: FieldopData | None) -> FieldopData | None:
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
    ) -> FieldopResult:
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

        # Name-matched builtins.
        if isinstance(node.fun, gtir.SymRef):
            name = str(node.fun.id)
            if name == "if_":
                return translate_if(node, ctx, self)
            if name == "make_tuple":
                return translate_make_tuple(node, ctx, self)
            if name == "tuple_get":
                return translate_tuple_get(node, ctx, self)
            if name == "concat_where":
                return translate_concat_where(node, ctx, self)
            if name == "index":
                # TODO: implement index for stree lowering
                raise NotImplementedError("'index' builtin not yet implemented in stree lowering.")

        # Fallback: scalar-valued expressions (e.g. math builtins like plus, cast_).
        if isinstance(node.type, ts.ScalarType):
            return translate_scalar_expr(node, ctx, self)

        raise NotImplementedError(f"Unexpected 'FunCall' expression ({node}).")

    def visit_Literal(
        self,
        node: gtir.Literal,
        ctx: SubgraphContext,
    ) -> FieldopResult:
        raise ValueError(f"Unexpected 'Literal' node ({node}).")

    def visit_SymRef(
        self,
        node: gtir.SymRef,
        ctx: SubgraphContext,
    ) -> FieldopResult:
        return translate_symbol_ref(node, ctx, self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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

    ir = inline_literal.InlineLiteral().visit(ir)
    ir = gtir_type_inference.infer(ir, offset_provider_type=offset_provider_type)
    ir = ir_prune_casts.PruneCasts().visit(ir)
    ir = replace_invalid_symbols(ir)

    visitor = GTIRToScheduleTree(offset_provider_type, column_axis)
    stree = visitor.visit(ir)
    assert isinstance(stree, tn.ScheduleTreeRoot)

    return stree
