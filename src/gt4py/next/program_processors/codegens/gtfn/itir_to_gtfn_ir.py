# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

import dataclasses
from typing import Any, ClassVar, Iterable, Optional, Type, Union

import gt4py.eve as eve
from gt4py.eve.concepts import SymbolName
from gt4py.eve.utils import UIDGenerator
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms.global_tmps import FencilWithTemporaries
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    CartesianDomain,
    CastExpr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    IntegralConstant,
    Lambda,
    Literal,
    OffsetLiteral,
    Scan,
    ScanExecution,
    ScanPassDefinition,
    SidComposite,
    StencilExecution,
    TagDefinition,
    TaggedValues,
    TemporaryAllocation,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Node, Sym, SymRef


def pytype_to_cpptype(t: str):
    try:
        return {
            "float32": "float",
            "float64": "double",
            "int32": "std::int32_t",
            "int64": "std::int64_t",
            "bool": "bool",
            "axis_literal": None,  # TODO: domain?
        }[t]
    except KeyError:
        raise TypeError(f"Unsupported type '{t}'.") from None


_vertical_dimension = "gtfn::unstructured::dim::vertical"
_horizontal_dimension = "gtfn::unstructured::dim::horizontal"


def _get_domains(closures: Iterable[itir.StencilClosure]) -> Iterable[itir.FunCall]:
    return (c.domain for c in closures)


def _extract_grid_type(domain: itir.FunCall) -> common.GridType:
    if domain.fun == itir.SymRef(id="cartesian_domain"):
        return common.GridType.CARTESIAN
    else:
        assert domain.fun == itir.SymRef(id="unstructured_domain")
        return common.GridType.UNSTRUCTURED


def _get_gridtype(closures: list[itir.StencilClosure]) -> common.GridType:
    domains = _get_domains(closures)
    grid_types = {_extract_grid_type(d) for d in domains}
    if len(grid_types) != 1:
        raise ValueError(
            f"Found 'StencilClosures' with more than one 'GridType': '{grid_types}'. This is currently not supported."
        )
    return grid_types.pop()


def _name_from_named_range(named_range_call: itir.FunCall) -> str:
    assert isinstance(named_range_call, itir.FunCall) and named_range_call.fun == itir.SymRef(
        id="named_range"
    )
    assert isinstance(named_range_call.args[0], itir.AxisLiteral)
    return named_range_call.args[0].value


def _collect_dimensions_from_domain(
    closures: Iterable[itir.StencilClosure],
) -> dict[str, TagDefinition]:
    domains = _get_domains(closures)
    offset_definitions = {}
    for domain in domains:
        if domain.fun == itir.SymRef(id="cartesian_domain"):
            for nr in domain.args:
                assert isinstance(nr, itir.FunCall)
                dim_name = _name_from_named_range(nr)
                offset_definitions[dim_name] = TagDefinition(name=Sym(id=dim_name))
        elif domain.fun == itir.SymRef(id="unstructured_domain"):
            if len(domain.args) > 2:
                raise ValueError("Unstructured_domain must not have more than 2 arguments.")
            if len(domain.args) > 0:
                horizontal_range = domain.args[0]
                assert isinstance(horizontal_range, itir.FunCall)
                horizontal_name = _name_from_named_range(horizontal_range)
                offset_definitions[horizontal_name] = TagDefinition(
                    name=Sym(id=horizontal_name), alias=_horizontal_dimension
                )
            if len(domain.args) > 1:
                vertical_range = domain.args[1]
                assert isinstance(vertical_range, itir.FunCall)
                vertical_name = _name_from_named_range(vertical_range)
                offset_definitions[vertical_name] = TagDefinition(
                    name=Sym(id=vertical_name), alias=_vertical_dimension
                )
        else:
            raise AssertionError(
                "Expected either a call to 'cartesian_domain' or to 'unstructured_domain'."
            )
    return offset_definitions


def _collect_offset_definitions(
    node: itir.Node,
    grid_type: common.GridType,
    offset_provider: dict[str, common.Dimension | common.Connectivity],
):
    used_offset_tags: set[itir.OffsetLiteral] = (
        node.walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .filter(lambda offset_literal: isinstance(offset_literal.value, str))
        .getattr("value")
    ).to_set()
    if not used_offset_tags.issubset(set(offset_provider.keys())):
        raise AssertionError("ITIR contains an offset tag without a corresponding offset provider.")
    offset_definitions = {}

    for offset_name, dim_or_connectivity in offset_provider.items():
        if isinstance(dim_or_connectivity, common.Dimension):
            dim: common.Dimension = dim_or_connectivity
            if grid_type == common.GridType.CARTESIAN:
                # create alias from offset to dimension
                offset_definitions[dim.value] = TagDefinition(name=Sym(id=dim.value))
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
            else:
                assert grid_type == common.GridType.UNSTRUCTURED
                if not dim.kind == common.DimensionKind.VERTICAL:
                    raise ValueError(
                        "Mapping an offset to a horizontal dimension in unstructured is not allowed."
                    )
                # create alias from vertical offset to vertical dimension
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
        elif isinstance(dim_or_connectivity, common.Connectivity):
            assert grid_type == common.GridType.UNSTRUCTURED
            offset_definitions[offset_name] = TagDefinition(name=Sym(id=offset_name))

            connectivity: common.Connectivity = dim_or_connectivity
            for dim in [
                connectivity.origin_axis,
                connectivity.neighbor_axis,
            ]:
                if dim.kind != common.DimensionKind.HORIZONTAL:
                    raise NotImplementedError()
                offset_definitions[dim.value] = TagDefinition(
                    name=Sym(id=dim.value), alias=_horizontal_dimension
                )
        else:
            raise AssertionError(
                "Elements of offset provider need to be either 'Dimension' or 'Connectivity'."
            )
    return offset_definitions


def _literal_as_integral_constant(node: itir.Literal) -> IntegralConstant:
    assert node.type in itir.INTEGER_BUILTINS
    return IntegralConstant(value=int(node.value))


@dataclasses.dataclass(frozen=True)
class GTFN_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    _binary_op_map: ClassVar[dict[str, str]] = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "not_eq": "!=",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "and_": "&&",
        "or_": "||",
        "xor_": "^",
        "mod": "%",
    }
    _unary_op_map: ClassVar[dict[str, str]] = {"not_": "!"}

    offset_provider: dict
    column_axis: Optional[common.Dimension]
    grid_type: common.GridType

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: UIDGenerator = dataclasses.field(init=False, repr=False, default_factory=UIDGenerator)

    @classmethod
    def apply(
        cls,
        node: itir.FencilDefinition | FencilWithTemporaries,
        *,
        offset_provider: dict,
        column_axis: Optional[common.Dimension],
    ):
        if isinstance(node, FencilWithTemporaries):
            fencil_definition = node.fencil
        elif isinstance(node, itir.FencilDefinition):
            fencil_definition = node
        else:
            raise TypeError(
                f"Expected a 'FencilDefinition' or 'FencilWithTemporaries', got '{type(node).__name__}'."
            )

        grid_type = _get_gridtype(fencil_definition.closures)
        return cls(
            offset_provider=offset_provider, column_axis=column_axis, grid_type=grid_type
        ).visit(node)

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(
        self,
        node: itir.SymRef,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> SymRef:
        if force_function_extraction and node.id == "deref":
            assert extracted_functions is not None
            fun_id = self.uids.sequential_id(prefix="_fun")
            fun_def = FunctionDefinition(
                id=fun_id,
                params=[Sym(id="x")],
                expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="x")]),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return SymRef(id=node.id)

    def visit_Lambda(
        self,
        node: itir.Lambda,
        *,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> Union[SymRef, Lambda]:
        if force_function_extraction:
            assert extracted_functions is not None
            fun_id = self.uids.sequential_id(prefix="_fun")
            fun_def = FunctionDefinition(
                id=fun_id,
                params=self.visit(node.params, **kwargs),
                expr=self.visit(node.expr, **kwargs),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return Lambda(
            params=self.visit(node.params, **kwargs), expr=self.visit(node.expr, **kwargs)
        )

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type)

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        return OffsetLiteral(value=node.value)

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type="axis_literal")

    def _make_domain(self, node: itir.FunCall):
        tags = []
        sizes = []
        offsets = []
        for named_range in node.args:
            if not (
                isinstance(named_range, itir.FunCall)
                and named_range.fun == itir.SymRef(id="named_range")
            ):
                raise ValueError("Arguments to 'domain' need to be calls to 'named_range'.")
            tags.append(self.visit(named_range.args[0]))
            sizes.append(
                BinaryExpr(
                    op="-", lhs=self.visit(named_range.args[2]), rhs=self.visit(named_range.args[1])
                )
            )
            offsets.append(self.visit(named_range.args[1]))
        return TaggedValues(tags=tags, values=sizes), TaggedValues(tags=tags, values=offsets)

    @staticmethod
    def _collect_offset_or_axis_node(
        node_type: Type, tree: eve.Node | Iterable[eve.Node]
    ) -> set[str]:
        if not isinstance(tree, Iterable):
            tree = [tree]
        result = set()
        for n in tree:
            result.update(
                n.pre_walk_values()
                .if_isinstance(node_type)
                .getattr("value")
                .if_isinstance(str)
                .to_set()
            )
        return result

    def _visit_if_(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert len(node.args) == 3
        return TernaryExpr(
            cond=self.visit(node.args[0], **kwargs),
            true_expr=self.visit(node.args[1], **kwargs),
            false_expr=self.visit(node.args[2], **kwargs),
        )

    def _visit_cast_(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert len(node.args) == 2
        return CastExpr(
            obj_expr=self.visit(node.args[0], **kwargs),
            new_dtype=self.visit(node.args[1], **kwargs),
        )

    def _visit_tuple_get(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert isinstance(node.args[0], itir.Literal)
        return FunCall(
            fun=SymRef(id="tuple_get"),
            args=[
                _literal_as_integral_constant(node.args[0]),
                self.visit(node.args[1]),
            ],
        )

    def _visit_list_get(self, node: itir.FunCall, **kwargs: Any) -> Node:
        # should only reach this for the case of an external sparse field
        tuple_idx = (
            _literal_as_integral_constant(node.args[0])
            if isinstance(node.args[0], itir.Literal)
            else self.visit(
                node.args[0]
            )  # from unroll_reduce we get a `SymRef` which is refering to an `OffsetLiteral` which is lowered to integral_constant
        )
        return FunCall(
            fun=SymRef(id="tuple_get"),
            args=[
                tuple_idx,
                self.visit(node.args[1]),
            ],
        )

    def _visit_cartesian_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        return CartesianDomain(tagged_sizes=sizes, tagged_offsets=domain_offsets)

    def _visit_unstructured_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        connectivities = []
        if "stencil" in kwargs:
            shift_offsets = self._collect_offset_or_axis_node(itir.OffsetLiteral, kwargs["stencil"])
            for o in shift_offsets:
                if o in self.offset_provider and isinstance(
                    self.offset_provider[o], common.Connectivity
                ):
                    connectivities.append(SymRef(id=o))
        return UnstructuredDomain(
            tagged_sizes=sizes,
            tagged_offsets=domain_offsets,
            connectivities=connectivities,
        )

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Node:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(
                    op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0], **kwargs)
                )
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0], **kwargs),
                    rhs=self.visit(node.args[1], **kwargs),
                )
            elif hasattr(self, visit_method := f"_visit_{node.fun.id}"):
                # special handling of applied builtins is handled in `_visit_<builtin>`
                return getattr(self, visit_method)(node, **kwargs)
            elif node.fun.id == "shift":
                raise ValueError("Unapplied shift call not supported: '{node}'.")
            elif node.fun.id == "scan":
                raise ValueError("Scans are only supported at the top level of a stencil closure.")
        if isinstance(node.fun, itir.FunCall):
            if node.fun.fun == itir.SymRef(id="shift"):
                assert len(node.args) == 1
                return FunCall(
                    fun=self.visit(node.fun.fun, **kwargs),
                    args=self.visit(node.args, **kwargs) + self.visit(node.fun.args, **kwargs),
                )
        return FunCall(fun=self.visit(node.fun, **kwargs), args=self.visit(node.args, **kwargs))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            expr=self.visit(node.expr, **kwargs),
        )

    @staticmethod
    def _is_scan(node: itir.Node):
        return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")

    def _visit_output_argument(self, node: itir.Expr):
        if isinstance(node, itir.SymRef):
            return self.visit(node)
        elif isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="make_tuple"):
            return SidComposite(values=[self._visit_output_argument(v) for v in node.args])
        raise ValueError("Expected 'SymRef' or 'make_tuple' in output argument.")

    @staticmethod
    def _bool_from_literal(node: itir.Node):
        assert isinstance(node, itir.Literal)
        assert node.type == "bool" and node.value in ("True", "False")
        return node.value == "True"

    def visit_StencilClosure(
        self, node: itir.StencilClosure, extracted_functions: list, **kwargs: Any
    ) -> Union[ScanExecution, StencilExecution]:
        backend = Backend(domain=self.visit(node.domain, stencil=node.stencil, **kwargs))
        if self._is_scan(node.stencil):
            scan_id = self.uids.sequential_id(prefix="_scan")
            assert isinstance(node.stencil, itir.FunCall)
            scan_lambda = self.visit(node.stencil.args[0], **kwargs)
            forward = self._bool_from_literal(node.stencil.args[1])
            scan_def = ScanPassDefinition(
                id=scan_id, params=scan_lambda.params, expr=scan_lambda.expr, forward=forward
            )
            extracted_functions.append(scan_def)
            scan = Scan(
                function=SymRef(id=scan_id),
                output=0,
                inputs=[i + 1 for i, _ in enumerate(node.inputs)],
                init=self.visit(node.stencil.args[2], **kwargs),
            )
            column_axis = self.column_axis
            assert isinstance(column_axis, common.Dimension)
            return ScanExecution(
                backend=backend,
                scans=[scan],
                args=[self._visit_output_argument(node.output), *self.visit(node.inputs)],
                axis=SymRef(id=column_axis.value),
            )
        return StencilExecution(
            stencil=self.visit(
                node.stencil,
                force_function_extraction=True,
                extracted_functions=extracted_functions,
                **kwargs,
            ),
            output=self._visit_output_argument(node.output),
            inputs=self.visit(node.inputs, **kwargs),
            backend=backend,
        )

    @staticmethod
    def _merge_scans(
        executions: list[Union[StencilExecution, ScanExecution]],
    ) -> list[Union[StencilExecution, ScanExecution]]:
        def merge(a: ScanExecution, b: ScanExecution) -> ScanExecution:
            assert a.backend == b.backend
            assert a.axis == b.axis

            index_map = dict[int, int]()
            compacted_b_args = list[Expr]()
            for b_idx, b_arg in enumerate(b.args):
                try:
                    a_idx = a.args.index(b_arg)
                    index_map[b_idx] = a_idx
                except ValueError:
                    index_map[b_idx] = len(a.args) + len(compacted_b_args)
                    compacted_b_args.append(b_arg)

            def remap_args(s: Scan) -> Scan:
                return Scan(
                    function=s.function,
                    output=index_map[s.output],
                    inputs=[index_map[i] for i in s.inputs],
                    init=s.init,
                )

            return ScanExecution(
                backend=a.backend,
                scans=a.scans + [remap_args(s) for s in b.scans],
                args=a.args + compacted_b_args,
                axis=a.axis,
            )

        res = executions[:1]
        for execution in executions[1:]:
            if (
                isinstance(execution, ScanExecution)
                and isinstance(res[-1], ScanExecution)
                and execution.backend == res[-1].backend
            ):
                res[-1] = merge(res[-1], execution)
            else:
                res.append(execution)
        return res

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, **kwargs: Any
    ) -> FencilDefinition:
        extracted_functions: list[Union[FunctionDefinition, ScanPassDefinition]] = []
        executions = self.visit(
            node.closures,
            extracted_functions=extracted_functions,
        )
        executions = self._merge_scans(executions)
        function_definitions = self.visit(node.function_definitions) + extracted_functions
        offset_definitions = {
            **_collect_dimensions_from_domain(node.closures),
            **_collect_offset_definitions(node, self.grid_type, self.offset_provider),
        }
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=self.grid_type,
            offset_definitions=list(offset_definitions.values()),
            function_definitions=function_definitions,
            temporaries=[],
        )

    def visit_Temporary(self, node, *, params: list, **kwargs) -> TemporaryAllocation:
        def dtype_to_cpp(x):
            if isinstance(x, int):
                return f"std::remove_const_t<::gridtools::sid::element_type<decltype({params[x]})>>"
            if isinstance(x, tuple):
                return "::gridtools::tuple<" + ", ".join(dtype_to_cpp(i) for i in x) + ">"
            assert isinstance(x, str)
            return pytype_to_cpptype(x)

        return TemporaryAllocation(
            id=node.id, dtype=dtype_to_cpp(node.dtype), domain=self.visit(node.domain, **kwargs)
        )

    def visit_FencilWithTemporaries(self, node, **kwargs) -> FencilDefinition:
        fencil = self.visit(node.fencil, **kwargs)
        return FencilDefinition(
            id=fencil.id,
            params=self.visit(node.params),
            executions=fencil.executions,
            grid_type=fencil.grid_type,
            offset_definitions=fencil.offset_definitions,
            function_definitions=fencil.function_definitions,
            temporaries=self.visit(node.tmps, params=[p.id for p in node.params]),
        )
