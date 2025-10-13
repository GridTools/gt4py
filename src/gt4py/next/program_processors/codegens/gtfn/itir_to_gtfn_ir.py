# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
from typing import Any, Callable, ClassVar, Iterable, Optional, Type, TypeGuard, Union

import gt4py.eve as eve
from gt4py.eve import utils as eve_utils
from gt4py.eve.concepts import SymbolName
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_utils_misc,
)
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.otf import cpp_utils
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    CartesianDomain,
    CastExpr,
    FunCall,
    FunctionDefinition,
    IfStmt,
    IntegralConstant,
    Lambda,
    Literal,
    OffsetLiteral,
    Program,
    Scan,
    ScanExecution,
    ScanPassDefinition,
    SidComposite,
    SidFromScalar,
    StencilExecution,
    TagDefinition,
    TaggedValues,
    TemporaryAllocation,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Node, Sym, SymRef
from gt4py.next.type_system import type_info, type_specifications as ts


_vertical_dimension = "gtfn::unstructured::dim::vertical"
_horizontal_dimension = "gtfn::unstructured::dim::horizontal"


def _is_tuple_of_ref_or_literal(expr: itir.Expr) -> bool:
    if (
        isinstance(expr, itir.FunCall)
        and isinstance(expr.fun, itir.SymRef)
        and expr.fun.id == "tuple_get"
        and len(expr.args) == 2
        and _is_tuple_of_ref_or_literal(expr.args[1])
    ):
        return True
    if (
        isinstance(expr, itir.FunCall)
        and isinstance(expr.fun, itir.SymRef)
        and expr.fun.id == "make_tuple"
        and all(_is_tuple_of_ref_or_literal(arg) for arg in expr.args)
    ):
        return True
    if isinstance(expr, (itir.SymRef, itir.Literal)):
        return True
    return False


def _get_domains(nodes: Iterable[itir.Stmt]) -> Iterable[itir.FunCall]:
    result = set()
    for node in nodes:
        result |= node.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    return result


def _name_from_named_range(named_range_call: itir.FunCall) -> str:
    assert isinstance(named_range_call, itir.FunCall) and named_range_call.fun == itir.SymRef(
        id="named_range"
    )
    assert isinstance(named_range_call.args[0], itir.AxisLiteral)
    return named_range_call.args[0].value


def _collect_dimensions_from_domain(
    body: Iterable[itir.Stmt],
) -> dict[str, TagDefinition]:
    domains = _get_domains(body)
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
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, TagDefinition]:
    used_offset_tags: set[str] = (
        node.walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .filter(lambda offset_literal: isinstance(offset_literal.value, str))
        .getattr("value")
    ).to_set()
    # implicit offsets don't occur in the `offset_provider_type`, get them from the used offset tags
    offset_provider_type = {
        offset_name: common.get_offset_type(offset_provider_type, offset_name)
        for offset_name in used_offset_tags
    } | {**offset_provider_type}
    offset_definitions = {}

    for offset_name, dim_or_connectivity_type in offset_provider_type.items():
        if isinstance(dim_or_connectivity_type, common.Dimension):
            dim: common.Dimension = dim_or_connectivity_type
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
                offset_definitions[dim.value] = TagDefinition(
                    name=Sym(id=dim.value), alias=_vertical_dimension
                )
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
        elif isinstance(
            connectivity_type := dim_or_connectivity_type, common.NeighborConnectivityType
        ):
            assert grid_type == common.GridType.UNSTRUCTURED
            offset_definitions[offset_name] = TagDefinition(name=Sym(id=offset_name))
            if offset_name != connectivity_type.neighbor_dim.value:
                offset_definitions[connectivity_type.neighbor_dim.value] = TagDefinition(
                    name=Sym(id=connectivity_type.neighbor_dim.value)
                )

            for dim in [connectivity_type.source_dim, connectivity_type.codomain]:
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
    assert type_info.is_integer(node.type)
    return IntegralConstant(value=int(node.value))


def _is_scan(node: itir.Node) -> TypeGuard[itir.FunCall]:
    return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")


def _bool_from_literal(node: itir.Node) -> bool:
    assert isinstance(node, itir.Literal)
    assert type_info.is_logical(node.type) and node.value in ("True", "False")
    return node.value == "True"


class _CannonicalizeUnstructuredDomain(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        if node.fun == itir.SymRef(id="unstructured_domain"):
            # for no good reason, the domain arguments for unstructured need to be in order (horizontal, vertical)
            assert isinstance(node.args[0], itir.FunCall)
            first_axis_literal = node.args[0].args[0]
            assert isinstance(first_axis_literal, itir.AxisLiteral)
            if first_axis_literal.kind == itir.DimensionKind.VERTICAL:
                assert len(node.args) == 2
                assert isinstance(node.args[1], itir.FunCall)
                assert isinstance(node.args[1].args[0], itir.AxisLiteral)
                assert node.args[1].args[0].kind == itir.DimensionKind.HORIZONTAL
                return itir.FunCall(fun=node.fun, args=[node.args[1], node.args[0]])
        return node

    @classmethod
    def apply(
        cls,
        node: itir.Program,
    ) -> itir.Program:
        if not isinstance(node, itir.Program):
            raise TypeError(f"Expected a 'Program', got '{type(node).__name__}'.")

        return cls().visit(node)


def _process_elements(
    process_func: Callable[..., Expr],
    obj: Expr,
    type_: ts.TypeSpec,
    *,
    tuple_constructor: Callable[..., Expr] = lambda *elements: FunCall(
        fun=SymRef(id="make_tuple"), args=list(elements)
    ),
) -> Expr:
    """
    Recursively applies a processing function to all primitive constituents of a tuple.

    Be aware that this function duplicates the `obj` expression and should hence be used with care.

    Arguments:
        process_func: A callable that takes a gtfn_ir.Expr representing a leaf-element of `obj`.
        obj: The object whose elements are to be transformed.
        type_: A type with the same structure as the elements of `obj`.
        tuple_constructor: By default all transformed tuple elements are just put in a tuple again.
            This can be customized by passing a different Callable.
    """
    assert isinstance(type_, ts.TypeSpec)

    def _gen_constituent_expr(el_type: ts.ScalarType | ts.FieldType, path: tuple[int, ...]) -> Expr:
        # construct expression for the currently processed element
        el = functools.reduce(
            lambda cur_expr, i: FunCall(
                fun=SymRef(id="tuple_get"), args=[IntegralConstant(value=i), cur_expr]
            ),
            path,
            obj,
        )
        return process_func(el, el_type)

    result = type_info.apply_to_primitive_constituents(
        _gen_constituent_expr,
        type_,
        with_path_arg=True,
        tuple_constructor=tuple_constructor,
    )
    return result


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

    offset_provider_type: common.OffsetProviderType
    column_axis: Optional[common.Dimension]
    grid_type: common.GridType

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: eve_utils.UIDGenerator = dataclasses.field(
        init=False, repr=False, default_factory=eve_utils.UIDGenerator
    )

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        offset_provider_type: common.OffsetProviderType,
        column_axis: Optional[common.Dimension],
    ) -> Program:
        if not isinstance(node, itir.Program):
            raise TypeError(f"Expected a 'Program', got '{type(node).__name__}'.")

        node = itir_type_inference.infer(node, offset_provider_type=offset_provider_type)
        grid_type = ir_utils_misc.grid_type_from_program(node)
        if grid_type == common.GridType.UNSTRUCTURED:
            node = _CannonicalizeUnstructuredDomain.apply(node)
        return cls(
            offset_provider_type=offset_provider_type, column_axis=column_axis, grid_type=grid_type
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
        return Literal(value=node.value, type=node.type.kind.name.lower())

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        return OffsetLiteral(value=node.value)

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type="axis_literal")

    def _make_domain(self, node: itir.FunCall) -> tuple[TaggedValues, TaggedValues]:
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
            args=[_literal_as_integral_constant(node.args[0]), self.visit(node.args[1])],
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
        return FunCall(fun=SymRef(id="tuple_get"), args=[tuple_idx, self.visit(node.args[1])])

    def _visit_cartesian_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        return CartesianDomain(tagged_sizes=sizes, tagged_offsets=domain_offsets)

    def _visit_unstructured_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        connectivities = []
        if "stencil" in kwargs:
            shift_offsets = self._collect_offset_or_axis_node(itir.OffsetLiteral, kwargs["stencil"])
            for o in shift_offsets:
                if o in self.offset_provider_type and isinstance(
                    common.get_offset_type(self.offset_provider_type, o),
                    common.NeighborConnectivityType,
                ):
                    connectivities.append(SymRef(id=o))
        return UnstructuredDomain(
            tagged_sizes=sizes, tagged_offsets=domain_offsets, connectivities=connectivities
        )

    def _visit_get_domain_range(self, node: itir.FunCall, **kwargs: Any) -> Node:
        field, dim = node.args

        return FunCall(
            fun=SymRef(id="get_domain_range"),
            args=[self.visit(field, **kwargs), self.visit(dim, **kwargs)],
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

    def _visit_output_argument(self, node: itir.Expr) -> SidComposite | SymRef:
        lowered_output = self.visit(node)

        # just a sanity check, identity function otherwise
        def check_el_type(el_expr: Expr, el_type: ts.ScalarType | ts.FieldType) -> Expr:
            assert isinstance(el_type, ts.FieldType)
            return el_expr

        assert isinstance(node.type, ts.TypeSpec)
        lowered_output_as_sid = _process_elements(
            check_el_type,
            lowered_output,
            node.type,
            tuple_constructor=lambda *elements: SidComposite(values=list(elements)),
        )

        assert isinstance(lowered_output_as_sid, (SidComposite, SymRef))
        return lowered_output_as_sid

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

    def visit_Stmt(self, node: itir.Stmt, **kwargs: Any) -> None:
        raise AssertionError("All Stmts need to be handled explicitly.")

    def visit_IfStmt(self, node: itir.IfStmt, **kwargs: Any) -> IfStmt:
        return IfStmt(
            cond=self.visit(node.cond, **kwargs),
            true_branch=self.visit(node.true_branch, **kwargs),
            false_branch=self.visit(node.false_branch, **kwargs),
        )

    def visit_SetAt(
        self, node: itir.SetAt, *, extracted_functions: list, **kwargs: Any
    ) -> Union[StencilExecution, ScanExecution]:
        if _is_tuple_of_ref_or_literal(node.expr):
            node.expr = im.as_fieldop("deref", node.domain)(node.expr)

        itir_projector, extracted_expr = ir_utils_misc.extract_projector(node.expr)
        projector = self.visit(itir_projector, **kwargs) if itir_projector is not None else None
        node.expr = extracted_expr

        assert cpm.is_applied_as_fieldop(node.expr), node.expr
        stencil = node.expr.fun.args[0]
        domain = node.domain
        inputs = node.expr.args
        lowered_inputs = []
        for input_ in inputs:
            lowered_input = self.visit(input_, **kwargs)

            # convert scalar elements into SIDs, leave rest as is
            def convert_el_to_sid(el_expr: Expr, el_type: ts.ScalarType | ts.FieldType) -> Expr:
                if isinstance(el_type, ts.ScalarType):
                    return SidFromScalar(arg=el_expr)
                else:
                    assert isinstance(el_type, ts.FieldType)
                    return el_expr

            assert isinstance(input_.type, ts.TypeSpec)
            lowered_input_as_sid = _process_elements(
                convert_el_to_sid,
                lowered_input,
                input_.type,
                tuple_constructor=lambda *elements: SidComposite(values=list(elements)),
            )

            lowered_inputs.append(lowered_input_as_sid)

        backend = Backend(domain=self.visit(domain, stencil=stencil, **kwargs))
        if _is_scan(stencil):
            scan_id = self.uids.sequential_id(prefix="_scan")
            scan_lambda = self.visit(stencil.args[0], **kwargs)
            forward = _bool_from_literal(stencil.args[1])
            scan_def = ScanPassDefinition(
                id=scan_id,
                params=scan_lambda.params,
                expr=scan_lambda.expr,
                forward=forward,
                projector=projector,
            )
            extracted_functions.append(scan_def)
            scan = Scan(
                function=SymRef(id=scan_id),
                output=0,
                inputs=[i + 1 for i, _ in enumerate(inputs)],
                init=self.visit(stencil.args[2], **kwargs),
            )
            column_axis = self.column_axis
            assert isinstance(column_axis, common.Dimension)
            return ScanExecution(
                backend=backend,
                scans=[scan],
                args=[self._visit_output_argument(node.target), *lowered_inputs],
                axis=SymRef(id=column_axis.value),
            )
        assert projector is None  # only scans have projectors
        return StencilExecution(
            stencil=self.visit(
                stencil,
                force_function_extraction=True,
                extracted_functions=extracted_functions,
                **kwargs,
            ),
            output=self._visit_output_argument(node.target),
            inputs=lowered_inputs,
            backend=backend,
        )

    def visit_Program(self, node: itir.Program, **kwargs: Any) -> Program:
        extracted_functions: list[Union[FunctionDefinition, ScanPassDefinition]] = []
        executions = self.visit(node.body, extracted_functions=extracted_functions)
        executions = self._merge_scans(executions)
        function_definitions = self.visit(node.function_definitions) + extracted_functions
        offset_definitions = {
            **_collect_dimensions_from_domain(node.body),
            **_collect_offset_definitions(node, self.grid_type, self.offset_provider_type),
        }
        return Program(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=self.grid_type,
            offset_definitions=list(offset_definitions.values()),
            function_definitions=function_definitions,
            temporaries=self.visit(node.declarations, params=[p.id for p in node.params]),
        )

    def visit_Temporary(
        self, node: itir.Temporary, *, params: list, **kwargs: Any
    ) -> TemporaryAllocation:
        def dtype_to_cpp(x: ts.DataType) -> str:
            if isinstance(x, ts.TupleType):
                assert all(isinstance(i, ts.ScalarType) for i in x.types)
                return "::gridtools::tuple<" + ", ".join(dtype_to_cpp(i) for i in x.types) + ">"  # type: ignore[arg-type] # ensured by assert
            assert isinstance(x, ts.ScalarType)
            res = cpp_utils.pytype_to_cpptype(x)
            assert isinstance(res, str)
            return res

        assert node.dtype
        return TemporaryAllocation(
            id=node.id, dtype=dtype_to_cpp(node.dtype), domain=self.visit(node.domain, **kwargs)
        )
