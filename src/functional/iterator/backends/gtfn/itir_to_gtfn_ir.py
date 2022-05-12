from typing import Any, Optional, Union

from eve import NodeTranslator, iter_tree
from eve.type_definitions import SymbolName
from eve.utils import UIDs
from functional.iterator import ir as itir
from functional.iterator.backends.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    Expr,
    FencilDefinition,
    FunCall,
    FunctionDefinition,
    GridType,
    Lambda,
    Literal,
    OffsetLiteral,
    Scan,
    ScanExecution,
    ScanPassDefinition,
    StencilExecution,
    Sym,
    SymRef,
    TemporaryAllocation,
    TernaryExpr,
    UnaryExpr,
)


def pytype_to_cpptype(t: str):
    try:
        return {
            "float": "double",
            "float32": "float",
            "float64": "double",
            "int": "int",
            "int32": "std::int32_t",
            "int64": "std::int64_t",
            "bool": "bool",
        }[t]
    except KeyError:
        raise TypeError(f"Unsupported type '{t}'") from None


class GTFN_lowering(NodeTranslator):
    _binary_op_map = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "less": "<",
        "greater": ">",
        "and_": "&&",
        "or_": "||",
    }
    _unary_op_map = {"not_": "!"}

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(
        self,
        node: itir.SymRef,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> SymRef:
        if force_function_extraction:
            assert extracted_functions is not None
            assert node.id == "deref"
            fun_id = UIDs.sequential_id(prefix="_fun")
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
    ) -> Lambda:
        if force_function_extraction:
            assert extracted_functions is not None
            fun_id = UIDs.sequential_id(prefix="_fun")
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
        return Literal(
            value="NOT_SUPPORTED", type="axis_literal"
        )  # TODO(havogt) decide if domain is part of the IR

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Expr:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0]))
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0], **kwargs),
                    rhs=self.visit(node.args[1], **kwargs),
                )
            elif node.fun.id == "if_":
                assert len(node.args) == 3
                return TernaryExpr(
                    cond=self.visit(node.args[0], **kwargs),
                    true_expr=self.visit(node.args[1], **kwargs),
                    false_expr=self.visit(node.args[2], **kwargs),
                )
            elif (
                node.fun.id == "deref"
                and isinstance(node.args[0], itir.FunCall)
                and isinstance(node.args[0].fun, itir.FunCall)
                and node.args[0].fun.fun == itir.SymRef(id="shift")
                and len((offsets := node.args[0].fun.args)) % 2
            ):
                # deref(shift(i)(sparse)) -> tuple_get(i, deref(sparse))
                # TODO: remove once ‘real’ sparse field handling is available
                deref_arg = node.args[0].args[0]
                if len(offsets) > 1:
                    deref_arg = itir.FunCall(
                        fun=itir.FunCall(fun=itir.SymRef(id="shift"), args=offsets[:-1]),
                        args=[deref_arg],
                    )
                derefed = itir.FunCall(fun=itir.SymRef(id="deref"), args=[deref_arg])
                sparse_access = itir.FunCall(
                    fun=itir.SymRef(id="tuple_get"), args=[offsets[-1], derefed]
                )
                return self.visit(sparse_access, **kwargs)
            elif node.fun.id == "shift":
                raise ValueError("unapplied shift call not supported: {node}")
            elif node.fun.id == "scan":
                raise ValueError("scans are only supported at the top level of a stencil closure")
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun == itir.SymRef(id="shift"):
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

    @staticmethod
    def _bool_from_literal(node: itir.Node):
        assert isinstance(node, itir.Literal)
        assert node.type == "bool" and node.value in ("True", "False")
        return node.value == "True"

    def visit_StencilClosure(
        self, node: itir.StencilClosure, extracted_functions: list, **kwargs: Any
    ) -> StencilExecution:
        backend = Backend(domain=self.visit(node.domain, **kwargs))
        if self._is_scan(node.stencil):
            scan_id = UIDs.sequential_id(prefix="_scan")
            scan_lambda = self.visit(node.stencil.args[0], **kwargs)
            forward = self._bool_from_literal(node.stencil.args[1])
            scan_def = ScanPassDefinition(
                id=scan_id, params=scan_lambda.params, expr=scan_lambda.expr, forward=forward
            )
            extracted_functions.append(scan_def)
            scan = Scan(
                function=SymRef(id=scan_id),
                output=Literal(value=0, type="int"),
                inputs=[Literal(value=i + 1, type="int") for i, _ in enumerate(node.inputs)],
                init=self.visit(node.stencil.args[2], **kwargs),
            )
            return ScanExecution(
                backend=backend,
                scans=[scan],
                args=[self.visit(node.output, **kwargs)] + self.visit(node.inputs),
            )
        return StencilExecution(
            stencil=self.visit(
                node.stencil,
                force_function_extraction=True,
                extracted_functions=extracted_functions,
                **kwargs,
            ),
            output=self.visit(node.output, **kwargs),
            inputs=self.visit(node.inputs, **kwargs),
            backend=backend,
        )

    @staticmethod
    def _merge_scans(
        executions: list[Union[StencilExecution, ScanExecution]]
    ) -> list[Union[StencilExecution, ScanExecution]]:
        def merge(a: ScanExecution, b: ScanExecution) -> ScanExecution:
            assert a.backend == b.backend

            index_map = dict[int, int]()
            compacted_b_args = list[SymRef]()
            for b_idx, b_arg in enumerate(b.args):
                try:
                    a_idx = a.args.index(b_arg)
                    index_map[b_idx] = a_idx
                except ValueError:
                    index_map[b_idx] = len(a.args) + len(compacted_b_args)
                    compacted_b_args.append(b_arg)

            def remap_args(s: Scan) -> Scan:
                def remap_literal(x: Literal) -> Literal:
                    return Literal(value=str(index_map[int(x.value)]), type=x.type)

                return Scan(
                    function=s.function,
                    output=remap_literal(s.output),
                    inputs=[remap_literal(i) for i in s.inputs],
                    init=s.init,
                )

            return ScanExecution(
                backend=a.backend,
                scans=a.scans + [remap_args(s) for s in b.scans],
                args=a.args + compacted_b_args,
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

    @staticmethod
    def _collect_offsets(node: itir.FencilDefinition) -> set[str]:
        return (
            iter_tree(node)
            .if_isinstance(itir.OffsetLiteral)
            .getattr("value")
            .if_isinstance(str)
            .to_set()
        )

    def visit_FencilDefinition(
        self, node: itir.FencilDefinition, *, grid_type: str, **kwargs: Any
    ) -> FencilDefinition:
        grid_type = getattr(GridType, grid_type.upper())
        extracted_functions: list[Union[FunctionDefinition, ScanPassDefinition]] = []
        executions = self.visit(node.closures, extracted_functions=extracted_functions)
        executions = self._merge_scans(executions)
        return FencilDefinition(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            offset_declarations=self._collect_offsets(node),
            function_definitions=self.visit(node.function_definitions) + extracted_functions,
            grid_type=grid_type,
            temporaries=[],
        )

    def visit_Temporary(self, node, *, params: list, **kwargs) -> TemporaryAllocation:
        def dtype_to_cpp(x):
            if isinstance(x, int):
                return f"std::remove_const_t<sid::element_type<decltype({params[x]})>>"
            if isinstance(x, tuple):
                return "tuple<" + ", ".join(dtype_to_cpp(i) for i in x) + ">"
            assert isinstance(x, str)
            return pytype_to_cpptype(x)

        return TemporaryAllocation(id=node.id, dtype=dtype_to_cpp(node.dtype))

    def visit_FencilWithTemporaries(self, node, **kwargs) -> FencilDefinition:
        fencil = self.visit(node.fencil, **kwargs)
        return FencilDefinition(
            id=fencil.id,
            params=self.visit(node.params),
            executions=fencil.executions,
            grid_type=fencil.grid_type,
            offset_declarations=fencil.offset_declarations,
            function_definitions=fencil.function_definitions,
            temporaries=self.visit(node.tmps, params=[p.id for p in node.params]),
        )
