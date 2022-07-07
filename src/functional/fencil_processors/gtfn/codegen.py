from typing import Any, Collection, Union

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from functional.fencil_processors.gtfn.gtfn_ir import (
    FencilDefinition,
    GridType,
    Literal,
    OffsetLiteral,
    SymRef,
)
from functional.fencil_processors.gtfn.itir_to_gtfn_ir import pytype_to_cpptype


class GTFNCodegen(codegen.TemplatedGenerator):
    _grid_type_str = {GridType.CARTESIAN: "cartesian", GridType.UNSTRUCTURED: "unstructured"}

    Sym = as_fmt("{id}")

    def visit_SymRef(self, node: SymRef, **kwargs: Any) -> str:
        if node.id == "get":
            return "tuple_util::get"
        return node.id

    @staticmethod
    def asfloat(value: str) -> str:
        if "." not in value and "e" not in value and "E" not in value:
            return f"{value}."
        return value

    def visit_Literal(self, node: Literal, **kwargs: Any) -> str:
        match pytype_to_cpptype(node.type):
            case "int":
                return node.value + "_c"
            case "float":
                return self.asfloat(node.value) + "f"
            case "double":
                return self.asfloat(node.value)
            case "bool":
                return node.value.lower()
            case _:
                return node.value

    UnaryExpr = as_fmt("{op}({expr})")
    BinaryExpr = as_fmt("({lhs}{op}{rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs: Any) -> str:
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture

    Backend = as_fmt("make_backend(backend, {domain})")

    StencilExecution = as_mako(
        """
        ${backend}.stencil_executor()().arg(${output})${''.join('.arg(' + i + ')' for i in inputs)}.assign(0_c, ${stencil}() ${',' if inputs else ''} ${','.join(str(i) + '_c' for i in range(1, len(inputs) + 1))}).execute();
        """
    )

    Scan = as_fmt("assign({output}, {function}(), {init}, {', '.join(inputs)})")
    ScanExecution = as_fmt(
        "{backend}.vertical_executor()().{'.'.join('arg(' + a + ')' for a in args)}.{'.'.join(scans)}.execute();"
    )

    ScanPassDefinition = as_mako(
        """
        struct ${id} : ${'fwd' if _this_node.forward else 'bwd'} {
            static constexpr GT_FUNCTION auto body() {
                return scan_pass([](${','.join('auto const& ' + p for p in params)}) {
                    return ${expr};
                }, host_device::identity());
            }
        };
        """
    )

    FunctionDefinition = as_mako(
        """
        struct ${id} {
            constexpr auto operator()() const {
                return [](${','.join('auto const& ' + p for p in params)}){
                    return ${expr};
                };
            }
        };
    """
    )

    def visit_FencilDefinition(
        self, node: FencilDefinition, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        is_cartesian = node.grid_type == GridType.CARTESIAN
        return self.generic_visit(
            node,
            is_cartesian=is_cartesian,
            grid_type_str=self._grid_type_str[node.grid_type],
            **kwargs,
        )

    TemporaryAllocation = as_fmt(
        "auto {id} = allocate_global_tmp<{dtype}>(_tmp_alloc, dom.sizes());"
    )

    FencilDefinition = as_mako(
        """
    #include <gridtools/fn/${grid_type_str}.hpp>

    namespace generated{
    using namespace gridtools;
    using namespace fn;
    using namespace literals;


    % if is_cartesian:
        // TODO allow non-default names
        using namespace cartesian;
        constexpr inline dim::i i = {};
        constexpr inline dim::j j = {};
        constexpr inline dim::k k = {};
    % else:
        ${''.join('struct ' + o + '_t{};' for o in offset_declarations)}
        ${''.join('constexpr inline ' + o + '_t ' + o + '{};' for o in offset_declarations)}
    % endif

    ${''.join(function_definitions)}

    inline auto ${id} = [](auto backend, ${','.join('auto&& ' + p for p in params)}){
        auto _tmp_alloc = tmp_allocator(backend);
        ${'\\n'.join(temporaries)}
        ${'\\n'.join(executions)}
    };
    }
    """
    )

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
