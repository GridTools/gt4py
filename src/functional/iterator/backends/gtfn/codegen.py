from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from functional.iterator.backends.gtfn.gtfn_ir import (
    FencilDefinition,
    GridType,
    OffsetLiteral,
    SymRef,
)


class GTFNCodegen(codegen.TemplatedGenerator):
    _grid_type_str = {GridType.CARTESIAN: "cartesian", GridType.UNSTRUCTURED: "unstructured"}

    Sym = as_fmt("{id}")

    def visit_SymRef(self, node: SymRef, **kwargs) -> str:
        if node.id == "get":
            return "tuple_util::get"
        return node.id

    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    UnaryExpr = as_fmt("{op}({expr})")
    BinaryExpr = as_fmt("({lhs}{op}{rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    FunCall = as_fmt("{fun}({','.join(args)})")
    TemplatedFunCall = as_fmt("{fun}<{','.join(template_args)}>({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture

    Backend = as_fmt("make_backend(backend, {domain})")

    StencilExecution = as_mako(
        """
        ${backend}.stencil_executor()().arg(${output})${''.join('.arg(' + i + ')' for i in inputs)}.assign(0_c, ${stencil}(), ${','.join(str(i) + '_c' for i in range(1, len(inputs) + 1))}).execute();
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

    def visit_FencilDefinition(self, node: FencilDefinition, **kwargs):
        is_cartesian = node.grid_type == GridType.CARTESIAN
        return self.generic_visit(
            node,
            is_cartesian=is_cartesian,
            grid_type_str=self._grid_type_str[node.grid_type],
            **kwargs,
        )

    FencilDefinition = as_mako(
        """
    #include <gridtools/fn/${grid_type_str}.hpp>

    namespace generated{
    using namespace gridtools;
    using namespace fn;
    using namespace literals;


    % if is_cartesian:
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
        ${'\\n'.join(executions)}
    };
    }
    """
    )

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
