from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from functional.iterator.backends.gtfn.gtfn_ir import FencilDefinition, GridType, OffsetLiteral


class gtfn_codegen(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    UnaryExpr = as_fmt("{op}({expr})")
    BinaryExpr = as_fmt("({lhs}{op}{rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    StringLiteral = as_fmt("{value}")

    FunCall = as_fmt("{fun}({','.join(args)})")
    TemplatedFunCall = as_fmt("{fun}<{','.join(template_args)}>({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture

    Backend = as_fmt("make_backend(backend, {domain})")

    StencilExecution = as_mako(
        """
        ${backend}.stencil_executor()().arg(${output})${''.join('.arg('+i+')' for i in inputs)}.assign(0_c, ${stencil}(), ${','.join(str(i)+'_c' for i in range(1,len(inputs)+1))}).execute();
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
        grid_type_str = "cartesian" if node.grid_type == GridType.Cartesian else "unstructured"
        is_cartesian = node.grid_type == GridType.Cartesian
        return self.generic_visit(
            node, is_cartesian=is_cartesian, grid_type_str=grid_type_str, **kwargs
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
        ${''.join('struct ' + o + '_t{};' for o in offsets)}
        ${''.join('constexpr inline ' + o + '_t ' + o + '{};' for o in offsets)}
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
