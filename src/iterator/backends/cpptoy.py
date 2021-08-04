from typing import Any
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from iterator.ir import OffsetLiteral
from iterator.backends import backend


class ToyCpp(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    StringLiteral = as_fmt("{value}")
    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture
    StencilClosure = as_mako(
        "closure(${domain}, ${stencil}, out(${','.join(outputs)}), ${','.join(inputs)})"
    )
    FencilDefinition = as_mako(
        """
    auto ${id} = [](${','.join('auto&& ' + p for p in params)}){
        fencil(${'\\n'.join(closures)});
    };
    """
    )
    FunctionDefinition = as_mako(
        """
    inline constexpr auto ${id} = [](${','.join('auto ' + p for p in params)}){
        return ${expr};
        };
    """
    )
    Program = as_fmt("{''.join(function_definitions)} {''.join(fencil_definitions)}")

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


backend.register_backend("cpptoy", lambda prog, *args, **kwargs: print(ToyCpp.apply(prog)))
