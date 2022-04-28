from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from functional.iterator.backends import backend
from functional.iterator.ir import OffsetLiteral
from functional.iterator.transforms import apply_common_transforms


class ToyCpp(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    Literal = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture
    StencilClosure = as_mako("closure(${domain}, ${stencil}, out(${output}), ${','.join(inputs)})")
    FencilDefinition = as_mako(
        """
    ${''.join(function_definitions)}
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

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        transformed = apply_common_transforms(
            root,
            use_tmps=kwargs.get("use_tmps", False),
            offset_provider=kwargs.get("offset_provider", None),
        )
        generated_code = super().apply(transformed, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


backend.register_backend(
    "cpptoy", lambda prog, *args, **kwargs: print(ToyCpp.apply(prog, **kwargs))
)
