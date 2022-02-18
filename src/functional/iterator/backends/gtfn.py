from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from functional.iterator.backends import backend
from functional.iterator.ir import FunCall, OffsetLiteral, Program, StencilClosure, SymRef
from functional.iterator.transforms import apply_common_transforms


# TODO test non-transformed code!
class gtfn(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    StringLiteral = as_fmt("{value}")

    def visit_FunCall(self, node: FunCall, **kwargs):
        if isinstance(node.fun, SymRef) and node.fun.id == "domain":
            sizes = []
            for a in node.args:
                if not (
                    isinstance(a, FunCall)
                    and isinstance(a.fun, SymRef)
                    and a.fun.id == "named_range"
                ):
                    raise RuntimeError(f"expected named_range, got {a.fun.id}")
                sizes.append(self.visit(a.args[2]))  # TODO start is ignored, and names are ignored
            return f"cartesian_domain({','.join(sizes)})"
        return self.generic_visit(node, **kwargs)

    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture

    def visit_StencilClosure(self, node: StencilClosure, **kwargs):
        if not isinstance(node.stencil, SymRef):
            raise NotImplementedError(
                "Stencil is required to be a SymRef, cannot use arbitrary expressions."
            )
        stencil_instantiation = self.visit(node.stencil) + "{}"
        return self.generic_visit(node, stencil_instantiation=stencil_instantiation, **kwargs)

    StencilClosure = as_mako(
        """
        [](auto &&executor, ${','.join('auto & ' + o for o in outputs)}, ${','.join('auto const & ' + i for i in inputs)}) {
            executor().arg(${outputs[0]})${''.join('.arg('+i+')' for i in inputs)}.assign(0_c, ${stencil_instantiation}, 1_c /*TODO*/);
        }(make_backend(gridtools::fn::backend::naive()/*TODO*/, ${domain}).stencil_executor(), ${outputs[0]}, ${','.join(inputs)});"""
    )
    FencilDefinition = as_mako(
        """
    inline auto ${id} = [](${','.join('auto&& ' + p for p in params)}){
        ${'\\n'.join(closures)}
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

    def visit_Program(self, node: Program, **kwargs):
        return self.generic_visit(node, **kwargs)

    Program = as_fmt(
        """
    #include <gridtools/fn/cartesian2.hpp>
    #include <gridtools/fn/unstructured2.hpp>
    #include <gridtools/fn/backend2/naive.hpp>

    namespace generated{{
    using namespace gridtools;
    using namespace fn;
    using namespace literals;
    {''.join(function_definitions)} {''.join(fencil_definitions)}
    }}
    """
    )

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        transformed = apply_common_transforms(
            root,
            use_tmps=kwargs.get("use_tmps", False),
            skip_inline_fundefs=True,
            offset_provider=kwargs.get("offset_provider", None),
        )
        generated_code = super().apply(transformed, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


backend.register_backend("gtfn", lambda prog, *args, **kwargs: print(gtfn.apply(prog, **kwargs)))
