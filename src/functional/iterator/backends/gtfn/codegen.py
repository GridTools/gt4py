from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.iterators import iter_tree
from functional.iterator.backends import backend
from functional.iterator.backends.gtfn import gtfn_ir
from functional.iterator.backends.gtfn.gtfn_ir import FunCall, OffsetLiteral, Program, SymRef
from functional.iterator.transforms import apply_common_transforms


class gtfn_codegen(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")
    UnaryExpr = as_fmt("{op}{expr}")

    # def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
    #     return node.value if isinstance(node.value, str) else f"{node.value}_c"

    StringLiteral = as_fmt("{value}")

    # def visit_FunCall(self, node: FunCall, **kwargs):
    #     if isinstance(node.fun, SymRef) and node.fun.id == "domain":
    #         sizes = []
    #         for a in node.args:
    #             if not (
    #                 isinstance(a, FunCall)
    #                 and isinstance(a.fun, SymRef)
    #                 and a.fun.id == "named_range"
    #             ):
    #                 raise RuntimeError(f"expected named_range, got {a.fun.id}")
    #             sizes.append(self.visit(a.args[2]))  # TODO start is ignored, and names are ignored
    #         return f"cartesian_domain({','.join(sizes)})"
    #     elif isinstance(node.fun, FunCall) and node.fun.fun.id == "shift":
    #         if len(node.fun.args) > 0:
    #             return self.generic_visit(
    #                 FunCall(fun=node.fun.fun, args=node.args + node.fun.args), **kwargs
    #             )
    #         else:
    #             # get rid of the shift call if there are no offsets, should be a separate pass
    #             assert len(node.args) == 1
    #             return self.visit(node.args[0])
    #         # return {"fun": "shift", "args": self.visit(node.args) + self.visit(node.fun.args)}
    #     elif isinstance(node.fun, SymRef) and node.fun.id in binary_ops:
    #         assert len(node.args) == 2
    #         return (
    #             f"({self.visit(node.args[0])} {binary_ops[node.fun.id]} {self.visit(node.args[1])})"
    #         )
    #     return self.generic_visit(node, **kwargs)

    FunCall = as_fmt("{fun}({','.join(args)})")
    # Lambda = as_mako(
    #     "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    # )  # TODO capture

    # def visit_StencilClosure(self, node: StencilClosure, **kwargs):
    #     if not isinstance(node.stencil, SymRef):
    #         raise NotImplementedError(
    #             "Stencil is required to be a SymRef, cannot use arbitrary expressions."
    #         )
    #     stencil_instantiation = self.visit(node.stencil) + "{}"
    #     return self.generic_visit(node, stencil_instantiation=stencil_instantiation, **kwargs)

    # StencilClosure = as_mako(
    #     """
    #     [](auto &&executor, ${','.join('auto & ' + o for o in outputs)}, ${','.join('auto const & ' + i for i in inputs)}) {
    #         executor().arg(${outputs[0]})${''.join('.arg('+i+')' for i in inputs)}.assign(0_c, ${stencil_instantiation}, 1_c /*TODO*/);
    #     }(make_backend(gridtools::fn::backend::naive()/*TODO*/, ${domain}).stencil_executor(), ${outputs[0]}, ${','.join(inputs)});"""
    # )

    Backend = as_fmt("make_backend({backend_tag}, domain)")

    StencilExecution = as_mako(
        """
        ${backend}.stencil_executor()().arg(${output})${''.join('.arg('+i+')' for i in inputs)}.assign(0_c, ${stencil}(), ${','.join(str(i)+'_c' for i in range(1,len(inputs)+1))});
        """
    )

    FencilDefinition = as_mako(
        """
    inline auto ${id} = [](${','.join('auto&& ' + p for p in params)}){
        ${'\\n'.join(executions)}
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

    # @staticmethod
    # def _collect_offsets(node: Program) -> list[str]:
    #     return (
    #         iter_tree(node)
    #         .if_isinstance(OffsetLiteral)
    #         .getattr("value")
    #         .if_isinstance(str)
    #         .to_set()
    #     )

    # def visit_Program(self, node: Program, **kwargs):
    #     return self.generic_visit(node, offsets=self._collect_offsets(node), **kwargs)

    Program = as_mako(
        """
    #include <gridtools/fn/cartesian2.hpp>
    #include <gridtools/fn/unstructured2.hpp>
    #include <gridtools/fn/backend2/naive.hpp>

    namespace generated{
    using namespace gridtools;
    using namespace fn;
    using namespace literals;


    //{''.join('struct ' + o + '_t{};' for o in offsets)}
    //{''.join('constexpr inline ' + o + '_t ' + o + '{};' for o in offsets)}
    using namespace cartesian;
    constexpr inline dim::i i = {};
    constexpr inline dim::j j = {};
    constexpr inline dim::k k = {};


    ${''.join(function_definitions)} ${''.join(fencil_definitions)}
    }
    """
    )

    @classmethod
    def apply(cls, root: gtfn_ir.Program, **kwargs: Any) -> str:
        transformed = root
        # transformed = apply_common_transforms(
        #     transformed,
        #     use_tmps=kwargs.get("use_tmps", False),
        #     offset_provider=kwargs.get("offset_provider", None),
        #     grid_type=kwargs.get("grid_type", None),
        # )
        generated_code = super().apply(transformed, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


backend.register_backend("gtfn", lambda prog, *args, **kwargs: print(gtfn.apply(prog, **kwargs)))
