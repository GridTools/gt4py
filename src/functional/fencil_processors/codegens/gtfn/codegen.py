from typing import Any, Collection, Union

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from functional.fencil_processors.codegens.gtfn.gtfn_ir import (
    FencilDefinition,
    GridType,
    Literal,
    OffsetLiteral,
    SymRef,
    TaggedValues,
)


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
        if node.type == "int":
            return node.value + "_c"
        elif node.type == "float32":
            return f"{self.asfloat(node.value)}f"
        elif node.type == "float" or node.type == "float64":
            return self.asfloat(node.value)
        elif node.type == "bool":
            return node.value.lower()
        return node.value

    UnaryExpr = as_fmt("{op}({expr})")
    BinaryExpr = as_fmt("({lhs}{op}{rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")

    def visit_TaggedValues(self, node: TaggedValues, **kwargs):
        tags = self.visit(node.tags)
        values = self.visit(node.values)
        if self.is_cartesian:
            return (
                f"hymap::keys<{','.join(t + '_t' for t in tags)}>::make_values({','.join(values)})"
            )
        else:
            return f"tuple({','.join(values)})"

    CartesianDomain = as_fmt("cartesian_domain({tagged_sizes}, {tagged_offsets})")
    UnstructuredDomain = as_mako(
        "unstructured_domain(${tagged_sizes}, ${tagged_offsets} ${',' if len(connectivities) else ''} ${','.join(f'at_key<{c}_t>(connectivities__)' for c in connectivities)})"
    )

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
        self.is_cartesian = node.grid_type == GridType.CARTESIAN
        return self.generic_visit(
            node,
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

    ${''.join('struct ' + o + '_t{};' for o in offset_declarations)}
    ${''.join('constexpr inline ' + o + '_t ' + o + '{};' for o in offset_declarations)}
    ${''.join(function_definitions)}

    inline auto ${id} = [](auto connectivities__){
        return [connectivities__](auto backend, ${','.join('auto&& ' + p for p in params)}){
            ${'\\n'.join(executions)}
        };
    };
    }
    """
    )

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
