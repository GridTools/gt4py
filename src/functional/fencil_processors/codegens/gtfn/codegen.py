# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


from typing import Any, Collection, Union

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from functional import common
from functional.fencil_processors.codegens.gtfn import gtfn_ir
from functional.fencil_processors.codegens.gtfn.itir_to_gtfn_ir import pytype_to_cpptype


class GTFNCodegen(codegen.TemplatedGenerator):
    _grid_type_str = {
        common.GridType.CARTESIAN: "cartesian",
        common.GridType.UNSTRUCTURED: "unstructured",
    }

    _builtins_mapping = {
        "abs": "std::abs",
        "sin": "std::sin",
        "cos": "std::cos",
        "tan": "std::tan",
        "arcsin": "std::asin",
        "arccos": "std::acos",
        "arctan": "std::atan",
        "sinh": "std::sinh",
        "cosh": "std::cosh",
        "tanh": "std::tanh",
        "arcsinh": "std::asinh",
        "arccosh": "std::acosh",
        "arctanh": "std::atanh",
        "sqrt": "std::sqrt",
        "exp": "std::exp",
        "log": "std::log",
        "gamma": "std::tgamma",
        "cbrt": "std::cbrt",
        "isfinite": "std::isfinite",
        "isinf": "std::isinf",
        "isnan": "std::isnan",
        "floor": "std::floor",
        "ceil": "std::ceil",
        "trunc": "std::trunc",
        "minimum": "std::min",
        "maximum": "std::max",
        "fmod": "std::fmod",
        "power": "std::pow",
    }

    Sym = as_fmt("{id}")

    def visit_SymRef(self, node: gtfn_ir.SymRef, **kwargs: Any) -> str:
        if node.id == "get":
            return "tuple_util::get"
        return node.id

    @staticmethod
    def asfloat(value: str) -> str:
        if "." not in value and "e" not in value and "E" not in value:
            return f"{value}."
        return value

    def visit_Literal(self, node: gtfn_ir.Literal, **kwargs: Any) -> str:
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

    def visit_TaggedValues(self, node: gtfn_ir.TaggedValues, **kwargs):
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
        "unstructured_domain(${tagged_sizes}, ${tagged_offsets}, connectivities__...)"
    )

    def visit_OffsetLiteral(self, node: gtfn_ir.OffsetLiteral, **kwargs: Any) -> str:
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    SidComposite = as_mako(
        "sid::composite::keys<${','.join(f'gridtools::integral_constant<int,{i}>' for i in range(len(values)))}>::make_values(${','.join(values)})"
    )

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs):
        if isinstance(node.fun, gtfn_ir.SymRef) and node.fun.id in self._builtins_mapping:
            return self.generic_visit(node, fun_name=self._builtins_mapping[node.fun.id])
        return self.generic_visit(node, fun_name=self.visit(node.fun))

    FunCall = as_fmt("{fun_name}({','.join(args)})")

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
        "{backend}.vertical_executor({axis})().{'.'.join('arg(' + a + ')' for a in args)}.{'.'.join(scans)}.execute();"
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

    TagDefinition = as_mako(
        """
        %if _this_node.alias:
            %if isinstance(_this_node.alias, str):
                using ${name}_t = ${alias};
            %else:
                using ${name}_t = ${alias}_t;
            %endif
        %else:
            struct ${name}_t{};
        %endif
        constexpr inline ${name}_t ${name}{};
        """
    )

    def visit_FencilDefinition(
        self, node: gtfn_ir.FencilDefinition, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        self.is_cartesian = node.grid_type == common.GridType.CARTESIAN
        return self.generic_visit(
            node,
            grid_type_str=self._grid_type_str[node.grid_type],
            **kwargs,
        )

    TemporaryAllocation = as_fmt(
        "auto {id} = allocate_global_tmp<{dtype}>(tmp_alloc__, {domain}.sizes());"
    )

    FencilDefinition = as_mako(
        """
    #include <cmath>
    #include <gridtools/fn/${grid_type_str}.hpp>

    namespace generated{
    namespace{
    using namespace gridtools;
    using namespace fn;
    using namespace literals;

    ${'\\n'.join(offset_definitions)}
    ${'\\n'.join(function_definitions)}

    inline auto ${id} = [](auto... connectivities__){
        return [connectivities__...](auto backend, ${','.join('auto&& ' + p for p in params)}){
            auto tmp_alloc__ = tmp_allocator(backend);
            ${'\\n'.join(temporaries)}
            ${'\\n'.join(executions)}
        };
    };
    }
    }
    """
    )

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
