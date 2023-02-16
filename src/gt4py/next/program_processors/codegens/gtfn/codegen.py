# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import common
from gt4py.next.program_processors.codegens.gtfn import gtfn_im_ir, gtfn_ir, gtfn_ir_common
from gt4py.next.program_processors.codegens.gtfn.itir_to_gtfn_ir import pytype_to_cpptype


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
        "float": "double",
        "float32": "float",
        "float64": "double",
        "int": "long",
        "int32": "std::int32_t",
        "int64": "std::int64_t",
        "bool": "bool",
        "plus": "std::plus{}",
        "minus": "std::minus{}",
        "multiplies": "std::multiplies{}",
        "divides": "std::divides{}",
        "eq": "std::equal_to{}",
        "not_eq": "std::not_equal_to{}",
        "less": "std::less{}",
        "less_equal": "std::less_equal{}",
        "greater": "std::greater{}",
        "greater_equal": "std::greater_equal{}",
        "and_": "std::logical_and{}",
        "or_": "std::logical_or{}",
        "xor_": "std::bit_xor{}",
        "mod": "std::modulus{}",
        "not_": "std::logical_not{}",
    }

    Sym = as_fmt("{id}")

    def visit_SymRef(self, node: gtfn_ir_common.SymRef, **kwargs: Any) -> str:
        if node.id == "get":
            return "::gridtools::tuple_util::get"
        if node.id in self._builtins_mapping:
            return self._builtins_mapping[node.id]
        if node.id in gtfn_ir.GTFN_BUILTINS:
            qualified_fun_name = f"gtfn::{node.id}"
            return qualified_fun_name

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
    CastExpr = as_fmt("static_cast<{new_dtype}>({obj_expr})")

    def visit_TaggedValues(self, node: gtfn_ir.TaggedValues, **kwargs):
        tags = self.visit(node.tags)
        values = self.visit(node.values)
        if self.is_cartesian:
            return f"::gridtools::hymap::keys<{','.join(t + '_t' for t in tags)}>::make_values({','.join(values)})"
        else:
            return f"::gridtools::tuple({','.join(values)})"

    CartesianDomain = as_fmt("gtfn::cartesian_domain({tagged_sizes}, {tagged_offsets})")
    UnstructuredDomain = as_mako(
        "gtfn::unstructured_domain(${tagged_sizes}, ${tagged_offsets}, connectivities__...)"
    )

    def visit_OffsetLiteral(self, node: gtfn_ir.OffsetLiteral, **kwargs: Any) -> str:
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    SidComposite = as_mako(
        "::gridtools::sid::composite::keys<${','.join(f'::gridtools::integral_constant<int,{i}>' for i in range(len(values)))}>::make_values(${','.join(values)})"
    )

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs):
        if (
            isinstance(node.fun, gtfn_ir_common.SymRef)
            and node.fun.id in self.user_defined_function_ids
        ):
            fun_name = f"{self.visit(node.fun)}{{}}()"
        else:
            fun_name = self.visit(node.fun)

        return self.generic_visit(node, fun_name=fun_name)

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

    Scan = as_fmt("assign({output}, {function}(), {', '.join([init] + inputs)})")
    ScanExecution = as_fmt(
        "{backend}.vertical_executor({axis})().{'.'.join('arg(' + a + ')' for a in args)}.{'.'.join(scans)}.execute();"
    )

    ScanPassDefinition = as_mako(
        """
        struct ${id} : ${'gtfn::fwd' if _this_node.forward else 'gtfn::bwd'} {
            static constexpr GT_FUNCTION auto body() {
                return gtfn::scan_pass([](${','.join('auto const& ' + p for p in params)}) {
                    return ${expr};
                }, ::gridtools::host_device::identity());
            }
        };
        """
    )

    FunctionDefinition = as_mako(
        """
        struct ${id} {
            constexpr auto operator()() const {
                return [](${','.join('auto const& ' + p for p in params)}){
                    ${expr_};
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

    def visit_FunctionDefinition(self, node: gtfn_ir.FunctionDefinition, **kwargs):
        expr_ = "return " + self.visit(node.expr)
        return self.generic_visit(node, expr_=expr_)

    def visit_FencilDefinition(
        self, node: gtfn_ir.FencilDefinition, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        self.is_cartesian = node.grid_type == common.GridType.CARTESIAN
        self.user_defined_function_ids = list(
            str(fundef.id) for fundef in node.function_definitions
        )
        return self.generic_visit(
            node,
            grid_type_str=self._grid_type_str[node.grid_type],
            **kwargs,
        )

    TemporaryAllocation = as_fmt(
        "auto {id} = gtfn::allocate_global_tmp<{dtype}>(tmp_alloc__, {domain}.sizes());"
    )

    FencilDefinition = as_mako(
        """
    #include <cmath>
    #include <cstdint>
    #include <functional>
    #include <gridtools/fn/${grid_type_str}.hpp>
    #include <gridtools/fn/sid_neighbor_table.hpp>

    namespace generated{

    namespace gtfn = ::gridtools::fn;

    namespace{
    using namespace ::gridtools::literals;

    ${'\\n'.join(offset_definitions)}
    ${'\\n'.join(function_definitions)}

    inline auto ${id} = [](auto... connectivities__){
        return [connectivities__...](auto backend, ${','.join('auto&& ' + p for p in params)}){
            auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);
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


class GTFNIMCodegen(GTFNCodegen):

    Stmt = as_fmt("{lhs} {op} {rhs};")

    InitStmt = as_fmt("{init_type} {lhs} {op} {rhs};")

    EmptyListInitializer = as_mako("{}")

    Conditional = as_mako(
        """
          using ${cond_type} = typename std::common_type<decltype(${if_rhs_}), decltype(${else_rhs_})>::type;
          ${init_stmt}
          if (${cond}) {
            ${if_stmt}
          } else {
            ${else_stmt}
          }
    """
    )

    ImperativeFunctionDefinition = as_mako(
        """
        struct ${id} {
            constexpr auto operator()() const {
                return [](${','.join('auto const& ' + p for p in params)}){
                    ${expr_};
                };
            }
        };
    """
    )

    ReturnStmt = as_fmt("return {ret};")

    def visit_Conditional(self, node: gtfn_im_ir.Conditional, **kwargs):
        if_rhs_ = self.visit(node.if_stmt.rhs)
        else_rhs_ = self.visit(node.else_stmt.rhs)
        return self.generic_visit(node, if_rhs_=if_rhs_, else_rhs_=else_rhs_)

    def visit_ImperativeFunctionDefinition(
        self, node: gtfn_im_ir.ImperativeFunctionDefinition, **kwargs
    ):
        expr_ = "".join(self.visit(stmt) for stmt in node.fun)
        return self.generic_visit(node, expr_=expr_)

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
