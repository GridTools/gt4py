# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Collection, Final, Union

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import common
from gt4py.next.otf import cpp_utils
from gt4py.next.program_processors.codegens.gtfn import gtfn_im_ir, gtfn_ir, gtfn_ir_common


class GTFNCodegen(codegen.TemplatedGenerator):
    _grid_type_str: Final = {
        common.GridType.CARTESIAN: "cartesian",
        common.GridType.UNSTRUCTURED: "unstructured",
    }

    _builtins_mapping: Final = {
        "abs": "std::abs",
        "neg": "std::negate{}",
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
        "float32": "float",
        "float64": "double",
        "int8": "std::int8_t",
        "uint8": "std::uint8_t",
        "int16": "std::int16_t",
        "uint16": "std::uint16_t",
        "int32": "std::int32_t",
        "uint32": "std::uint32_t",
        "int64": "std::int64_t",
        "uint64": "std::uint64_t",
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
        if node.type == "axis_literal":
            return node.value

        # TODO(tehrengruber): isn't this wrong and int32 should be casted to an actual int32?
        match cpp_utils.pytype_to_cpptype(node.type):
            case "float":
                return self.asfloat(node.value) + "f"
            case "double":
                return self.asfloat(node.value)
            case "bool":
                return node.value.lower()
            case _:
                # TODO(tehrengruber): we should probably shouldn't just allow anything here. Revisit.
                return node.value

    IntegralConstant = as_fmt("{value}_c")

    UnaryExpr = as_fmt("{op}({expr})")
    # add an extra space between the operators is needed such that `minus(1, -1)` does not get
    # translated into `1--1`, but `1 - -1`
    BinaryExpr = as_fmt("({lhs} {op} {rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")
    CastExpr = as_fmt("static_cast<{new_dtype}>({obj_expr})")

    def visit_TaggedValues(self, node: gtfn_ir.TaggedValues, **kwargs: Any) -> str:
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

    SidFromScalar = as_fmt("gridtools::stencil::global_parameter({arg})")

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs: Any) -> str:
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

    Scan = as_fmt(
        "assign({output}_c, {function}(), {', '.join([init] + [input + '_c' for input in inputs])})"
    )
    ScanExecution = as_fmt(
        "{backend}.vertical_executor({axis})().{'.'.join('arg(' + a + ')' for a in args)}.{'.'.join(scans)}.execute();"
    )

    IfStmt = as_mako(
        """
          if (${cond}) {
            ${'\\n'.join(true_branch)}
          } else {
            ${'\\n'.join(false_branch)}
          }
        """
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

    def visit_FunctionDefinition(self, node: gtfn_ir.FunctionDefinition, **kwargs: Any) -> str:
        expr_ = "return " + self.visit(node.expr)
        return self.generic_visit(node, expr_=expr_)

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

    def visit_TemporaryAllocation(self, node: gtfn_ir.TemporaryAllocation, **kwargs: Any) -> str:
        # TODO(tehrengruber): Revisit. We are currently converting an itir.NamedRange with
        #  start and stop values into an gtfn_ir.(Cartesian|Unstructured)Domain with
        #  size and offset values, just to here convert back in order to obtain stop values again.
        # TODO(tehrengruber): Fix memory alignment.
        assert isinstance(node.domain, (gtfn_ir.CartesianDomain, gtfn_ir.UnstructuredDomain))
        assert node.domain.tagged_offsets.tags == node.domain.tagged_sizes.tags
        tags = node.domain.tagged_offsets.tags
        new_sizes = []
        for size, offset in zip(node.domain.tagged_offsets.values, node.domain.tagged_sizes.values):
            new_sizes.append(gtfn_ir.BinaryExpr(op="+", lhs=size, rhs=offset))
        return self.generic_visit(
            node,
            tmp_sizes=self.visit(gtfn_ir.TaggedValues(tags=tags, values=new_sizes), **kwargs),
            **kwargs,
        )

    TemporaryAllocation = as_fmt(
        "auto {id} = gtfn::allocate_global_tmp<{dtype}>(tmp_alloc__, {tmp_sizes});"
    )

    def visit_Program(self, node: gtfn_ir.Program, **kwargs: Any) -> Union[str, Collection[str]]:
        self.is_cartesian = node.grid_type == common.GridType.CARTESIAN
        self.user_defined_function_ids = list(
            str(fundef.id) for fundef in node.function_definitions
        )
        return self.generic_visit(
            node,
            grid_type_str=self._grid_type_str[node.grid_type],
            block_sizes=self._block_sizes(node.offset_definitions),
            **kwargs,
        )

    Program = as_mako(
        """
    #include <cmath>
    #include <cstdint>
    #include <functional>
    #include <gridtools/fn/${grid_type_str}.hpp>
    #include <gridtools/fn/sid_neighbor_table.hpp>
    #include <gridtools/stencil/global_parameter.hpp>
    
    namespace generated{

    namespace gtfn = ::gridtools::fn;

    namespace{
    using namespace ::gridtools::literals;

    ${'\\n'.join(offset_definitions)}
    ${'\\n'.join(function_definitions)}

    ${block_sizes}

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

    def _block_sizes(self, offset_definitions: list[gtfn_ir.TagDefinition]) -> str:
        if self.is_cartesian:
            block_dims = []
            block_sizes = [32, 8] + [1] * (len(offset_definitions) - 2)
            for i, tag in enumerate(offset_definitions):
                if tag.alias is None:
                    block_dims.append(
                        f"gridtools::meta::list<{tag.name.id}_t, "
                        f"gridtools::integral_constant<int, {block_sizes[i]}>>"
                    )
            sizes_str = ",\n".join(block_dims)
            return f"using block_sizes_t = gridtools::meta::list<{sizes_str}>;"
        else:
            return "using block_sizes_t = gridtools::meta::list<gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, 32>>, gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, 8>>>;"

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

    def visit_Conditional(self, node: gtfn_im_ir.Conditional, **kwargs: Any) -> str:
        if_rhs_ = self.visit(node.if_stmt.rhs)
        else_rhs_ = self.visit(node.else_stmt.rhs)
        return self.generic_visit(node, if_rhs_=if_rhs_, else_rhs_=else_rhs_)

    def visit_ImperativeFunctionDefinition(
        self, node: gtfn_im_ir.ImperativeFunctionDefinition, **kwargs: Any
    ) -> str:
        expr_ = "".join(self.visit(stmt) for stmt in node.fun)
        return self.generic_visit(node, expr_=expr_)

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
