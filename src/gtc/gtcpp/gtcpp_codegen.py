# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from eve import Node, codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from eve.concepts import LeafNode
from gtc.common import BuiltInLiteral, DataType, LoopOrder, NativeFunction, UnaryOperator
from gtc.gtcpp import gtcpp


def _offset_limit(root: Node) -> int:
    return (
        root.iter_tree()
        .if_isinstance(gtcpp.GTLevel)
        .getattr("offset")
        .reduce(lambda state, cur: max(state, abs(cur)), init=0)
    ) + 1


class GTCppCodegen(codegen.TemplatedGenerator):

    GTExtent = as_fmt("extent<{i[0]},{i[1]},{j[0]},{j[1]},{k[0]},{k[1]}>")

    GTAccessor = as_fmt("using {name} = {intent}_accessor<{id}, {extent}, {ndim}>;")

    GTParamList = as_mako(
        """${ '\\n'.join(accessors) }

        using param_list = make_param_list<${ ','.join(a.name for a in _this_node.accessors)}>;
        """
    )

    GTFunctor = as_mako(
        """struct ${ name } {
        ${param_list}

        ${ '\\n'.join(applies) }
    };
    """
    )

    GTLevel = as_fmt("gridtools::stencil::core::level<{splitter}, {offset}, {offset_limit}>")

    GTInterval = as_fmt("gridtools::stencil::core::interval<{from_level}, {to_level}>")

    LocalVarDecl = as_fmt("{dtype} {name};")

    GTApplyMethod = as_mako(
        """
    template<typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, ${interval}) {
        ${ ' '.join(local_variables) }
        ${ '\\n'.join(body) }
    }
    """
    )

    AssignStmt = as_fmt("{left} = {right};")

    AccessorRef = as_fmt("eval({name}({', '.join([offset, *data_index])}))")

    ScalarAccess = as_fmt("{name}")

    CartesianOffset = as_fmt("{i}, {j}, {k}")

    BinaryOp = as_fmt("({left} {op} {right})")

    UnaryOp = as_fmt("({op}{expr})")

    TernaryOp = as_fmt("({cond} ? {true_expr} : {false_expr})")

    Cast = as_fmt("static_cast<{dtype}>({expr})")

    def visit_BuiltInLiteral(self, builtin: BuiltInLiteral, **kwargs: Any) -> str:
        if builtin == BuiltInLiteral.TRUE:
            return "true"
        elif builtin == BuiltInLiteral.FALSE:
            return "false"
        raise NotImplementedError("Not implemented BuiltInLiteral encountered.")

    Literal = as_mako("static_cast<${dtype}>(${value})")

    def visit_NativeFunction(self, func: NativeFunction, **kwargs: Any) -> str:
        try:
            return {
                NativeFunction.ABS: "std::abs",
                NativeFunction.MIN: "std::min",
                NativeFunction.MAX: "std::max",
                NativeFunction.MOD: "std::fmod",
                NativeFunction.SIN: "std::sin",
                NativeFunction.COS: "std::cos",
                NativeFunction.TAN: "std::tan",
                NativeFunction.ARCSIN: "std::asin",
                NativeFunction.ARCCOS: "std::acos",
                NativeFunction.ARCTAN: "std::atan",
                NativeFunction.SQRT: "std::sqrt",
                NativeFunction.POW: "std::pow",
                NativeFunction.EXP: "std::exp",
                NativeFunction.LOG: "std::log",
                NativeFunction.ISFINITE: "std::isfinite",
                NativeFunction.ISINF: "std::isinf",
                NativeFunction.ISNAN: "std::isnan",
                NativeFunction.FLOOR: "std::floor",
                NativeFunction.CEIL: "std::ceil",
                NativeFunction.TRUNC: "std::trunc",
            }[func]
        except KeyError as error:
            raise NotImplementedError(
                f"Not implemented NativeFunction '{func}' encountered."
            ) from error

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    DATA_TYPE_TO_CODE = {
        DataType.BOOL: "bool",
        DataType.INT8: "std::int8_t",
        DataType.INT16: "std::int16_t",
        DataType.INT32: "std::int32_t",
        DataType.INT64: "std::int64_t",
        DataType.FLOAT32: "float",
        DataType.FLOAT64: "double",
    }

    def visit_DataType(self, dtype: DataType, **kwargs: Any) -> str:
        try:
            return self.DATA_TYPE_TO_CODE[dtype]
        except KeyError as error:
            raise NotImplementedError(
                f"Not implemented DataType '{dtype.name}' encountered."
            ) from error

    UNARY_OPERATOR_TO_CODE = {
        UnaryOperator.NOT: "!",
        UnaryOperator.NEG: "-",
        UnaryOperator.POS: "+",
    }

    UnaryOp = as_fmt("({_this_generator.UNARY_OPERATOR_TO_CODE[_this_node.op]}{expr})")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    ApiParamDecl = as_fmt("{name}")

    GTStage = as_mako(".stage(${functor}(), ${','.join(args)})")

    GTMultiStage = as_mako("execute_${ loop_order }()${''.join(caches)}${''.join(stages)}")

    IJCache = as_fmt(".ij_cached({name})")
    KCache = as_mako(
        ".k_cached(${'cache_io_policy::fill(), ' if _this_node.fill else ''}${'cache_io_policy::flush(), ' if _this_node.flush else ''}${name})"
    )

    def visit_LoopOrder(self, looporder: LoopOrder, **kwargs: Any) -> str:
        return {
            LoopOrder.PARALLEL: "parallel",
            LoopOrder.FORWARD: "forward",
            LoopOrder.BACKWARD: "backward",
        }[looporder]

    Temporary = as_fmt("GT_DECLARE_TMP({dtype}, {name});")

    IfStmt = as_mako(
        """if(${cond}) ${true_branch}
        %if _this_node.false_branch:
            else ${false_branch}
        %endif
        """
    )

    BlockStmt = as_mako("{${''.join(body)}}")

    def visit_GTComputationCall(
        self, node: gtcpp.GTComputationCall, **kwargs: Any
    ) -> Union[str, Collection[str]]:
        computation_name = type(node).__name__ + str(id(node))
        return self.generic_visit(node, computation_name=computation_name, **kwargs)

    GTComputationCall = as_mako(
        """
        %if len(multi_stages) > 0 and len(arguments) > 0:
        {
            auto grid = make_grid(domain[0], domain[1], axis<1,
                axis_config::offset_limit<${offset_limit}>>{domain[2]});

            auto ${ computation_name } = [](${ ','.join('auto ' + a for a in arguments) }) {

                ${ '\\n'.join(temporaries) }
                return multi_pass(${ ','.join(multi_stages) });
            };

            run(${computation_name}, ${gt_backend_t}<>{}, grid, ${','.join(f"std::forward<decltype({arg})>({arg})" for arg in arguments)});
        }
        %endif
        """
    )

    Program = as_mako(
        """
        #include <gridtools/stencil/${gt_backend_t}.hpp>
        #include <gridtools/stencil/cartesian.hpp>

        namespace ${ name }_impl_{
            using Domain = std::array<gridtools::uint_t, 3>;
            using namespace gridtools::stencil;
            using namespace gridtools::stencil::cartesian;

            ${'\\n'.join(functors)}

            auto ${name}(Domain domain){
                return [domain](${ ','.join( 'auto&& ' + p for p in parameters)}){
                    ${gt_computation}
                };
            }
        }

        auto ${name}(${name}_impl_::Domain domain){
            return ${name}_impl_::${name}(domain);
        }
        """
    )

    @classmethod
    def apply(cls, root: LeafNode, **kwargs: Any) -> str:
        if not isinstance(root, gtcpp.Program):
            raise ValueError("apply() requires gtcpp.Progam root node")
        if "gt_backend_t" not in kwargs:
            raise TypeError("apply() missing 1 required keyword-only argument: 'gt_backend_t'")
        generated_code = super().apply(root, offset_limit=_offset_limit(root), **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code
