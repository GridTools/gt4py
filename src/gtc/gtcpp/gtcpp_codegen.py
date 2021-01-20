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
    )


class GTCppCodegen(codegen.TemplatedGenerator):

    GTExtent = as_fmt("extent<{i[0]},{i[1]},{j[0]},{j[1]},{k[0]},{k[1]}>")

    GTAccessor = as_fmt("using {name} = {intent}_accessor<{id}, {extent}>;")

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

    GTApplyMethod = as_mako(
        """
    template<typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, ${interval}) {
        ${ '\\n'.join(body) }
    }
    """
    )

    AssignStmt = as_fmt("{left} = {right};")

    AccessorRef = as_fmt("eval({name}({offset}))")

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
        if func == NativeFunction.SQRT:
            return "gridtools::math::sqrt"
        elif func == NativeFunction.MIN:
            return "gridtools::math::min"
        elif func == NativeFunction.MAX:
            return "gridtools::math::max"
        raise NotImplementedError("Not implemented NativeFunction encountered.")

    NativeFuncCall = as_mako("${func}(${','.join(args)})")

    def visit_DataType(self, dtype: DataType, **kwargs: Any) -> str:
        if dtype == DataType.INT64:
            return "long long"
        elif dtype == DataType.FLOAT64:
            return "double"
        elif dtype == DataType.FLOAT32:
            return "float"
        elif dtype == DataType.BOOL:
            return "bool"
        raise NotImplementedError("Not implemented NativeFunction encountered.")

    def visit_UnaryOperator(self, op: UnaryOperator, **kwargs: Any) -> str:
        if op == UnaryOperator.NOT:
            return "!"
        elif op == UnaryOperator.NEG:
            return "-"
        elif op == UnaryOperator.POS:
            return "+"
        raise NotImplementedError("Not implemented UnaryOperator encountered.")

    Arg = as_fmt("{name}")

    Param = as_fmt("{name}")

    ApiParamDecl = as_fmt("{name}")

    GTStage = as_mako(".stage(${functor}(), ${','.join(args)})")

    GTMultiStage = as_mako("execute_${ loop_order }()${''.join(caches)}${''.join(stages)}")

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
        return self.generic_visit(node, computation_name=node.id_, **kwargs)

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

            run(${computation_name}, cpu_ifirst<>{} /* TODO */, grid, ${','.join(arguments)});
        }
        %endif
        """
    )

    Program = as_mako(
        """#include <gridtools/stencil/cpu_ifirst.hpp>
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
        generated_code = super().apply(root, offset_limit=_offset_limit(root), **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code
