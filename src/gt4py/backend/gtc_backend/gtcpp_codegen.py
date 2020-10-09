from eve import codegen

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from gt4py.backend.gtc_backend.common import LoopOrder

from gt4py.backend.gtc_backend.gtcppir import (
    GTAccessor,
    GTApplyMethod,
    GTComputation,
    GTFunctor,
    GTMultiStage,
)


class GTCppCodegen(codegen.TemplatedGenerator):

    GTExtent = as_fmt("extent<{i[0]},{i[1]},{j[0]},{j[1]},{k[0]},{k[1]}>")

    GTAccessor = as_fmt("using {name} = {intent}_accessor<{id}, {extent}>;")

    GTParamList = as_mako(
        """${ '\\n'.join(accessors) }

    using param_list = gridtools::stencil::make_param_list<${ ','.join(a.name for a in _this_node.accessors)}>;
    """
    )

    GTFunctor = as_mako(
        """struct ${ name } {
        ${param_list}

        ${ '\\n'.join(applies) }
    };
    """
    )

    GTApplyMethod = as_mako(
        """
    template<typename Evaluation> // TODO interval
    GT_FUNCTION static void apply(Evaluation eval) {
        ${ '\\n'.join(body) }
    }
    """
    )

    AssignStmt = as_fmt("{left} = {right};")

    AccessorRef = as_fmt("eval({name}({offset}))")

    Offset = as_fmt("{i}, {j}, {k}")

    BinaryOp = as_fmt("{left} {op} {right}")

    Literal = as_fmt("{value}")  # TODO cast

    ParamArg = as_fmt("{name}")

    GTStage = as_mako("stage(${functor}(), ${','.join(args)})")

    GTMultiStage = as_mako("gridtools::stencil::execute_${ loop_order }().${'.'.join(stages)}")

    def visit_LoopOrder(self, looporder: LoopOrder, **kwargs):  # TODO what's the pattern?
        if looporder == LoopOrder.PARALLEL:
            return "parallel"
        if looporder == LoopOrder.FORWARD:
            return "forward"
        if looporder == LoopOrder.BACKWARD:
            return "backward"

    GTComputation = as_mako(
        """{
            auto grid = gridtools::stencil::make_grid(domain[0], domain[1], domain[2]);

            auto ${ name } = [](${ ','.join('auto ' + p for p in parameters) }) {
            return gridtools::stencil::multi_pass(${ ','.join(multistages) });
            };

            st::run(${name}, gridtools::stencil::cpu_ifirst /* TODO */, grid, ${','.join(parameters)});
        }
        """
    )

    Computation = as_mako(
        """namespace ${ name }_impl_{
           ${'\\n'.join(functors)}

        auto ${name}(Domain domain) {
            return [domain](${ ','.join( 'auto&& ' + p for p in parameters)}){
                // allocate inter-gtcomputation-temporaries (in the future)

                // ctrl flow with calls to gt computations
                ${'\\n'.join(ctrl_flow_ast)}
            }
        }
        }

        auto ${name}(Domain domain){
            return ${name}_impl_::${name}(domain);
        }
        """
    )

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code
