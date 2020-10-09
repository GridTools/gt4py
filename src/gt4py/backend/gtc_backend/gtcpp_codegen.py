from eve import codegen

from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako

from gt4py.backend.gtc_backend.gtcppir import GTAccessor, GTApplyMethod, GTFunctor


class GTCppCodegen(codegen.TemplatedGenerator):

    GTExtent = as_fmt("extent<{i[0]},{i[1]},{j[0]},{j[1]},{k[0]},{k[1]}>")

    GTAccessor = as_fmt("using {name} = {intent}_accessor<{id}, {extent}>")

    GTParamList = as_mako(
        """
    ${ '\\n'.join(accessors) }
    using param_list = gridtools::stencil::make_param_list<${ ','.join(a.name for a in _this_node.accessors)}>;
    """
    )

    GTFunctor_template = as_mako(
        """
    struct ${ name } {
        ${param_list}

        ${ '\\n'.join(applies) }
    };
    """
    )

    GTApplyMethod_template = as_mako(
        """
    template<typename Evaluation> // TODO interval
    GT_FUNCTION static void apply(Evaluation eval) {
        ${ '\\n'.join(body) }
    }
    """
    )

    AssignStmt_template = "{left} = {right};"

    AccessorRef_template = "eval({name}({offset}))"

    Offset_template = "{i}, {j}, {k}"

    BinaryOp_template = "{left} {op} {right}"

    Literal_template = "{value}"  # TODO cast

    @classmethod
    def apply(cls, root, **kwargs) -> str:
        generated_code = super().apply(root, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code
