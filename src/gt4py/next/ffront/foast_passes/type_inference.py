from gt4py.next.ffront import field_operator_ast as foast
from gt4py import eve


class TypeInferencePass(eve.NodeTranslator):
    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs):
        ...

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        ...