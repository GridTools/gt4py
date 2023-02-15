import gt4py.eve as eve
import gt4py.eve.codegen
from gt4py.next.iterator import ir as itir
from typing import Any
from gt4py.next.common import Dimension


_BUILTINS_MAPPING = {
    #     "abs": "abs",
    #     "sin": "std::sin",
    #     "cos": "std::cos",
    #     "tan": "std::tan",
    #     "arcsin": "std::asin",
    #     "arccos": "std::acos",
    #     "arctan": "std::atan",
    #     "sinh": "std::sinh",
    #     "cosh": "std::cosh",
    #     "tanh": "std::tanh",
    #     "arcsinh": "std::asinh",
    #     "arccosh": "std::acosh",
    #     "arctanh": "std::atanh",
    #     "sqrt": "std::sqrt",
    #     "exp": "std::exp",
    #     "log": "std::log",
    #     "gamma": "std::tgamma",
    #     "cbrt": "std::cbrt",
    #     "isfinite": "std::isfinite",
    #     "isinf": "std::isinf",
    #     "isnan": "std::isnan",
    #     "floor": "std::floor",
    #     "ceil": "std::ceil",
    #     "trunc": "std::trunc",
    #     "minimum": "std::min",
    #     "maximum": "std::max",
    #     "fmod": "std::fmod",
    #     "power": "std::pow",
    #     "float": "double",
    #     "float32": "float",
    #     "float64": "double",
    #     "int": "long",
    #     "int32": "std::int32_t",
    #     "int64": "std::int64_t",
    #     "bool": "bool",
    "plus": "({} + {})",
    "minus": "({} - {})",
    "multiplies": "({} * {})",
    "divides": "({} / {})",
    #     "eq": "std::equal_to{}",
    #     "not_eq": "std::not_equal_to{}",
    #     "less": "std::less{}",
    #     "less_equal": "std::less_equal{}",
    #     "greater": "std::greater{}",
    #     "greater_equal": "std::greater_equal{}",
    #     "and_": "std::logical_and{}",
    #     "or_": "std::logical_or{}",
    #     "xor_": "std::bit_xor{}",
    #     "mod": "std::modulus{}",
    #     "not_": "std::logical_not{}",
}


class PythonTaskletCodegen(eve.codegen.TemplatedGenerator):
    offset_provider: dict[str, Any]

    def __init__(self, offset_provider: dict[str, Any]):
        self.offset_provider = offset_provider

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition):
        raise ValueError("Can only lower expressions, not whole functions.")

    def visit_Lambda(self, node: itir.Lambda):
        raise ValueError("Lambdas are not supported.")

    def visit_SymRef(self, node: itir.SymRef):
        return str(node.id)

    def visit_Literal(self, node: itir.Literal):
        return str(node.value)

    def visit_FunCall(self, node: itir.FunCall):
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        if isinstance(node.fun, itir.FunCall) and node.fun.fun.id == "shift":
            offset = node.fun.args[0]
            assert isinstance(offset, itir.OffsetLiteral)
            offset_name = offset.value
            if offset_name not in self.offset_provider:
                raise ValueError(f"offset provider for `{offset_name}` is missing")
            offset_provider = self.offset_provider[offset_name]
            if isinstance(offset_provider, Dimension):
                return self._visit_shift(node)
            else:
                return self._visit_indirect_addressing(node)

        if isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in _BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def _visit_iterator_sym(self, node: itir.SymRef):
        return node.id, ("i_IDim",)

    def _visit_deref(self, node: itir.FunCall):
        iterator = node.args[0]
        if isinstance(iterator, itir.SymRef):
            sym, index = self._visit_iterator_sym(iterator)
        else:
            sym, index = self.visit(node.args[0])
        return f"{sym}[{', '.join(index)}]"

    def _visit_shift(self, node: itir.FunCall) -> tuple[str, tuple[str, ...]]:
        iterator = node.args[0]
        if isinstance(iterator, itir.SymRef):
            sym, index = self._visit_iterator_sym(iterator)
        else:
            sym, index = self.visit(iterator)
        amount = self.visit(node.fun.args[1])

        shifted_axis = 0  # TODO: compute actual index

        offseted_index = tuple(
            value if axis != shifted_axis else f"({value} + {amount})"
            for axis, value in enumerate(index)
        )

        return sym, offseted_index

    def _visit_indirect_addressing(self, node: itir.FunCall):
        iterator = node.args[0]
        if isinstance(iterator, itir.SymRef):
            sym, index = self._visit_iterator_sym(iterator)
        else:
            sym, index = self.visit(iterator)

        offset = node.fun.args[0].value
        element = self.visit(node.fun.args[1])

        shifted_axis = 0  # TODO: compute actual index

        offseted_index = tuple(
            value if axis != shifted_axis else f"__connectivity_{offset}_full[{value}, {element}]"
            for axis, value in enumerate(index)
        )

        return sym, offseted_index

    def _visit_numeric_builtin(self, node: itir.FunCall):
        fmt = _BUILTINS_MAPPING[str(node.fun.id)]
        args = self.visit(node.args)
        return fmt.format(*args)


def closure_to_tasklet(node: itir.StencilClosure, offset_provider: dict[str, Any]) -> str:
    if isinstance(node.stencil, itir.Lambda):
        return PythonTaskletCodegen(offset_provider).visit(node.stencil.expr)
    elif isinstance(node.stencil, itir.SymRef):
        raise NotImplementedError()
    raise ValueError("invalid argument?")
