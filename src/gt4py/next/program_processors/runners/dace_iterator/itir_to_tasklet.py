import gt4py.eve as eve
import gt4py.eve.codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next.iterator import ir as itir


class CPPTaskletCodegen(eve.codegen.TemplatedGenerator):
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

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition):
        raise NotImplementedError()

    def visit_SymRef(self, node: itir.SymRef):
        return str(node.id)

    Sym = as_fmt("auto {id}")

    def _visit_deref(self, node: itir.FunCall):
        return "1"

    def _visit_bin_op_builtin(self, node: itir.FunCall):
        return CPPTaskletCodegen._builtins_mapping[str(node.fun.id)]

    def visit_FunCall(self, node: itir.FunCall):
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        if isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in CPPTaskletCodegen._builtins_mapping:
                function = self._visit_bin_op_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def visit_Lambda(self, node: itir.Lambda):
        params = ", ".join(self.visit(node.params))
        expr = self.visit(node.expr)

        return f"[]({params}){{return {expr};}}"


class PythonTaskletCodegen(eve.codegen.TemplatedGenerator):
    _builtins_mapping = {
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
        #     "multiplies": "std::multiplies{}",
        #     "divides": "std::divides{}",
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

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition):
        raise NotImplementedError()

    def visit_SymRef(self, node: itir.SymRef):
        return str(node.id)

    Sym = as_fmt("auto {id}")

    def _visit_deref(self, node: itir.FunCall):
        if isinstance(node.args[0], itir.SymRef):
            return f"{node.args[0].id}[i_IDim]"
        sym_ref, index = self.visit(node.args[0])
        return f"{sym_ref}[{index}]"

    def _visit_shift(self, node: itir.FunCall) -> tuple[str, tuple[str, ...]]:
        if isinstance(node.args[0], itir.SymRef):
            return self.visit(node.args[0]), ("i_IDim",)

        raise NotImplementedError()

    def _visit_bin_op_builtin(self, node: itir.FunCall):
        fmt = PythonTaskletCodegen._builtins_mapping[str(node.fun.id)]
        args = self.visit(node.args)
        return fmt.format(*args)

    def visit_FunCall(self, node: itir.FunCall):
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        if isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in PythonTaskletCodegen._builtins_mapping:
                return self._visit_bin_op_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def visit_Lambda(self, node: itir.Lambda):
        return self.visit(node.expr)
