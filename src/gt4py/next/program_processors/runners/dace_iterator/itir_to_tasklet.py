import dace

import gt4py.eve as eve
import gt4py.eve.codegen
from gt4py.next.iterator import ir as itir
from typing import Any
from gt4py.next.common import Dimension
from gt4py.next.iterator.embedded import NeighborTableOffsetProvider


_BUILTINS_MAPPING = {
    "abs": "abs({})",
    "sin": "math.sin({})",
    "cos": "math.cos({})",
    "tan": "math.tan({})",
    "arcsin": "math.asin({})",
    "arccos": "math.acos({})",
    "arctan": "math.atan({})",
    "sinh": "math.sinh({})",
    "cosh": "math.cosh({})",
    "tanh": "math.tanh({})",
    "arcsinh": "math.asinh({})",
    "arccosh": "math.acosh({})",
    "arctanh": "math.atanh({})",
    "sqrt": "math.sqrt({})",
    "exp": "math.exp({})",
    "log": "math.log({})",
    "gamma": "math.gamma({})",
    "cbrt": "math.cbrt({})",
    "isfinite": "math.isfinite({})",
    "isinf": "math.isinf({})",
    "isnan": "math.isnan({})",
    "floor": "math.floor({})",
    "ceil": "math.ceil({})",
    "trunc": "math.trunc({})",
    "minimum": "min({}, {})",
    "maximum": "max({}, {})",
    "fmod": "math.fmod({}, {})",
    "power": "math.pow({}, {})",
    "float": "dace.float64({})",
    "float32": "dace.float32({})",
    "float64": "dace.float64({})",
    #     "int": "long",
    "int32": "dace.int32({})",
    "int64": "dace.int64({})",
    "bool": "dace.bool_({})",
    "plus": "({} + {})",
    "minus": "({} - {})",
    "multiplies": "({} * {})",
    "divides": "({} / {})",
    "eq": "({} == {})",
    "not_eq": "({} != {})",
    "less": "({} < {})",
    "less_equal": "({} <= {})",
    "greater": "({} > {})",
    "greater_equal": "({} >= {})",
    "and_": "({} & {})",
    "or_": "({} | {})",
    "xor_": "({} ^ {})",
    "mod": "({} % {})",
    "not_": "~{}",
}


class PythonTaskletCodegen(eve.codegen.TemplatedGenerator):
    sdfg: dace.SDFG
    offset_provider: dict[str, Any]
    domain: dict[str, str]

    def __init__(self, offset_provider: dict[str, Any], domain: dict[str, str]):
        self.offset_provider = offset_provider
        self.domain = domain

    def visit_FunctionDefinition(self, node: itir.FunctionDefinition, **kwargs):
        raise ValueError("Can only lower expressions, not whole functions.")

    def visit_Lambda(self, node: itir.Lambda, **kwargs):
        raise ValueError("Lambdas are not supported.")

    def visit_SymRef(self, node: itir.SymRef, *, hack_is_iterator=False, **kwargs) -> Any:
        if hack_is_iterator:
            index = self.domain
            return str(node.id), index
        return str(node.id)

    def visit_Literal(self, node: itir.Literal, **kwargs) -> str:
        return str(node.value)

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> Any:
        if isinstance(node.fun, itir.SymRef) and node.fun.id == "deref":
            return self._visit_deref(node)
        elif isinstance(node.fun, itir.FunCall) and node.fun.fun.id == "shift":
            offset = node.fun.args[0]
            assert isinstance(offset, itir.OffsetLiteral)
            offset_name = offset.value
            if offset_name not in self.offset_provider:
                raise ValueError(f"offset provider for `{offset_name}` is missing")
            offset_provider = self.offset_provider[offset_name]
            if isinstance(offset_provider, Dimension):
                return self._visit_direct_addressing(node)
            else:
                return self._visit_indirect_addressing(node)
        elif isinstance(node.fun, itir.SymRef):
            if str(node.fun.id) in _BUILTINS_MAPPING:
                return self._visit_numeric_builtin(node)
            else:
                raise NotImplementedError()
        else:
            function = self.visit(node.fun)
        args = ", ".join(self.visit(node.args))
        return f"{function}({args})"

    def _visit_deref(self, node: itir.FunCall, **kwargs):
        iterator = node.args[0]
        sym, index = self.visit(iterator, hack_is_iterator=True)
        flat_index = index.items()
        flat_index = sorted(flat_index, key=lambda x: x[0])
        flat_index = [x[1] for x in flat_index]
        return f"{sym}[{', '.join(flat_index)}]"

    def _visit_direct_addressing(self, node: itir.FunCall, **kwargs) -> tuple[str, dict[str, str]]:
        iterator = node.args[0]
        sym, index = self.visit(iterator)
        axis = self.visit(node.fun.args[0])
        amount = self.visit(node.fun.args[1])

        shifted_axis = 0  # TODO: compute actual index

        offseted_index = tuple(
            value if axis != shifted_axis else f"({value} + {amount})"
            for axis, value in enumerate(index)
        )

        return sym, offseted_index

    def _visit_indirect_addressing(self, node: itir.FunCall, **kwargs):
        iterator = node.args[0]
        sym, index = self.visit(iterator, hack_is_iterator=True)

        offset: str = node.fun.args[0].value
        element: str = self.visit(node.fun.args[1])

        table: NeighborTableOffsetProvider = self.offset_provider[offset]
        shifted_axis = table.origin_axis.value
        target_axis = table.neighbor_axis.value

        value = index[shifted_axis]
        new_value = f"__connectivity_{offset}_full[{value}, {element}]"

        new_index = {
            **{axis: value for axis, value in index.items() if axis != shifted_axis},
            target_axis: new_value,
        }
        return sym, new_index

    def _visit_numeric_builtin(self, node: itir.FunCall, **kwargs):
        fmt = _BUILTINS_MAPPING[str(node.fun.id)]
        args = self.visit(node.args)
        return fmt.format(*args)


def closure_to_tasklet(
    node: itir.StencilClosure, offset_provider: dict[str, Any], domain: dict[str, str]
) -> str:
    if isinstance(node.stencil, itir.Lambda):
        return PythonTaskletCodegen(offset_provider, domain).visit(node.stencil.expr)
    elif isinstance(node.stencil, itir.SymRef):
        raise NotImplementedError()
    raise ValueError("invalid argument?")
