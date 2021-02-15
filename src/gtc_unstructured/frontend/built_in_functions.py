from typing import List, Tuple, Sequence
from numbers import Number
from .built_in_types import LocalField

__all__ = []

class Method:
    name: str
    signature: Tuple[Sequence[type], type]

    def __init__(self, name: str, sig: Tuple[Sequence[type], type]):
        self.name = name
        self.sig = sig

    def applicable(self, arg_sig: Sequence[type]):
        return all(issubclass(arg, method_arg) for arg, method_arg in zip(arg_sig, self.sig[0]))

class BuiltInFunction:
    name: str
    methods: List[Method]

    def __init__(self, name):
        self.name = name
        self.methods = []

def declare_built_in_function(name, sig):
    """Given the `name` of a built in function define a method with the given signature.

    The signature is a tuple of a tuple containing the argument types and the return type, e.g. ((Number,), Number)
    for a method taking a number as an argument and returning a number.
    """
    if not name in globals():
        globals()[name] = BuiltInFunction(name)
        __all__.append(name)

    globals()[name].methods.append(Method(name, sig))

neighbor_reductions = ["sum", "product", "min", "max"]  # list of methods for reductions on neighbors
for reduction_function in neighbor_reductions:
    declare_built_in_function(reduction_function, ((LocalField,), Number))

native_reductions = ["min", "max"]  # list of methods for native reductions
for reduction_function in native_reductions:
    # todo(tehrengruber): varargs
    declare_built_in_function(reduction_function, ((Number, Number), Number))