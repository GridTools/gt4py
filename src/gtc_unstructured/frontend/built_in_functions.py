from numbers import Number
from typing import List, Sequence, Tuple

from .built_in_types import LocalField


__all__ = []


class Method:
    name: str
    sig: Tuple[Sequence[type], type]

    def __init__(self, name: str, sig: Tuple[Sequence[type], type]):
        self.name = name
        self.sig = sig

    def applicable(self, arg_sig: Sequence[type]):
        return all(issubclass(arg, method_arg) for arg, method_arg in zip(arg_sig, self.sig[0]))

    def __repr__(self):
        arg_sig_str = ", ".join(
            f"arg_{i}: {arg_type.__module__}.{arg_type.__name__}"
            for i, arg_type in enumerate(self.sig[0])
        )
        return f"{self.name}({arg_sig_str}) -> {self.sig[1].__module__}.{self.sig[1].__name__}"


class BuiltInFunction:
    name: str
    methods: List[Method]

    def __init__(self, name):
        self.name = name
        self.methods = []

    def find(self, *arg_sig: type):
        applicable_methods = [method for method in self.methods if method.applicable(arg_sig)]
        if len(applicable_methods) != 1:
            if len(applicable_methods) > 1:
                msg = f"Multiple methods for function {self.name} match signature {arg_sig}.\n"
            else:
                msg = f"No method for function {self.name} matches signature {arg_sig}.\n"
            msg = msg + "Available methods: \n"
            for method in self.methods:
                msg = msg + "  " + str(method) + "\n"
            raise TypeError(msg)
        return applicable_methods[0]


def declare_built_in_function(name, sig):
    """Given the `name` of a built in function define a method with the given signature.

    The signature is a tuple of a tuple containing the argument types and the return type, e.g. ((Number,), Number)
    for a method taking a number as an argument and returning a number.
    """
    if name not in globals():
        globals()[name] = BuiltInFunction(name)
        __all__.append(name)

    method = Method(name, sig)
    globals()[name].methods.append(method)
    return method


_neighbor_reductions = []
for reduction_function in ["sum", "product", "min", "max"]:
    _neighbor_reductions.append(
        declare_built_in_function(reduction_function, ((LocalField,), Number))
    )

# TODO(workshop): add other native functions, log etc
# TODO(tehrengruber): varargs
_native_functions = [
    declare_built_in_function("max", ((Number, Number), Number)),
    declare_built_in_function("min", ((Number, Number), Number)),
    declare_built_in_function("mod", ((Number, Number), Number)),
    declare_built_in_function("sqrt", ((Number,), Number)),
]
