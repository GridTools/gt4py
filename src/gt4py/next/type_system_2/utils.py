import itertools
from collections.abc import Sequence
from . import types as ts


def flatten_tuples(ty: ts.Type) -> list[ts.Type]:
    if isinstance(ty, ts.TupleType):
        return list(itertools.chain(*(flatten_tuples(element) for element in ty.elements)))
    return [ty]


def unflatten_tuples(tys: Sequence[ts.Type], structure: ts.Type) -> ts.Type:
    def helper(tys: Sequence[ts.Type], structure: ts.Type) -> tuple[ts.Type, Sequence[ts.Type]]:
        if isinstance(structure, ts.TupleType):
            remaining = tys
            elements: list[ts.Type] = []
            for se in structure.elements:
                e, remaining = helper(remaining, se)
                elements.append(e)
            return ts.TupleType(elements), remaining
        if tys:
            return tys[0], tys[1:]
        raise ValueError("not enough values in for each element of the structure")
    new_ty, remaining = helper(tys, structure)
    if remaining:
        raise ValueError("too many values for structure")
    return new_ty


def link_params_to_args(
        parameters: Sequence[ts.FunctionParameter],
        arguments: Sequence[ts.FunctionArgument]
) -> dict[int, int]:
    name_to_index = {param.name: idx for idx, param in enumerate(parameters) if param.keyword}
    links: dict[int, int] = {}
    for arg_idx, arg in enumerate(arguments):
        if isinstance(arg.location, int):
            if arg.location >= len(parameters):
                raise ValueError("argument index out of range")
            if arg.location in links:
                raise ValueError(f"argument for '{parameters[arg.location].name}' supplied multiple times")
            links[arg.location] = arg_idx
        else:
            if arg.location not in name_to_index:
                raise ValueError(f"unexpected keyword argument '{arg.location}'")
            param_idx = name_to_index[arg.location]
            if param_idx in links:
                raise ValueError(f"argument for '{parameters[arg.location].name}' supplied multiple times")
            links[param_idx] = arg_idx
    return links
